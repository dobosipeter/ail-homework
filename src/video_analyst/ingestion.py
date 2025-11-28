""" This module is responsible for processing the incoming video files. """
import os
import math
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import streamlit as st
from openai import OpenAI, APIConnectionError, RateLimitError, APIError
from pydub import AudioSegment

TRANSCRIPT_DIR = Path("data/transcripts")
CHECKPOINT_DIR = TRANSCRIPT_DIR / "checkpoints"
AUDIO_DIR = Path("data/audio")

for d in [TRANSCRIPT_DIR, CHECKPOINT_DIR, AUDIO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_audio(video_path: str, output_dir: Path = AUDIO_DIR) -> str:
    """
    Extracts audio from a video file and saves it as a lossless FLAC file.

    Args:
        video_path (str): The filesystem path to the input video file.
        output_dir (str): The directory where the extracted audio should be saved. 
                          Defaults to "data/audio".

    Returns:
        str: The full filesystem path to the generated .flac file.

    Raises:
        RuntimeError: If the FFmpeg subprocess returns a non-zero exit code.
    """

    """
    This function uses FFmpeg to extract the audio track, downmix it to mono,
    and resample it to 16kHz. We use FLAC (Free Lossless Audio Codec) instead 
    of MP3 to prevent compression artifacts that can degrade transcription accuracy.
    It includes a caching mechanism to skip extraction if the target file already exists.

    A 10-minute FLAC file at 16kHz mono is approximately 11MB, which fits 
    comfortably within OpenAI's 25MB file size limit.
    """

    video_path_obj = Path(video_path)
    audio_path = output_dir / f"{video_path_obj.stem}.flac"

    if audio_path.exists():
        logger.info("Audio already extracted: %s", audio_path)
        return str(audio_path)

    logger.info("Extracting audio from: %s", video_path)
    cmd = f'ffmpeg -y -i "{video_path}" -vn -ac 1 -ar 16000 -c:a flac "{audio_path}"'

    ret_code = os.system(cmd)

    if ret_code != 0:
        logger.error("FFmpeg failed with return code: %i", ret_code)
        raise RuntimeError(f"FFmpeg extraction failed. Command: {cmd}")

    return str(audio_path)

def split_audio(file_path: str, chunk_length_ms: int = 10 * 60 * 1000) -> list[str]:
    """
    Splits the audio file into smaller FLAC chunks to stay under OpenAI's 25MB API limit.

    Args:
        file_path (str): Path to the source audio file (expected .flac).
        chunk_length_ms (int): Duration of each chunk in milliseconds. 
                               Defaults to 600,000 ms (10 minutes).

    Returns:
        list[str]: A list of file paths to the generated audio chunks.

    Raises:
        RuntimeError: If the audio file loading fails.
    """

    """
    We default to 10-minute chunks because 10 mins of 16kHz Mono FLAC is 
    approximately 10-12MB. Checks if chunks already exist to avoid 
    redundant processing on resumed runs.
    """

    file_path_obj = Path(file_path)
    file_stem = file_path_obj.stem

    try:
        audio = AudioSegment.from_file(str(file_path), format="flac")
    except Exception as e:
        logger.error("Failed to load audio: %s", str(e))
        raise RuntimeError(f"Failed to load audio file {file_path}. Error: {e}") from e

    total_length_ms = len(audio)
    num_chunks = math.ceil(total_length_ms / chunk_length_ms)

    chunk_paths: list[str] = []
    missing_chunks = False

    for i in range(num_chunks):
        chunk_name = f"{file_stem}_part{i}.flac"
        chunk_path = file_path_obj.parent / chunk_name
        chunk_paths.append(str(chunk_path))
        if not chunk_path.exists():
            missing_chunks = True

    if not missing_chunks:
        logger.info("All %i audio chunks found. Skipping split.", num_chunks)
        return chunk_paths

    logger.info("Splitting audio into %i chunks.", num_chunks)

    for i in range(num_chunks):
        chunk_path = chunk_paths[i]

        if Path(chunk_path).exists():
            continue

        start_ms = i * chunk_length_ms
        end_ms = min((i + 1) * chunk_length_ms, total_length_ms)

        chunk = audio[start_ms:end_ms]
        chunk.export(chunk_path, format="flac")

    return chunk_paths


def _transcribe_single_chunk(
    client: OpenAI,
    chunk_path: str,
    chunk_index: int, 
    checkpoint_path: Path,
    language: str | None = None
) -> dict[str, Any]:
    """
    Transcribes a single audio chunk using the OpenAI Whisper API.

    This helper function handles the API call, automatic retries for transient 
    errors (rate limits, timeouts), and checkpoint saving. If a checkpoint 
    already exists for this chunk, it is loaded instead of calling the API.

    Args:
        client (OpenAI): The authenticated OpenAI client instance.
        chunk_path (str): Path to the audio chunk file.
        chunk_index (int): The zero-based index of this chunk in the sequence.
        checkpoint_path (Path): Path where the JSON result should be saved.
        language (str | None): ISO-639-1 language code (e.g., "en"). 
                                  If None, the model auto-detects language.

    Returns:
        dict[str, Any]: A dictionary containing the transcript segment data:
            - "chunk_index" (int): The index of the chunk.
            - "text" (str): The transcribed text.
            - "words" (List[Dict]): List of word objects with "word", "start", "end".

    Raises:
        APIConnectionError, RateLimitError, APIError: If API calls fail after retries.
    """

    if checkpoint_path.exists():
        logger.info("Chunk %s: Loaded from checkpoint.", chunk_index)
        with open(checkpoint_path, "r") as f:
            return json.load(f)

    logger.info("Chunk %s: Transcribing via API...", chunk_index)
    max_retries = 3

    for attempt in range(max_retries):
        logger.debug("Attempt %i of %i for chunk %i", attempt+1, max_retries, chunk_index)
        try:
            with open(chunk_path, "rb") as audio_file:
                kwargs = {
                    "model": "whisper-1",
                    "file": audio_file,
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["word"]
                }
                if language:
                    kwargs["language"] = language

                logger.debug("Sending transcription request for chunck: %s", chunk_index)
                response = client.audio.transcriptions.create(**kwargs)
                logger.info("Received transcription response for chunck: %s", chunk_index)

            logger.debug("Parsing chunk %i", chunk_index)
            chunk_data = {
                "chunk_index": chunk_index,
                "text": response.text,
                "words": []
            }

            for word_obj in response.words:
                # Handle potential differences in Pydantic vs Dict return types
                if isinstance(word_obj, dict):
                    w_data = word_obj
                else:
                    w_data = {
                        "word": word_obj.word,
                        "start": word_obj.start,
                        "end": word_obj.end
                    }
                chunk_data["words"].append(w_data)

            with open(checkpoint_path, "w") as f:
                logger.debug("Saving checkpoint chunck to %s", str(checkpoint_path))
                json.dump(chunk_data, f)

            return chunk_data

        except (APIConnectionError, APIError, RateLimitError) as e:
            logger.warning("Chunk %i: API Error %s. Retry %i/%i", chunk_index, str(e), attempt+1, max_retries)
            if attempt == max_retries - 1:
                logger.error("Max retries?")
                raise e
            time.sleep(2 * (attempt + 1))


@st.cache_data(show_spinner=False)
def transcribe_audio_pipeline(video_path: str, language: str | None = None) -> list[dict[str, Any]]:
    """
    Runs the full ingestion pipeline: Extract -> Split -> Transcribe -> Merge.
    This function uses Streamlit's caching to avoid re-processing the same video.
    If the process crashes, re-running it will skip already transcribed chunks.

    Args:
        video_path (str): The filesystem path to the input video file.
        language (str | None): ISO-639-1 language code (e.g., "en", "es"). 
                                If None, Whisper will auto-detect the language.

    Returns:
        list[dict[str, Any]]: A list of transcript segments. Each segment dictionary contains:
            - "chunk_index" (int): The index of the processing chunk (0, 1, 2...).
            - "text" (str): The full transcribed text of that chunk.
            - "words" (list[dict]): A list of word objects, where each word has:
                - "word" (str): The actual word spoken.
                - "start" (float): Start time in seconds (relative to the video start).
                - "end" (float): End time in seconds (relative to the video start).

    Raises:
        ValueError: If OPENAI_API_KEY is not set in environment variables.
        Exception: If any chunk fails to transcribe after retries.
    """

    """
    Orchestrates the full audio ingestion pipeline: Extraction, Chunking, and Transcription.

    This function manages the entire workflow including:
    1. Extracting audio from video.
    2. Splitting audio into chunks.
    3. Transcribing chunks in parallel using ThreadPoolExecutor.
    4. Managing resume capabilities via checkpoints.
    5. Assembling the final transcript with globally aligned timestamps.
    6. Cleaning up temporary files upon success.
    """

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API Key not found, have you set the OPENAI_API_KEY environment variable?")
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    client = OpenAI(api_key=api_key)

    video_stem = Path(video_path).stem
    final_cache_path = TRANSCRIPT_DIR / f"{video_stem}.json"

    if final_cache_path.exists():
        logger.info("Loading final transcript from disk: %s", final_cache_path)
        with open(final_cache_path, "r") as f:
            return json.load(f)

    with st.spinner("Step 1/4: Extracting audio from video..."):
        audio_path = extract_audio(video_path)

    with st.spinner("Step 2/4: Chunking audio..."):
        chunk_paths = split_audio(audio_path)

    total_chunks = len(chunk_paths)
    results_map = {}

    progress_bar = st.progress(0, text="Step 3/4: Starting parallel transcription...")

    # We use max_workers=3 to safely stay within OpenAI's concurrent request limits
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_index = {}

        for i, path in enumerate(chunk_paths):
            checkpoint = CHECKPOINT_DIR / f"{video_stem}_part{i}.json"
            # Submit task to the thread pool
            future = executor.submit(_transcribe_single_chunk, client, path, i, checkpoint, language)
            future_to_index[future] = i

        # Process futures as they complete
        completed_count = 0
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                data = future.result()
                results_map[idx] = data
                completed_count += 1

                # Update the UI
                pct = completed_count / total_chunks
                progress_bar.progress(pct, text=f"Step 3/4: Transcribed chunk {completed_count}/{total_chunks} ({int(pct*100)}%)")

            except Exception as e:
                logger.error("Chunk %i failed: %s", idx, str(e))
                st.error(f"Error processing chunk {idx}: {e}")
                raise e

    with st.spinner("Step 4/4: Assembling & Cleaning up..."):
        full_transcript = []
        global_time_offset = 0.0

        # Sort by index to ensure correct chronological order
        for i in range(total_chunks):
            chunk_data = results_map[i]

            # Create a new segment for the final list
            shifted_data = {
                "chunk_index": i,
                "text": chunk_data["text"],
                "words": []
            }

            # Shift raw timestamps by the global offset
            for w in chunk_data["words"]:
                shifted_data["words"].append({
                    "word": w["word"],
                    "start": w["start"] + global_time_offset,
                    "end": w["end"] + global_time_offset
                })

            full_transcript.append(shifted_data)

            # Calculate exact offset from audio file duration
            chunk_duration = len(AudioSegment.from_file(chunk_paths[i], format="flac")) / 1000.0
            global_time_offset += chunk_duration

        with open(final_cache_path, "w") as f:
            json.dump(full_transcript, f, indent=2)

        logger.info("Cleaning up temporary files...")

        for p in chunk_paths:
            if os.path.exists(p):
                os.remove(p)

        for i in range(total_chunks):
            cp = CHECKPOINT_DIR / f"{video_stem}_part{i}.json"
            if cp.exists():
                os.remove(cp)

    return full_transcript
