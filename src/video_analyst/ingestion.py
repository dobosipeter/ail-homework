""" This module is responsible for processing the incoming video files. """
import os
import math
from pathlib import Path
from typing import Any

import streamlit as st
from openai import OpenAI
from pydub import AudioSegment

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def extract_audio(video_path: str, output_dir: str = "data/audio") -> str:
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
    
    A 10-minute FLAC file at 16kHz mono is approximately 11MB, which fits 
    comfortably within OpenAI's 25MB file size limit.
    """

    video_path_obj = Path(video_path)
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    audio_path = output_dir_obj / f"{video_path_obj.stem}.flac"

    if audio_path.exists():
        return str(audio_path)

    cmd = f'ffmpeg -y -i "{video_path}" -vn -ac 1 -ar 16000 -c:a flac "{audio_path}"'

    ret_code = os.system(cmd)

    if ret_code != 0:
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
    approximately 10-12MB. 
    """

    file_path_obj = Path(file_path)

    try:
        audio = AudioSegment.from_file(str(file_path), format="flac")
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file_path}. Error: {e}") from e

    total_length_ms = len(audio)
    num_chunks = math.ceil(total_length_ms / chunk_length_ms)

    chunk_paths: list[str] = []

    for i in range(num_chunks):
        start_ms = i * chunk_length_ms
        end_ms = min((i + 1) * chunk_length_ms, total_length_ms)

        chunk = audio[start_ms:end_ms]

        chunk_filename = f"{file_path_obj.stem}_part{i}.flac"
        chunk_path = file_path_obj.parent / chunk_filename

        chunk.export(str(chunk_path), format="flac")
        chunk_paths.append(str(chunk_path))

    return chunk_paths

@st.cache_data(show_spinner=False)
def transcribe_audio_pipeline(video_path: str) -> list[dict[str, Any]]:
    """
    Runs the full ingestion pipeline: Extract -> Split -> Transcribe -> Merge.
    This function uses Streamlit's caching to avoid re-processing the same video.

    Args:
        video_path (str): The filesystem path to the input video file.

    Returns:
        List[Dict[str, Any]]: A list of transcript segments. Each segment dictionary contains:
            - "chunk_index" (int): The index of the processing chunk (0, 1, 2...).
            - "text" (str): The full transcribed text of that chunk.
            - "words" (List[Dict]): A list of word objects, where each word has:
                - "word" (str): The actual word spoken.
                - "start" (float): Start time in seconds (relative to the video start).
                - "end" (float): End time in seconds (relative to the video start).
    """

    with st.spinner("Step 1/3: Extracting audio from video..."):
        audio_path = extract_audio(video_path)

    with st.spinner("Step 2/3: Preparing audio for AI processing..."):
        chunk_paths = split_audio(audio_path)

    full_transcript = []
    global_time_offset = 0.0

    progress_text = "Step 3/3: Transcribing with OpenAI Whisper..."
    progress_bar = st.progress(0, text=progress_text)

    try:
        for i, chunk_path in enumerate(chunk_paths):

            with open(chunk_path, "rb") as audio_file:
                # API Call: We request 'verbose_json' to get the 'words' list
                # timestamp_granularities=['word'] ensures we get start/end for every word
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )

            chunk_text = response.text
            chunk_words = response.words

            segment_data = {
                "chunk_index": i,
                "text": chunk_text,
                "words": []
            }

            # Shift every word's timestamp by the global_time_offset
            for word_obj in chunk_words:
                # word_obj is typically a dict in the response list
                # We handle both dict and object access for safety
                w_word = word_obj["word"] if isinstance(word_obj, dict) else word_obj.word
                w_start = word_obj["start"] if isinstance(word_obj, dict) else word_obj.start
                w_end = word_obj["end"] if isinstance(word_obj, dict) else word_obj.end

                segment_data["words"].append({
                    "word": w_word,
                    "start": w_start + global_time_offset,
                    "end": w_end + global_time_offset
                })

            full_transcript.append(segment_data)

            # We calculate the precise duration of the chunk we just processed
            # to ensure the next chunk starts at exactly the right second.
            chunk_duration_sec = len(AudioSegment.from_file(chunk_path, format="flac")) / 1000.0
            global_time_offset += chunk_duration_sec

            progress_bar.progress((i + 1) / len(chunk_paths), text=progress_text)

    finally:
        for p in chunk_paths:
            if os.path.exists(p):
                os.remove(p)

    return full_transcript
