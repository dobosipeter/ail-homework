""" This module is responsible for segmenting the raw text we've received. """

import os
import json
import logging
from pathlib import Path
from typing import Any

import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field, model_validator

SEGMENTATION_DIR = Path("data/segmentations")
SEGMENTATION_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the strict schema we want the LLM to output, using Pydantic models
class Chapter(BaseModel):
    """
    Represents a single logical chapter or topic segment within the video.

    This model is used to structure the LLM's output, forcing it to identify 
    distinct topics with precise start and end times based on the injected 
    timestamp anchors. Strictly enforces that end_time is greater than start_time.

    Attributes:
        title (str): A concise and engaging title for the chapter.
        summary (str): A 2-3 sentence summary describing the key concepts discussed in this segment.
        start_time (float): The start time of the chapter in seconds. This should correspond 
                            to the nearest <t=X> anchor found in the transcript.
        end_time (float): The end time of the chapter in seconds.
        topic_keywords (list[str]): A list of 3-5 keywords or tags related to the topic 
                                    (e.g., ["backpropagation", "gradient descent", "loss function"]).
    """

    title: str = Field(..., description="A concise, engaging title for this chapter.")
    summary: str = Field(..., description="A 2-3 sentence summary of the topics discussed.")
    start_time: float = Field(..., description="The start time in seconds (derived from the nearest <t=X> tag).")
    end_time: float = Field(..., description="The end time in seconds.")
    topic_keywords: list[str] = Field(..., description="3-5 keywords related to the topic.")

    @model_validator(mode='after')
    def check_times(self) -> 'Chapter':
        """Sanity check to ensure time flows forward."""
        if self.end_time <= self.start_time:
            logger.warning("Invalid chapter times detected (%s -> %s). Auto-correcting.", self.start_time, self.end_time)
            if self.end_time == self.start_time:
                self.end_time += 1.0
            else:
                self.start_time, self.end_time = self.end_time, self.start_time
                logger.warning("Swapped, timestamps, current start: %s end: %s ", self.start_time, self.end_time)
        return self

class VideoStructure(BaseModel):
    """
    Represents the complete semantic structure of a video.

    This container model holds the list of all identified chapters. It is the 
    top-level object returned by the LLM during the segmentation phase.

    Attributes:
        chapters (list[Chapter]): An ordered list of Chapter objects that collectively 
                                  cover the entire video content.
    """
    chapters: list[Chapter] = Field(..., description="The list of logical chapters extracted from the transcript.")

def _inject_time_anchors(transcript_segments: list[dict[str, Any]], interval_sec: int = 30) -> str:
    """
    Transforms raw transcript segments into a single text block with injected timestamp anchors.

    This function flattens the hierarchical transcript data (chunks -> words) and 
    inserts a tag like `<t=120.5>` at specific time intervals. This helps the LLM 
    understand the temporal flow of the text and assign accurate start/end times 
    to the generated chapters.

    Args:
        transcript_segments (list[dict[str, Any]]): The list of transcript segments returned 
                                                    by the ingestion pipeline. Each segment 
                                                    must contain a "words" list with "start" timestamps.
        interval_sec (int): The minimum duration in seconds between two timestamp anchors. 
                            Defaults to 30 seconds.

    Returns:
        str: A single string containing the full transcript text with embedded timestamp tags.
    """
    full_text_with_anchors = []
    last_anchor_time = -interval_sec # Force anchor at start (0.0)

    # Flatten the list of words from all segments
    all_words = []
    for seg in transcript_segments:
        all_words.extend(seg["words"])

    current_sentence = []

    for word_obj in all_words:
        word = word_obj["word"]
        start = word_obj["start"]

        current_sentence.append(word)

        # Check if we need to insert an anchor
        if start - last_anchor_time >= interval_sec:
            # Join current sentence buffer
            full_text_with_anchors.append(" ".join(current_sentence))
            current_sentence = []

            # Add Anchor tag
            full_text_with_anchors.append(f"<t={start:.1f}>")
            last_anchor_time = start

    # Flush remaining words
    if current_sentence:
        full_text_with_anchors.append(" ".join(current_sentence))

    return " ".join(full_text_with_anchors)

@st.cache_data(show_spinner=False)
def semantic_segmentation_pipeline(video_path: str, transcript_data: list[dict[str, Any]]) -> VideoStructure:
    """
    Analyzes the transcript using an LLM to segment it into logical chapters.

    This function orchestrates the semantic segmentation process:
    1. Checks for a cached result to avoid redundant API calls.
    2. Preprocesses the transcript to inject timestamp anchors (`<t=X>`).
    3. Sends the prepared text to an LLM (GPT-4.1-mini) with a system prompt enforcing the `VideoStructure` schema.
    4. Validates and parses the LLM's JSON response into Pydantic objects.
    5. Caches the result to disk for future use.

    Args:
        video_path (str): The filesystem path to the original video file (used for cache key generation).
        transcript_data (list[dict[str, Any]]): The raw transcript data produced by the ingestion pipeline.

    Returns:
        VideoStructure: A structured object containing the list of identified chapters, 
                        each with a title, summary, and precise timestamps.

    Raises:
        ValueError: If the OPENAI_API_KEY environment variable is missing.
        Exception: If the LLM call fails or returns unparseable data.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found.")
    client = OpenAI(api_key=api_key)

    video_stem = Path(video_path).stem
    cache_path = SEGMENTATION_DIR / f"{video_stem}_chapters.json"

    if cache_path.exists():
        logger.info("Loading segmentation from cache: %s", cache_path)
        with open(cache_path, "r") as f:
            data = json.load(f)
            return VideoStructure(**data)

    with st.spinner("Preparing transcript for analysis..."):
        # We inject timestamps so the LLM knows WHERE topics change
        tagged_text = _inject_time_anchors(transcript_data, interval_sec=30)
        logger.info("Prepared text length: %d chars", len(tagged_text))

    with st.spinner("Model is analyzing content structure (using GPT-4.1-mini)..."):
        system_prompt = (
            "You are an expert video editor. Segment the transcript into logical 'Chapters'.\n"
            "The text has embedded timestamps like <t=12.5>. "
            "Use these tags to determine the EXACT start_time and end_time for each chapter.\n"
            "Rules:\n"
            "1. Chapter times MUST be sequential and non-overlapping.\n"
            "2. start_time must be strictly less than end_time.\n"
            "3. Cover the entire duration of the video.\n"
            "4. Titles should be descriptive educational topics."
        )

        try:
            logger.debug("Sending input to model.")
            completion = client.beta.chat.completions.parse(
                model="gpt-4.1-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Here is the transcript:\n\n{tagged_text}"}
                ],
                response_format=VideoStructure,
            )

            result: VideoStructure = completion.choices[0].message.parsed
            logger.debug("Received response from model.")

            # Basic sanity check on timestamps
            if not result.chapters:
                logger.warning("LLM returned 0 chapters!")

            logger.info("Generated %d chapters.", len(result.chapters))

            with open(cache_path, "w") as f:
                f.write(result.model_dump_json(indent=2))

            return result

        except Exception as e:
            logger.error("Segmentation failed: %s", e)
            st.error(f"Segmentation failed: {e}")
            raise e
