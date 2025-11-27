""" This module is the main entry-point of the solution, providing the streamlit app to be used. """
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from video_analyst.ingestion import transcribe_audio_pipeline

load_dotenv()


PAGE_TITLE = "AI Video Analyst"
PAGE_ICON = "🎥"

# Setup local storage directories
DATA_DIR = Path("data")
VIDEO_DIR = DATA_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

def main():
    """ Main entry-point of the solution """
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.markdown("""
    **System Status:** Ready to ingest video, split audio, and extract precise timestamps.
    """)

    with st.sidebar:
        st.header("⚙️ Settings & Diagnostics")

        # Security Check
        if os.environ.get("OPENAI_API_KEY"):
            st.success("✅ OpenAI API Key detected")
        else:
            st.error("❌ OPENAI_API_KEY missing from .env")
            st.info("Please create a .env file in the root directory.")

        st.divider()
        st.markdown("### How it works")
        st.markdown("1. **Extracts** audio (FLAC)")
        st.markdown("2. **Splits** into 10m chunks")
        st.markdown("3. **Transcribes** via Whisper API")
        st.markdown("4. **Merges** timestamps")

    uploaded_file = st.file_uploader(
        "Upload a Video Lecture", 
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_file:
        video_path = VIDEO_DIR / uploaded_file.name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.video(str(video_path))

        if st.button("Analyze Video", type="primary"):
            if not os.environ.get("OPENAI_API_KEY"):
                st.error("Cannot proceed without API Key.")
                st.stop()

            try:
                transcript_data = transcribe_audio_pipeline(str(video_path))

                st.success("✅ Ingestion Complete!")

                # Store results in session state so they survive UI interactions
                st.session_state["transcript_data"] = transcript_data

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

    if "transcript_data" in st.session_state:
        data = st.session_state["transcript_data"]

        st.divider()
        st.subheader("📊 Analysis Results")

        tab1, tab2 = st.tabs(["📝 Full Transcript", "🔍 Debug: Word Timestamps"])

        with tab1:
            # Reconstruct the full text from segments
            full_text = "\n\n".join([seg["text"] for seg in data])
            st.text_area("Complete Text", full_text, height=400)

        with tab2:
            st.info("This view proves that we have granular word-level data for Phase 2.")

            # Show the structured data for the first few chunks
            for i, segment in enumerate(data):
                with st.expander(f"Chunk {i+1}: {len(segment['words'])} words"):
                    st.write(f"**Chunk Text:** {segment['text'][:100]}...")
                    st.json(segment['words'][:10])

if __name__ == "__main__":
    main()
