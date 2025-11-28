""" This module is the main entry-point of the solution, providing the streamlit app to be used. """
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from video_analyst.ingestion import transcribe_audio_pipeline
from video_analyst.segmentation import semantic_segmentation_pipeline

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
        st.header("⚙️ Settings")

        language = st.selectbox(
            "Audio Language",
            options=["auto", "en", "hu", "de", "fr", "es"],
            format_func=lambda x: "Auto-Detect" if x == "auto" else x.upper(),
            help="Forcing the language can improve accuracy for non-English content."
        )
        lang_code = None if language == "auto" else language

        st.divider()
        st.header("Diagnostics")
        if os.environ.get("OPENAI_API_KEY"):
            st.success("✅ OpenAI API Key detected")
        else:
            st.error("❌ OPENAI_API_KEY missing from .env")
            st.info("Please create a .env file in the root directory.")


        st.info(
            "**Pipeline Steps:**\n"
            "1. Extract Audio (FLAC)\n"
            "2. Split (10m chunks)\n"
            "3. Parallel Transcribe\n"
            "4. Merge & Checkpoint"
        )

    uploaded_file = st.file_uploader(
        "Upload a Video Lecture", 
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_file:
        video_path = VIDEO_DIR / uploaded_file.name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.video(str(video_path))

        if st.button("Step 1: Analyze Video (Transcribe)", type="primary"):
            if not os.environ.get("OPENAI_API_KEY"):
                st.error("Cannot proceed without API Key.")
                st.stop()

            try:
                transcript_data = transcribe_audio_pipeline(
                    str(video_path),
                    language=lang_code
                )

                st.success("✅ Ingestion Complete!")
                st.session_state["transcript_data"] = transcript_data

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

    if "transcript_data" in st.session_state:
        data = st.session_state["transcript_data"]

        st.divider()
        st.header("Step 2: Semantic Analysis")

        # We check if chapters exist to avoid re-running on simple UI clicks
        if "chapters" not in st.session_state:
            st.info("The transcript is ready. Click below to use AI to find logical chapters.")
            if st.button("🔮 Analyze Topics & Generate Chapters"):
                try:
                    chapters_structure = semantic_segmentation_pipeline(str(video_path), data)
                    st.session_state["chapters"] = chapters_structure
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        if "chapters" in st.session_state:
            video_struct = st.session_state["chapters"]

            st.subheader("📚 Video Chapters")

            for chap in video_struct.chapters:
                duration = chap.end_time - chap.start_time
                minutes = int(duration // 60)
                seconds = int(duration % 60)

                # Expandable card for each chapter
                with st.expander(f"**{chap.title}** ({minutes}m {seconds}s)"):
                    st.markdown(f"_{chap.summary}_")
                    st.markdown(f"**Keywords:** {', '.join(chap.topic_keywords)}")
                    st.caption(f"Starts at: {int(chap.start_time // 60)}:{int(chap.start_time % 60):02d}")

        # Debug Tabs
        st.divider()
        st.subheader("📊 Raw Data")

        tab1, tab2 = st.tabs(["📝 Full Transcript", "🔍 Word Timestamps"])

        with tab1:
            full_text = "\n\n".join([seg["text"] for seg in data])
            st.text_area("Complete Text", full_text, height=400)

        with tab2:
            st.info("Verifying word-level timestamps used for segmentation.")
            for i, segment in enumerate(data):
                with st.expander(f"Chunk {i+1} ({len(segment['words'])} words)"):
                    st.write(f"**Text:** {segment['text'][:150]}...")
                    st.json(segment['words'][:5])
if __name__ == "__main__":
    main()
