""" This module is the main entry-point of the solution, providing the streamlit app to be used. """

import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from video_analyst.ingestion import transcribe_audio_pipeline
from video_analyst.segmentation import semantic_segmentation_pipeline
from video_analyst.rag import build_vector_store, query_knowledge_base, list_collections

PAGE_TITLE = "AI Video Analyst"
PAGE_ICON = "🎥"

DATA_DIR = Path("data")
VIDEO_DIR = DATA_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

def main():
    """ Main entry-point of the solution """
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.markdown("""
    **System Status:** Ready.  
    Ingest videos, build course collections, and ask questions.
    """)

    with st.sidebar:
        st.header("⚙️ Settings")

        language = st.selectbox(
            "Audio Language",
            options=["auto", "en", "hu", "de", "fr", "es"],
            format_func=lambda x: "Auto-Detect" if x == "auto" else x.upper()
        )
        lang_code = None if language == "auto" else language

        st.divider()

        st.header("📚 Knowledge Base")
        kb_name = st.text_input(
            "Target Collection Name", 
            value="Default_Collection",
            help="Videos with the same Collection Name will be grouped together for searching."
        )

        st.divider()
        st.header("Diagnostics")
        if os.environ.get("OPENAI_API_KEY"):
            st.success("✅ OpenAI Key detected")
        else:
            st.error("❌ OpenAI Key missing")


    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file:
        video_path = VIDEO_DIR / uploaded_file.name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.video(str(video_path))

        if st.button("Step 1: Transcribe", type="primary"):
            if not os.environ.get("OPENAI_API_KEY"):
                st.error("Missing API Key")
                st.stop()
            try:
                transcript = transcribe_audio_pipeline(str(video_path), language=lang_code)
                st.session_state["transcript_data"] = transcript
                st.success("✅ Transcription Complete")
            except Exception as e:
                st.error(f"Error: {e}")

    if "transcript_data" in st.session_state:
        data = st.session_state["transcript_data"]

        # If we haven't segmented yet, show button
        if "chapters" not in st.session_state:
            st.info("Transcript ready. Analyze topics?")
            if st.button("Step 2: Generate Chapters"):
                try:
                    v_path_str = str(VIDEO_DIR / uploaded_file.name) if uploaded_file else "unknown_video"
                    chapters = semantic_segmentation_pipeline(v_path_str, data)
                    st.session_state["chapters"] = chapters
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        if "chapters" in st.session_state:
            chapters = st.session_state["chapters"]

            st.divider()
            st.subheader("📚 Video Chapters")

            for chap in chapters.chapters:
                duration = chap.end_time - chap.start_time
                minutes = int(duration // 60)
                seconds = int(duration % 60)

                with st.expander(f"**{chap.title}** ({minutes}m {seconds}s)"):
                    st.markdown(f"_{chap.summary}_")
                    st.markdown(f"**Keywords:** {', '.join(chap.topic_keywords)}")
                    st.caption(f"Starts at: {int(chap.start_time // 60)}:{int(chap.start_time % 60):02d}")

            st.divider()
            st.subheader(f"Step 3: Add to Collection '{kb_name}'")

            if st.button("💾 Save to Knowledge Base"):
                try:
                    v_path_str = str(VIDEO_DIR / uploaded_file.name) if uploaded_file else "unknown_video"
                    build_vector_store(v_path_str, chapters, data, kb_name)
                    st.success(f"Successfully added to collection: {kb_name}")
                    st.balloons()
                except Exception as e:
                    st.error(f"RAG Build failed: {e}")

    # --- Step 4: Chat Interface ---
    st.divider()
    st.header("💬 Chat with your Videos")

    available_collections = list_collections()

    if not available_collections:
        st.info("No Knowledge Bases found. Process a video and save it to start chatting.")
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_collection = st.selectbox("Select Collection", available_collections, index=0)

        # Chat Input
        if query := st.chat_input(f"Ask about '{selected_collection}'..."):
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)

            # Generate Answer
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    try:
                        result = query_knowledge_base(query, selected_collection)

                        st.markdown(result["answer"])

                        if result["sources"]:
                            st.markdown("---")
                            st.caption("Sources used:")
                            for src in result["sources"]:
                                st.caption(f"• {src}")

                    except Exception as e:
                        st.error(f"Error generating answer: {e}")

if __name__ == "__main__":
    main()
