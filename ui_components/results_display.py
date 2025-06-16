# ui_components/results_display.py
import streamlit as st
import os
import time
from services import system_service # For sanitize_filename

def render_results(results_placeholder_col):
    """Displays the processing summary and generated clips."""
    with results_placeholder_col:
        if "processing_complete" in st.session_state and st.session_state.processing_complete:
            st.subheader("📊 Processing Summary & Results")
            if st.session_state.get("analysis_summary"):
                with st.expander("📝 Video Analysis", expanded=True):
                    st.markdown(f"**Summary:** {st.session_state.analysis_summary}")
                    if st.session_state.get("analysis_topics"):
                        st.markdown(f"**Identified Topics:**")
                        for t_topic in st.session_state.analysis_topics: st.markdown(f"- {t_topic}")
                    if st.session_state.get("analysis_audience"):
                        st.markdown(f"**Suggested Target Audience:** {st.session_state.analysis_audience}")
            
            if st.session_state.get("generated_clips_info"):
                st.subheader(f"🎬 Generated Clips ({len(st.session_state.generated_clips_info)})")
                for clip_info in st.session_state.generated_clips_info:
                    container = st.container(border=True)
                    with container:
                        clip_col1, clip_col2 = st.columns([4,1])
                        with clip_col1:
                            st.markdown(f"##### {clip_info.get('title', 'Untitled Clip')}")
                            st.caption(
                                f"Topic: {clip_info.get('topic', 'N/A')} | "
                                f"Duration: {clip_info.get('duration', 'N/A')}s | "
                                f"Time: {clip_info.get('start_time', 'N/A')}-{clip_info.get('end_time', 'N/A')}s"
                            )
                            st.caption(f"Description: {clip_info.get('description', 'No description.')}")
                        
                        with clip_col2:
                            clip_file_path = clip_info.get('path')
                            clip_filename = clip_info.get('filename')
                            if clip_file_path and clip_filename and os.path.exists(clip_file_path):
                                try:
                                    with open(clip_file_path, "rb") as fp_clip:
                                        st.download_button(
                                            label="⬇️ Download", data=fp_clip, file_name=clip_filename,
                                            mime="video/mp4", 
                                            key=f"dl_{system_service.sanitize_filename(clip_filename)}", # Filename should be unique
                                            use_container_width=True
                                        )
                                except Exception as e_dl: st.error(f"Download error: {e_dl}")
                            else: st.error(f"Clip file missing: {clip_filename}")
            elif st.session_state.get("analysis_topics"): 
                st.info("No clips were generated for the identified topics based on the current criteria. Try adjusting duration or checking logs.")
            elif st.session_state.get("processing_log"): 
                st.info("Processing finished, but no analysis or clips were produced. Please check the log.")

        elif "processing_log" in st.session_state and st.session_state.processing_log : 
            st.warning("Processing was initiated but may not have completed successfully. Check the log above.")