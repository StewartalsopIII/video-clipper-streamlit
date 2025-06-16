# video_clipper_streamlit/app.py

import streamlit as st
import os
import time
from dotenv import load_dotenv
import html

import utils # Now imports the thin facade
from core import constants

# --- Page Configuration ---
st.set_page_config(page_title="AI Video Clipper", layout="wide", initial_sidebar_state="expanded")

# --- Load Environment Variables ---
load_dotenv()

# --- Initialize Session State ---
# Simplified initialization
for key, default_value in {
    "processing_complete": False, "generated_clips_info": [], "analysis_summary": "",
    "analysis_topics": [], "analysis_audience": "", "processing_log": [],
    "user_openai_api_key": "", "gpu_info_cache": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- UI Layout ---
st.title("✨ AI Video Clipper ✂️")
st.markdown("Automatically extract relevant short video clips from longer videos using AI. Powered by OpenAI & FFMPEG.")

# Placeholder for the log, defined once where it should appear.
log_display_area = st.empty()

def display_log_in_dedicated_area():
    """Renders the processing log into the log_display_area placeholder."""
    if st.session_state.processing_log:
        log_html = ""
        for entry in reversed(st.session_state.processing_log): # Newest first
            color = {"INFO": "#555555", "WARNING": "orange", "ERROR": "red", "DEBUG": "#888888"}.get(entry["level"], "inherit")
            escaped_message = html.escape(entry['message'])
            log_html += f"<div style='color: {color}; font-family: monospace; font-size: 0.9em;'>{entry['time']} [{entry['level']}] {escaped_message}</div>"
        
        log_display_area.markdown(f"""
        <div style="background-color: #f0f2f6; border: 1px solid #cccccc; padding: 10px; border-radius: 5px; max-height: 300px; overflow-y: auto; overflow-x: hidden; word-wrap: break-word;">
            {log_html}
        </div>
        """, unsafe_allow_html=True)
    else:
        log_display_area.empty() # Clear the area if there are no logs

def add_status_log_entry(message: str, level: str = "info"):
    """Adds a log entry and immediately updates the log display."""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.processing_log.append({"time": timestamp, "level": level.upper(), "message": message})
    display_log_in_dedicated_area() # Update the log display

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    input_source = st.radio("1. Select Input Source:", ("YouTube URL", "Upload MP4 File"), key="input_source_radio", horizontal=True)
    youtube_url_input = ""
    uploaded_mp4_file = None
    desired_yt_resolution = constants.DEFAULT_YT_RESOLUTION

    if input_source == "YouTube URL":
        youtube_url_input = st.text_input("Enter YouTube URL:", key="youtube_url", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        desired_yt_resolution = st.selectbox("YouTube Download Resolution:", ["best_progressive", "720p", "480p", "360p"], index=1, key="yt_resolution")
    else:
        uploaded_mp4_file = st.file_uploader("Upload an MP4 file:", type=["mp4"], key="mp4_upload")

    st.markdown("---")
    st.subheader("Transcription Options")
    transcription_options_list = ["Local Whisper Model", "OpenAI Whisper API"]
    if input_source == "YouTube URL":
        transcription_options_list.insert(0, "YouTube API (for YouTube URLs, if available)")
    
    transcription_method_choice = st.selectbox("Choose Transcription Method:", options=transcription_options_list, key="transcription_method_select", index=0)

    local_whisper_model_to_use = constants.DEFAULT_LOCAL_WHISPER_MODEL
    if transcription_method_choice == "Local Whisper Model":
        local_whisper_model_to_use = st.selectbox("Local Whisper Model Size:", ["tiny", "base", "small", "medium"], index=1, key="whisper_model_select")
        st.caption("`medium` recommended if a good GPU is available.")

    st.markdown("---")
    st.subheader("Clipping Parameters")
    segment_duration_min_input = st.slider("Min Clip Duration (s):", 10, 120, constants.MIN_CLIP_DURATION_DEFAULT, key="min_dur_slider")
    segment_duration_max_input = st.slider("Max Clip Duration (s):", 15, 180, constants.MAX_CLIP_DURATION_DEFAULT, key="max_dur_slider")
    if segment_duration_min_input > segment_duration_max_input:
        st.warning("Min duration > Max. Adjusting max.")
        segment_duration_max_input = segment_duration_min_input
    max_topics_to_generate = st.number_input("Max Topics by LLM:", 1, 10, constants.MAX_TOPICS_DEFAULT, key="max_topics_number")

    st.markdown("---")
    env_api_key = utils.get_openai_api_key()
    if not st.session_state.user_openai_api_key and env_api_key: # Load from .env if session state is empty
        st.session_state.user_openai_api_key = env_api_key

    # Allow user to input/override API key
    current_api_key_in_field = st.text_input(
        "OpenAI API Key:", 
        value=st.session_state.user_openai_api_key, 
        type="password", 
        key="openai_key_manual_input_sidebar" # Unique key
    )
    st.session_state.user_openai_api_key = current_api_key_in_field # Update session state from input
    
    openai_api_key_for_processing = st.session_state.user_openai_api_key

    if openai_api_key_for_processing:
        if env_api_key and openai_api_key_for_processing == env_api_key:
            st.success("OpenAI API Key loaded from .env.")
        else:
            st.info("Using API Key entered in session.")
    else:
        st.warning("OpenAI API Key is required.")

    process_button_disabled = (
        (input_source == "YouTube URL" and not youtube_url_input) or
        (input_source == "Upload MP4 File" and not uploaded_mp4_file) or
        not openai_api_key_for_processing
    )
    process_button = st.button("🚀 Generate Clips", use_container_width=True, type="primary", disabled=process_button_disabled)

# --- Main Area Content ---

# GPU Information Expander (below the log area, which is log_display_area)
with st.expander("🖥️ GPU Information", expanded=False): # Default to not expanded, or True if preferred
    gpu_info_placeholder = st.empty()
    if st.button("Refresh GPU Info", key="refresh_gpu_main_button_expander", use_container_width=True):
        st.session_state.gpu_info_cache = utils.get_nvidia_smi_output()
    if not st.session_state.gpu_info_cache: # Load initially if not cached
         st.session_state.gpu_info_cache = utils.get_nvidia_smi_output()
    gpu_info_placeholder.text(st.session_state.gpu_info_cache if st.session_state.gpu_info_cache else "NVIDIA GPU info not available or nvidia-smi not found.")


# Initial display of the log area (e.g., if there are logs from a previous state before button press)
# This also ensures the area is established correctly.
display_log_in_dedicated_area()


if process_button:
    # Reset states for a new run
    st.session_state.processing_complete = False
    st.session_state.generated_clips_info = []
    st.session_state.analysis_summary = ""
    st.session_state.analysis_topics = []
    st.session_state.analysis_audience = ""
    st.session_state.processing_log = [] # CRITICAL: Clear logs for the new processing run
    display_log_in_dedicated_area() # Visually clear the log display area

    add_status_log_entry("🚀 Processing started...", "info")
    # No need to call display_log_in_dedicated_area() again here, add_status_log_entry does it.

    # Ensure output directories exist
    for dir_path in [constants.VIDEO_DOWNLOAD_DIR, constants.CLIPS_OUTPUT_DIR, constants.TEMP_AUDIO_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            # Use the status_callback (add_status_log_entry) to log this
            add_status_log_entry(f"Ensured directory: {dir_path}", "debug")

    processing_results_dict = None
    # The status_callback for orchestrators is now add_status_log_entry,
    # which will update the log display automatically.
    if input_source == "YouTube URL":
        processing_results_dict = utils.process_youtube_url(
            youtube_url=youtube_url_input,
            desired_yt_resolution=desired_yt_resolution,
            transcription_method_choice=transcription_method_choice,
            local_whisper_model_size=local_whisper_model_to_use,
            openai_api_key=openai_api_key_for_processing,
            max_topics=max_topics_to_generate,
            min_clip_duration=segment_duration_min_input,
            max_clip_duration=segment_duration_max_input,
            status_callback=add_status_log_entry
        )
    elif input_source == "Upload MP4 File":
        processing_results_dict = utils.process_uploaded_mp4(
            uploaded_file_obj=uploaded_mp4_file,
            transcription_method_choice=transcription_method_choice,
            local_whisper_model_size=local_whisper_model_to_use,
            openai_api_key=openai_api_key_for_processing,
            max_topics=max_topics_to_generate,
            min_clip_duration=segment_duration_min_input,
            max_clip_duration=segment_duration_max_input,
            status_callback=add_status_log_entry
        )

    if processing_results_dict:
        st.session_state.processing_complete = processing_results_dict.get("success", False)
        analysis = processing_results_dict.get("analysis")
        if analysis:
            st.session_state.analysis_summary = analysis.summary
            st.session_state.analysis_topics = analysis.main_topics
            st.session_state.analysis_audience = analysis.target_audience or "Not specified"
        
        st.session_state.generated_clips_info = processing_results_dict.get("generated_clips_info", [])
        
        if not st.session_state.processing_complete:
            error_msg = processing_results_dict.get("error_message", "Processing failed.")
            add_status_log_entry(f"Processing failed: {error_msg}", "error")
        else:
            add_status_log_entry("✅ All processing steps completed!", "info")
            if not st.session_state.generated_clips_info and st.session_state.analysis_topics:
                add_status_log_entry("No clips were generated based on the criteria for the identified topics.", "warning")
            st.success("🎉 Video clipping process finished!") # Toast-like message
    else:
        add_status_log_entry("Critical error: Orchestrator did not return results.", "error")
    
    # Log display is already updated by the last add_status_log_entry call.

# --- Display Results Area (Below Log and GPU info) ---
# This part runs on every script rerun, showing results based on session state.
if st.session_state.processing_complete:
    st.subheader("📊 Processing Summary & Results")
    # Video Analysis Expander
    if st.session_state.analysis_summary:
        with st.expander("📝 Video Analysis", expanded=True):
            st.markdown(f"**Summary:** {st.session_state.analysis_summary}")
            if st.session_state.analysis_topics:
                st.markdown(f"**Identified Topics:**")
                for t_topic in st.session_state.analysis_topics: st.markdown(f"- {t_topic}")
            if st.session_state.analysis_audience:
                st.markdown(f"**Suggested Target Audience:** {st.session_state.analysis_audience}")
    
    # Generated Clips Display
    if st.session_state.generated_clips_info:
        st.subheader(f"🎬 Generated Clips ({len(st.session_state.generated_clips_info)})")
        for i, clip_info in enumerate(st.session_state.generated_clips_info): # Added index for unique keys
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
                                    # Use a more robust unique key
                                    key=f"dl_{utils.sanitize_filename(clip_filename)}_{i}_{int(time.time())}",
                                    use_container_width=True
                                )
                        except Exception as e_dl: 
                            st.error(f"Download error: {e_dl}")
                    else: 
                        st.error(f"Clip file missing: {clip_filename}")
    elif st.session_state.analysis_topics: # Topics found, but no clips generated
        st.info("No clips were generated for the identified topics based on the current criteria. Try adjusting duration or checking logs.")
    # If processing_complete is True but no analysis and no clips, the "All processing steps completed!" log and st.success will show.

elif st.session_state.processing_log: # Processing was initiated but did not complete (processing_complete is False)
    # Check if the "Processing failed:" log entry exists
    failed_log_exists = any("Processing failed:" in entry['message'] for entry in st.session_state.processing_log)
    if failed_log_exists:
        st.warning("Processing did not complete successfully. Please review the log above for error details.")
    # else:  Could be processing_log has "Processing started..." but button not fully processed yet.
    # This state is typically transient or covered by the log itself.