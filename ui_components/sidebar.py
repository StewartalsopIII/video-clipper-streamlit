# ui_components/sidebar.py
import streamlit as st
from core import constants
import utils 

def render_sidebar() -> dict:
    """Renders the sidebar and returns user configuration."""
    config = {}
    with st.sidebar:
        st.header("⚙️ Configuration")

        config["input_source"] = st.radio(
            "1. Select Input Source:", 
            ("YouTube URL", "Upload MP4 File"), 
            key="input_source_radio_sidebar", # Ensure keys are unique if refactoring
            horizontal=True
        )

        config["youtube_url_input"] = ""
        config["uploaded_mp4_file"] = None
        config["desired_yt_resolution"] = constants.DEFAULT_YT_RESOLUTION 

        if config["input_source"] == "YouTube URL":
            config["youtube_url_input"] = st.text_input(
                "Enter YouTube URL:", 
                key="youtube_url_sidebar", 
                placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            )
            config["desired_yt_resolution"] = st.selectbox(
                "YouTube Download Resolution:",
                ["best_progressive", "720p", "480p", "360p"], index=1, key="yt_resolution_sidebar"
            )
        else: # Upload MP4 File
            config["uploaded_mp4_file"] = st.file_uploader(
                "Upload an MP4 file:", 
                type=["mp4"], 
                key="mp4_upload_sidebar"
            )

        st.markdown("---")
        st.subheader("Transcription Options")
        transcription_options = ["Local Whisper Model", "OpenAI Whisper API"]
        if config["input_source"] == "YouTube URL":
            transcription_options.insert(0, "YouTube API (for YouTube URLs, if available)")
        
        config["transcription_method_choice"] = st.selectbox(
            "Choose Transcription Method:",
            options=transcription_options,
            key="transcription_method_sidebar",
            index=0
        )

        config["local_whisper_model_to_use"] = constants.DEFAULT_LOCAL_WHISPER_MODEL 
        if config["transcription_method_choice"] == "Local Whisper Model":
            config["local_whisper_model_to_use"] = st.selectbox(
                "Local Whisper Model Size:",
                ["tiny", "base", "small", "medium"], 
                index=1, key="whisper_model_sidebar"
            )
            st.caption("`medium` recommended if a good GPU is available.")

        st.markdown("---")
        st.subheader("Clipping Parameters")
        config["segment_duration_min_input"] = st.slider(
            "Min Clip Duration (s):", 10, 120, constants.MIN_CLIP_DURATION_DEFAULT, key="min_dur_sidebar"
        )
        config["segment_duration_max_input"] = st.slider(
            "Max Clip Duration (s):", 15, 180, constants.MAX_CLIP_DURATION_DEFAULT, key="max_dur_sidebar"
        )
        if config["segment_duration_min_input"] > config["segment_duration_max_input"]:
            st.warning("Min duration > Max. Adjusting max.")
            config["segment_duration_max_input"] = config["segment_duration_min_input"]
        
        config["max_topics_to_generate"] = st.number_input(
            "Max Topics by LLM:", 1, 10, constants.MAX_TOPICS_DEFAULT, key="max_topics_sidebar"
        )

        st.markdown("---")
        env_api_key = utils.get_openai_api_key() # utils is the facade
        if 'user_openai_api_key_sidebar_val' not in st.session_state: # Use a unique key for this input
            st.session_state.user_openai_api_key_sidebar_val = env_api_key if env_api_key else ""

        config["openai_api_key_for_processing"] = env_api_key
        if not env_api_key:
            st.session_state.user_openai_api_key_sidebar_val = st.text_input(
                "Enter OpenAI API Key:", 
                value=st.session_state.user_openai_api_key_sidebar_val, 
                type="password", 
                key="openai_key_manual_sidebar_input"
            )
            config["openai_api_key_for_processing"] = st.session_state.user_openai_api_key_sidebar_val
        
        if config["openai_api_key_for_processing"] and env_api_key:
            st.success("OpenAI API Key loaded from .env.")
        elif config["openai_api_key_for_processing"] and not env_api_key:
            st.info("API Key entered for this session.")
        else:
            st.warning("OpenAI API Key is required.")

        process_button_disabled = (
            (config["input_source"] == "YouTube URL" and not config["youtube_url_input"]) or
            (config["input_source"] == "Upload MP4 File" and not config["uploaded_mp4_file"]) or
            not config["openai_api_key_for_processing"]
        )
        config["process_button_pressed"] = st.button(
            "🚀 Generate Clips", 
            use_container_width=True, 
            type="primary", 
            disabled=process_button_disabled
        )
    return config