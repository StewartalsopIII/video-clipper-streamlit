Project Path: video_clipper_streamlit

Source Tree:

```txt
video_clipper_streamlit
├── README.md
├── __init__.py
├── app.py
├── clipper.ipynb
├── core
│   ├── constants.py
│   └── models.py
├── ffmpeg.exe
├── ffplay.exe
├── ffprobe.exe
├── orchestrators
│   ├── __init__.py
│   ├── common_steps.py
│   ├── local_mp4_pipeline.py
│   └── youtube_pipeline.py
├── services
│   ├── __init__.py
│   ├── ffmpeg_service.py
│   ├── llm_service.py
│   ├── system_service.py
│   ├── transcription_service.py
│   └── video_processing_service.py
├── test_cuda.py
├── ui_components
│   ├── results_display.py
│   ├── sidebar.py
│   └── status_logger.py
└── utils.py

```

`video_clipper_streamlit/README.md`:

```md
# ✨ AI Video Clipper ✂️

Automatically extract relevant, short video clips (snippets) from longer video content using AI. This Streamlit web application supports both YouTube videos (via URL) and locally uploaded MP4 files, offering flexible transcription methods and leveraging Large Language Models (LLMs) for intelligent content analysis and segment extraction.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [How to Run](#how-to-run)
- [Usage Guide](#usage-guide)
  - [Input Options](#input-options)
  - [Transcription Options](#transcription-options)
  - [Clipping Parameters](#clipping-parameters)
  - [Processing](#processing)
  - [Results](#results)
- [Technical Details](#technical-details)
  - [Core Workflow](#core-workflow)
  - [Key Libraries Used](#key-libraries-used)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

## Features

-   **Flexible Video Input:**
    -   Process videos directly from YouTube URLs.
    -   Upload and process local MP4 files.
-   **Multiple Transcription Engines:**
    -   **YouTube API:** Utilizes YouTube's own captions for the fastest transcription of YouTube videos (if available and of good quality).
    -   **Local Whisper:** Employs OpenAI's Whisper model (via `stable-whisper`) running on your local machine for high-quality transcription. Supports GPU (NVIDIA CUDA) acceleration for significantly faster processing. Users can select different Whisper model sizes (`tiny`, `base`, `small`, `medium`) to balance speed and accuracy.
    -   **OpenAI Whisper API:** Leverages OpenAI's official Whisper API (`whisper-1` model) for transcription, offering potentially the highest accuracy by using their large, managed models. Requires an OpenAI API key and incurs API usage costs.
-   **AI-Powered Topic Identification:**
    -   Analyzes the full video transcript using an LLM (e.g., `gpt-4o-mini`) to automatically identify a configurable number of main topics or themes within the video.
-   **Targeted Segment Extraction:**
    *   For each identified topic, an LLM (e.g., `gpt-4o-mini` or `gpt-4o`) scans the transcript to extract specific video segments that are:
        *   Relevant to the current topic.
        *   Within a user-configurable duration range.
    *   The LLM generates a title and description for each potential clip.
-   **Automated Video Snippet Generation:**
    *   Uses FFMPEG to precisely cut the identified video segments from the original video file.
    *   Generated clips are named descriptively and made available for download.
-   **Interactive User Interface (Streamlit):**
    *   Intuitive sidebar for all configurations.
    *   Real-time processing log and status updates.
    *   Display of GPU information (if an NVIDIA GPU is detected via `nvidia-smi`).
    *   Clear presentation of identified video summary, topics, and generated clips with download buttons.
-   **Modular and Maintainable Codebase:**
    *   Separation of UI (`app.py`), orchestration (`utils.py`), core models/constants (`core/`), and distinct functionalities into service modules (`services/`).

## Project Structure

```txt
video_clipper_streamlit/
├── app.py                     # Main Streamlit application UI
├── core/                      # Core components
│   ├── constants.py           # Shared constants
│   └── models.py              # Pydantic models
├── orchestrators/             # Pipeline orchestration logic
│   ├── __init__.py
│   ├── common_steps.py
│   ├── local_mp4_pipeline.py
│   └── youtube_pipeline.py
├── services/                  # Individual service modules
│   ├── __init__.py
│   ├── ffmpeg_service.py
│   ├── llm_service.py
│   ├── system_service.py
│   ├── transcription_service.py
│   └── video_processing_service.py
├── ui_components/             # UI component functions (if further refactored)
│   ├── __init__.py
│   ├── sidebar.py
│   ├── results_display.py
│   └── status_logger.py
├── utils.py                   # Thin facade for orchestrators & some system functions
├── downloaded_video/          # Stores downloaded/uploaded videos
├── generated_clips/           # Stores generated video snippets
├── temp_audio/                # Stores temporary audio files for API transcription
├── ffmpeg.exe                 # (If bundling - ensure license compliance)
├── ffprobe.exe                # (If bundling - ensure license compliance)
├── .env                       # For API keys and environment variables
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```
*(Note: `ffmpeg.exe` and `ffprobe.exe` are shown in the structure if you choose to bundle them. Ensure you comply with FFMPEG licensing if distributing.)*

## Setup Instructions

### Prerequisites

1.  **Python:** Version 3.9 or higher.
2.  **FFMPEG:** Must be installed and accessible in your system's PATH, OR `ffmpeg.exe` and `ffprobe.exe` must be placed in the `video_clipper_streamlit/` directory. Download from [ffmpeg.org](https://ffmpeg.org/download.html).
3.  **NVIDIA GPU & CUDA (Optional, for Local Whisper GPU acceleration):**
    *   NVIDIA GPU with CUDA support.
    *   NVIDIA drivers installed.
    *   CUDA Toolkit compatible with your PyTorch version (see PyTorch website for installation).
    *   cuDNN library.
4.  **Git (Optional, for cloning):**

### Installation

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd video_clipper_streamlit 
    ```
    If you have the files directly, navigate to the `video_clipper_streamlit` directory.

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the following content (versions might need adjustment based on your setup and compatibility):
    ```txt
    streamlit
    pytubefix
    youtube-transcript-api
    pydantic>=2.0
    langchain-openai>=0.1.0
    langchain>=0.1.0
    openai>=1.0 # For OpenAI API (Whisper and LLMs)
    python-dotenv
    torch # Install with CUDA support if using GPU, see PyTorch website
    torchvision # Often installed with PyTorch
    torchaudio # Often installed with PyTorch
    openai-whisper # Or stable-whisper, if you are using that fork
    # ffmpeg-python # Only if you choose to use this wrapper for direct ffmpeg calls
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```
    **Important for GPU users:** Install PyTorch with the correct CUDA version. Visit [pytorch.org](https://pytorch.org/get-started/locally/) and select your OS, pip, Python, and CUDA version to get the specific installation command. Example for CUDA 12.1:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

### Configuration

1.  **OpenAI API Key:**
    *   Create a `.env` file in the `video_clipper_streamlit/` directory.
    *   Add your OpenAI API key to it:
        ```env
        OPENAI_API_KEY="sk-YourActualOpenAIAPIKey"
        ```
    *   The application will also allow you to enter the API key in the sidebar if not found in the `.env` file.

## How to Run

1.  Ensure your virtual environment is activated.
2.  Navigate to the `video_clipper_streamlit/` directory in your terminal.
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
4.  Open the local URL (usually `http://localhost:8501`) provided in your terminal in a web browser.

## Usage Guide

The application interface is divided into a sidebar for configuration and a main area for logs and results.

### Input Options

1.  **Select Input Source:**
    *   **YouTube URL:** Paste the full URL of the YouTube video.
        *   **YouTube Download Resolution:** Choose the desired quality for the downloaded video. "best\_progressive" usually gives the highest single-file quality.
    *   **Upload MP4 File:** Click "Browse files" or drag and drop your MP4 file.

### Transcription Options

Choose how the video's audio will be transcribed:

1.  **YouTube API (for YouTube URLs, if available):**
    *   Fastest method for YouTube videos.
    *   Uses YouTube's automatically generated or creator-uploaded captions.
    *   Quality can vary. If unavailable or poor, another method will be needed.
2.  **Local Whisper Model:**
    *   Processes audio on your machine/server using OpenAI's Whisper.
    *   **Local Whisper Model Size:** Select from `tiny`, `base`, `small`, `medium`.
        *   `tiny`/`base`: Faster, less accurate, good for quick tests or powerful CPUs.
        *   `small`/`medium`: More accurate, slower, **GPU strongly recommended for these.**
    *   The app will attempt to use CUDA GPU if available.
3.  **OpenAI Whisper API:**
    *   Sends the audio (extracted from the video) to OpenAI's `whisper-1` model for transcription.
    *   Requires a valid OpenAI API Key.
    *   Incurs costs based on OpenAI's pricing.
    *   Generally provides very high accuracy.

### Clipping Parameters

*   **Min/Max Clip Duration (seconds):** Define the desired length range for the extracted clips.
*   **Max Topics by LLM:** Specify how many distinct topics the LLM should try to identify in the video.

### Processing

*   **OpenAI API Key:** Ensure your key is loaded from `.env` or entered in the sidebar.
*   Click the **"🚀 Generate Clips"** button.
*   Monitor the "📊 Processing Log" in the main area for real-time updates on each step.
*   GPU information can be viewed in the "🖥️ GPU Information" expander.

### Results

Once processing is complete:

1.  **Processing Summary & Results:**
    *   **Video Analysis:** Shows the LLM-generated summary, identified topics, and suggested target audience.
    *   **Generated Clips:** If clips were successfully created, they will be listed here. Each entry includes:
        *   Title (generated by LLM)
        *   Associated Topic
        *   Duration and Time Range
        *   Description (generated by LLM)
        *   A "⬇️ Download" button for the MP4 clip.
    *   If no clips were generated for the given criteria, an informational message will be displayed.

## Technical Details

### Core Workflow

1.  **Video Input:** User provides a YouTube URL or uploads an MP4.
2.  **Video Acquisition:**
    *   YouTube: Video downloaded using `pytubefix`.
    *   Upload: MP4 file is saved locally.
3.  **Transcription:** Based on user choice:
    *   YouTube API: Transcript fetched using `youtube-transcript-api`.
    *   Local Whisper: Audio processed by `stable-whisper` (or `openai-whisper`) on local hardware (CPU/GPU).
    *   OpenAI API: Audio extracted using FFMPEG, then sent to OpenAI's `whisper-1` API.
    *   All methods produce a standardized transcript format (list of segments with text, start, end, duration).
4.  **Topic Identification:** The transcript text is sent to an OpenAI LLM (e.g., `gpt-4o-mini`) to identify key topics. LangChain with Pydantic models ensures structured output.
5.  **Segment Extraction:** For each identified topic, the full structured transcript is sent to an OpenAI LLM (e.g., `gpt-4o-mini` or `gpt-4o`) with instructions to find relevant segments matching the user-defined duration. LangChain with Pydantic models is used.
6.  **Clip Generation:** FFMPEG is used via `subprocess` to cut the video segments based on the start and end times provided by the LLM.
7.  **Display:** Streamlit UI updates with logs, analysis, and downloadable clips.

### Key Libraries Used

-   **Streamlit:** Web application framework.
-   **Pytubefix:** Downloading YouTube videos.
-   **youtube-transcript-api:** Fetching YouTube captions.
-   **openai-whisper / stable-whisper:** Local speech-to-text.
-   **openai (SDK v1.x+):** Interacting with OpenAI APIs (Whisper API, Chat models).
-   **langchain-openai, langchain-core:** Orchestrating LLM calls and ensuring structured output with Pydantic.
-   **Pydantic (v2):** Data validation and modeling for LLM outputs.
-   **python-dotenv:** Managing environment variables (API keys).
-   **Torch:** For local Whisper and GPU detection.
-   **subprocess:** Running FFMPEG and `nvidia-smi`.

## Troubleshooting

-   **`FileNotFoundError: [WinError 2] The system cannot find the file specified` (related to ffmpeg):**
    -   Ensure `ffmpeg.exe` and `ffprobe.exe` are either in your system PATH or placed directly in the `video_clipper_streamlit/` directory (where `app.py` runs).
    -   Restart your terminal/IDE after modifying PATH.
-   **Local Whisper runs on CPU instead of GPU:**
    -   Verify your NVIDIA drivers, CUDA Toolkit, and PyTorch (with CUDA support) installation are correct and compatible.
    -   Check the console logs and Streamlit UI logs for messages from `system_service.check_gpu_availability()` to see if CUDA is detected within the Streamlit app's environment.
    -   Ensure you have enough VRAM for the selected Whisper model size.
-   **OpenAI API Errors (e.g., AuthenticationError, RateLimitError):**
    -   Check that your OpenAI API key is correct, active, and has sufficient funds/quota.
    -   Consult the console logs for specific error messages from the OpenAI API.
-   **No Segments Found by LLM:**
    -   The transcript quality might be poor.
    -   The duration constraints might be too restrictive for the video content. Try widening the range.
    -   The topics identified might not have substantial, clippable content within the duration limits.
    -   The LLM (especially `gpt-4o-mini`) might struggle with the complexity. Try `gpt-4o` for the "Segment Extraction LLM" (configurable in `core/constants.py`).
    -   Review the prompts in `services/llm_service.py` and try making them more explicit, especially regarding combining transcript parts to meet duration.
-   **Streamlit "Watcher" Errors on Startup (e.g., `RuntimeError: no running event loop`, `torch._classes` errors):**
    -   These are often due to Streamlit's file watcher interacting with complex libraries like PyTorch. If the app loads and the core functionality works, these can often be ignored during development. Ensure libraries are up-to-date.

## Future Enhancements

-   Batch processing of multiple videos.
-   Advanced FFMPEG options (e.g., adding text overlays, watermarks, different output formats/resolutions).
-   User accounts and processing history.
-   Interactive UI for reviewing and selecting/deselecting LLM-suggested segments before FFMPEG processing.
-   Support for more Speech-to-Text engines.
-   Option to choose different LLMs for topic identification and segment extraction.
-   Deployment to a cloud platform (e.g., Streamlit Community Cloud, Heroku, AWS).
-   More sophisticated transcript pre-processing or chunking for very long videos sent to LLMs.


```

`video_clipper_streamlit/app.py`:

```py
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
```

`video_clipper_streamlit/clipper.ipynb`:

```ipynb
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4a2095d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import json\n",
    "from pytubefix import YouTube # Using pytubefix\n",
    "# from pytubefix.exceptions import PytubeFixError # Check actual exception name if needed for specific pytubefix errors\n",
    "from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound\n",
    "from pydantic import BaseModel, Field, RootModel # For Pydantic V2\n",
    "from langchain_openai.chat_models import ChatOpenAI\n",
    "import subprocess\n",
    "from typing import List, Optional\n",
    "from dotenv import load_dotenv\n",
    "import logging # For more structured logging\n",
    "\n",
    "# Configure basic logging\n",
    "logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "11856426",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-06-15 18:14:44,192 - INFO - OpenAI API key loaded successfully.\n"
     ]
    }
   ],
   "source": [
    "load_dotenv()\n",
    "OPENAI_API_KEY = os.getenv(\"OPENAI_API_KEY\")\n",
    "\n",
    "if not OPENAI_API_KEY:\n",
    "    logging.error(\"OpenAI API key not found. Please set it in your .env file.\")\n",
    "    # You might want to raise an exception here or handle it more gracefully\n",
    "    # raise ValueError(\"OpenAI API key not found.\")\n",
    "else:\n",
    "    logging.info(\"OpenAI API key loaded successfully.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "26782a07",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-06-15 18:14:47,872 - INFO - Configuration loaded. Target YouTube URL: https://youtu.be/nvuAt8sl7Ag?si=6x3KLf63BttTX_qx&utm_source=ZTQxO\n"
     ]
    }
   ],
   "source": [
    "# --- User-Defined Configuration ---\n",
    "youtube_url = \"https://youtu.be/nvuAt8sl7Ag?si=6x3KLf63BttTX_qx&utm_source=ZTQxO\"\n",
    "video_download_directory = \"downloaded_video\"\n",
    "clips_output_directory = \"generated_clips\"\n",
    "desired_video_resolution = \"720p\" # e.g., \"720p\", \"360p\", or None for best progressive\n",
    "segment_duration_min_seconds = 50\n",
    "segment_duration_max_seconds = 59\n",
    "max_topics_to_identify = 5 # How many topics to ask the first LLM for\n",
    "\n",
    "# --- Global State Variables (initialized) ---\n",
    "downloaded_video_path = None\n",
    "video_base_title = None\n",
    "raw_transcript_data = None      \n",
    "transcript_file_path = None\n",
    "identified_topics: List[str] = []\n",
    "all_extracted_segments: List[dict] = [] # Will store Segment-like dictionaries\n",
    "\n",
    "logging.info(f\"Configuration loaded. Target YouTube URL: {youtube_url}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0d28d155",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Ensure Directories Exist ---\n",
    "if not os.path.exists(video_download_directory):\n",
    "    os.makedirs(video_download_directory)\n",
    "    logging.info(f\"Created directory: {video_download_directory}\")\n",
    "\n",
    "if not os.path.exists(clips_output_directory):\n",
    "    os.makedirs(clips_output_directory)\n",
    "    logging.info(f\"Created directory: {clips_output_directory}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "fe362d70",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-06-15 18:14:52,477 - INFO - Fetching video info for: https://youtu.be/nvuAt8sl7Ag?si=6x3KLf63BttTX_qx&utm_source=ZTQxO\n",
      "2025-06-15 18:14:53,040 - INFO - Video Title: The High-Paying AI Job Nobody Knows About (Yet) ft. Rachel Woods\n",
      "2025-06-15 18:14:53,041 - INFO - Sanitized base title: The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods\n",
      "2025-06-15 18:14:53,041 - INFO - Attempting to download video to: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\n",
      "2025-06-15 18:14:53,042 - WARNING - Resolution 720p not found as progressive mp4. Trying best available.\n",
      "2025-06-15 18:14:53,042 - INFO - Found video stream: Resolution 360p, MIME Type video/mp4\n",
      "2025-06-15 18:15:06,408 - INFO - Successfully downloaded video: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\n"
     ]
    }
   ],
   "source": [
    "# --- Download Video ---\n",
    "try:\n",
    "    logging.info(f\"Fetching video info for: {youtube_url}\")\n",
    "    yt = YouTube(youtube_url)\n",
    "\n",
    "    video_title_raw = yt.title\n",
    "    # Sanitize title for filename: Keep alphanumeric, spaces, hyphens, underscores. Replace others with underscore.\n",
    "    video_base_title_temp = \"\".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in video_title_raw).rstrip()\n",
    "    video_base_title_temp = video_base_title_temp.replace(\" \", \"_\")\n",
    "    if not video_base_title_temp: # Handle cases where title becomes empty\n",
    "        video_base_title_temp = f\"youtube_video_{yt.video_id}\"\n",
    "    video_base_title = video_base_title_temp # Assign to global variable\n",
    "\n",
    "    video_filename_mp4 = f\"{video_base_title}.mp4\"\n",
    "    downloaded_video_path_temp = os.path.join(video_download_directory, video_filename_mp4)\n",
    "\n",
    "    logging.info(f\"Video Title: {video_title_raw}\")\n",
    "    logging.info(f\"Sanitized base title: {video_base_title}\")\n",
    "    logging.info(f\"Attempting to download video to: {downloaded_video_path_temp}\")\n",
    "\n",
    "    # Check if video already exists to avoid re-downloading\n",
    "    if os.path.exists(downloaded_video_path_temp):\n",
    "        logging.info(f\"Video already exists at {downloaded_video_path_temp}. Skipping download.\")\n",
    "        downloaded_video_path = downloaded_video_path_temp\n",
    "    else:\n",
    "        streams_query = yt.streams.filter(progressive=True, file_extension='mp4')\n",
    "        stream_to_download = None\n",
    "        if desired_video_resolution:\n",
    "            stream_to_download = streams_query.filter(res=desired_video_resolution).first()\n",
    "            if not stream_to_download:\n",
    "                logging.warning(f\"Resolution {desired_video_resolution} not found as progressive mp4. Trying best available.\")\n",
    "                stream_to_download = streams_query.order_by('resolution').desc().first()\n",
    "        else:\n",
    "            stream_to_download = streams_query.order_by('resolution').desc().first()\n",
    "\n",
    "        if stream_to_download:\n",
    "            logging.info(f\"Found video stream: Resolution {stream_to_download.resolution}, MIME Type {stream_to_download.mime_type}\")\n",
    "            stream_to_download.download(output_path=video_download_directory, filename=video_filename_mp4)\n",
    "            logging.info(f\"Successfully downloaded video: {downloaded_video_path_temp}\")\n",
    "            downloaded_video_path = downloaded_video_path_temp # Assign to global\n",
    "        else:\n",
    "            logging.error(\"No suitable progressive MP4 stream found for this video.\")\n",
    "            # Consider raising an error to stop execution if video is essential\n",
    "\n",
    "except Exception as e: # Catching a broader exception for pytubefix initially\n",
    "    logging.error(f\"An error occurred during video download: {e}\", exc_info=True)\n",
    "    # Consider how to handle this - maybe exit or skip subsequent steps"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "248fe260",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-06-15 18:15:09,214 - INFO - Fetching transcript for video ID: nvuAt8sl7Ag...\n",
      "2025-06-15 18:15:10,300 - INFO - Successfully fetched and saved transcript to: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_transcript.json\n",
      "2025-06-15 18:15:10,301 - INFO - Transcript loaded/fetched successfully. Number of segments: 1426\n",
      "2025-06-15 18:15:10,302 - INFO - Full transcript text concatenated. Length: 53713 characters.\n"
     ]
    }
   ],
   "source": [
    "# --- Fetch and Save Transcript ---\n",
    "if downloaded_video_path and video_base_title: # Proceed only if video download was successful\n",
    "    try:\n",
    "        video_id = yt.video_id # yt object should still be in scope from Cell 5\n",
    "        logging.info(f\"Fetching transcript for video ID: {video_id}...\")\n",
    "\n",
    "        transcript_filename_json = f\"{video_base_title}_transcript.json\"\n",
    "        transcript_file_path_temp = os.path.join(video_download_directory, transcript_filename_json)\n",
    "\n",
    "        # Check if transcript file already exists\n",
    "        if os.path.exists(transcript_file_path_temp):\n",
    "            logging.info(f\"Transcript file already exists: {transcript_file_path_temp}. Loading from file.\")\n",
    "            with open(transcript_file_path_temp, \"r\", encoding=\"utf-8\") as f:\n",
    "                raw_transcript_data_temp = json.load(f)\n",
    "        else:\n",
    "            raw_transcript_data_temp = YouTubeTranscriptApi.get_transcript(video_id)\n",
    "            with open(transcript_file_path_temp, \"w\", encoding=\"utf-8\") as f:\n",
    "                json.dump(raw_transcript_data_temp, f, ensure_ascii=False, indent=4)\n",
    "            logging.info(f\"Successfully fetched and saved transcript to: {transcript_file_path_temp}\")\n",
    "\n",
    "        raw_transcript_data = raw_transcript_data_temp # Assign to global\n",
    "        transcript_file_path = transcript_file_path_temp # Assign to global\n",
    "\n",
    "        # Optional: Basic validation of transcript data\n",
    "        if isinstance(raw_transcript_data, list) and len(raw_transcript_data) > 0 and isinstance(raw_transcript_data[0], dict):\n",
    "            logging.info(f\"Transcript loaded/fetched successfully. Number of segments: {len(raw_transcript_data)}\")\n",
    "        else:\n",
    "            logging.warning(\"Transcript data seems to be empty or not in the expected format.\")\n",
    "            raw_transcript_data = None # Reset if invalid\n",
    "\n",
    "    except TranscriptsDisabled:\n",
    "        logging.error(f\"Transcripts are disabled for video: {youtube_url}\")\n",
    "        raw_transcript_data = None\n",
    "    except NoTranscriptFound:\n",
    "        logging.error(f\"No transcript found for video: {youtube_url}. It might be missing or not in a supported language.\")\n",
    "        raw_transcript_data = None\n",
    "    except Exception as e:\n",
    "        logging.error(f\"An error occurred while fetching or saving the transcript: {e}\", exc_info=True)\n",
    "        raw_transcript_data = None\n",
    "else:\n",
    "    logging.warning(\"Skipping transcript fetching because video download failed or video_base_title was not set.\")\n",
    "\n",
    "# For the LLM, we might need a single string of the transcript text for some prompts\n",
    "# For topic identification, sending the whole structured transcript might be too much for cheaper models.\n",
    "# Let's create a concatenated text version.\n",
    "full_transcript_text = \"\"\n",
    "if raw_transcript_data:\n",
    "    full_transcript_text = \" \".join([segment['text'] for segment in raw_transcript_data])\n",
    "    logging.info(f\"Full transcript text concatenated. Length: {len(full_transcript_text)} characters.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "629c4f75",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-06-15 18:15:13,852 - INFO - Pydantic model 'VideoAnalysis' defined for topic identification.\n"
     ]
    }
   ],
   "source": [
    "# --- Pydantic Model for Topic Analysis ---\n",
    "class VideoAnalysis(BaseModel):\n",
    "    summary: str = Field(..., description=\"A brief summary of the video content, in 2-3 sentences.\")\n",
    "    main_topics: List[str] = Field(\n",
    "        ...,\n",
    "        description=f\"A list of {max_topics_to_identify} distinct main topics or themes discussed in the video. Each topic should be a short phrase (2-5 words) suitable for guiding clip extraction.\",\n",
    "        min_length=1, # Pydantic v2: min_items, for v1 it was min_items in Field. For Pydantic v2 List, use model_validator or check during Field\n",
    "        max_length=max_topics_to_identify # Pydantic v2: max_items\n",
    "    )\n",
    "    target_audience: Optional[str] = Field(None, description=\"The likely target audience for this video (e.g., 'Tech Enthusiasts', 'Business Leaders', 'Students').\")\n",
    "\n",
    "logging.info(\"Pydantic model 'VideoAnalysis' defined for topic identification.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "4db76dd3",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-06-15 18:15:16,589 - INFO - Initializing LLM for topic identification...\n",
      "2025-06-15 18:15:16,835 - INFO - Invoking LLM for topic identification...\n",
      "2025-06-15 18:15:19,017 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions \"HTTP/1.1 200 OK\"\n",
      "2025-06-15 18:15:19,036 - INFO - Video Summary: In this video, Rachel Woods discusses the transformative potential of AI in business operations, emphasizing the importance of defining processes for effective AI implementation. She highlights the roles of AI operators, visionaries, and implementers, advocating for a structured approach to integrating AI into business workflows.\n",
      "2025-06-15 18:15:19,036 - INFO - Identified Topics: ['AI in Business Operations', 'Roles in AI Implementation', 'Defining Processes for AI', 'AI Operator Importance', 'Mindset Shift for AI Usage']\n",
      "2025-06-15 18:15:19,037 - INFO - Target Audience: Business Leaders\n"
     ]
    }
   ],
   "source": [
    "# --- LLM for Topic Identification ---\n",
    "identified_topics = [] # Reset global variable\n",
    "\n",
    "if OPENAI_API_KEY and full_transcript_text: # Proceed only if API key and transcript text exist\n",
    "    try:\n",
    "        logging.info(\"Initializing LLM for topic identification...\")\n",
    "        # Using a potentially faster/cheaper model for this broader task\n",
    "        llm_topic_identifier = ChatOpenAI(\n",
    "            api_key=OPENAI_API_KEY,\n",
    "            model=\"gpt-4o-mini\", # or \"gpt-3.5-turbo\"\n",
    "            temperature=0.3\n",
    "        )\n",
    "\n",
    "        topic_prompt_text = (\n",
    "            f\"Please analyze the following video transcript. Provide a brief summary of the video's content, \"\n",
    "            f\"identify exactly {max_topics_to_identify} distinct main topics or themes that would be suitable for creating engaging short video clips, \"\n",
    "            f\"and suggest a likely target audience for this video.\\n\\n\"\n",
    "            f\"Each topic should be a concise phrase (2-5 words).\\n\\n\"\n",
    "            f\"Respond ONLY in the requested JSON structure.\\n\\n\"\n",
    "            f\"Transcript Text:\\n\\\"\\\"\\\"\\n{full_transcript_text[:15000]}\\n\\\"\\\"\\\"\" # Send a truncated version if too long for this model\n",
    "        ) # Limiting to first 15k chars for topic ID to manage token limits & cost\n",
    "\n",
    "        topic_messages = [\n",
    "            {\"role\": \"system\", \"content\": \"You are an expert video content analyst. Your task is to analyze a video transcript and identify its core themes. Respond ONLY with the requested JSON structure.\"},\n",
    "            {\"role\": \"user\", \"content\": topic_prompt_text}\n",
    "        ]\n",
    "\n",
    "        logging.info(\"Invoking LLM for topic identification...\")\n",
    "        structured_llm_topic = llm_topic_identifier.with_structured_output(VideoAnalysis)\n",
    "        analysis_result = structured_llm_topic.invoke(topic_messages)\n",
    "\n",
    "        if analysis_result:\n",
    "            logging.info(f\"Video Summary: {analysis_result.summary}\")\n",
    "            logging.info(f\"Identified Topics: {analysis_result.main_topics}\")\n",
    "            if analysis_result.target_audience:\n",
    "                logging.info(f\"Target Audience: {analysis_result.target_audience}\")\n",
    "            identified_topics = analysis_result.main_topics # Assign to global\n",
    "        else:\n",
    "            logging.warning(\"LLM for topic identification did not return a valid result.\")\n",
    "\n",
    "    except Exception as e:\n",
    "        logging.error(f\"An error occurred during topic identification with LLM: {e}\", exc_info=True)\n",
    "else:\n",
    "    if not OPENAI_API_KEY:\n",
    "        logging.warning(\"OpenAI API key not available. Skipping topic identification.\")\n",
    "    if not full_transcript_text:\n",
    "        logging.warning(\"Transcript text not available. Skipping topic identification.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "f2ec0fcd",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-06-15 18:15:23,244 - INFO - Pydantic models 'Segment' and 'SegmentResponse' defined for segment extraction.\n"
     ]
    }
   ],
   "source": [
    "# --- Pydantic Models for Segment Extraction ---\n",
    "class Segment(BaseModel):\n",
    "    start_time: int = Field(..., description=\"The start time of the segment in SECONDS from the beginning of the video.\")\n",
    "    end_time: int = Field(..., description=\"The end time of the segment in SECONDS from the beginning of the video.\")\n",
    "    title: str = Field(..., description=\"A concise, engaging title for this video segment (max 10 words).\")\n",
    "    description: str = Field(..., description=\"A brief 1-2 sentence description of what this segment is about.\")\n",
    "    # Duration can be calculated: end_time - start_time.\n",
    "    # LLM should focus on getting start_time and end_time to match the duration constraints.\n",
    "\n",
    "class SegmentResponse(BaseModel): # Renamed for clarity\n",
    "    segments: List[Segment] = Field(..., description=\"A list of identified video segments matching the criteria.\")\n",
    "\n",
    "logging.info(\"Pydantic models 'Segment' and 'SegmentResponse' defined for segment extraction.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "571d2976",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-06-15 18:15:25,334 - INFO - Initializing LLM for segment extraction...\n",
      "2025-06-15 18:15:25,336 - INFO - Processing topic 1/5: 'AI in Business Operations'\n",
      "2025-06-15 18:15:25,337 - INFO - Invoking LLM for segment extraction on topic: 'AI in Business Operations'...\n",
      "2025-06-15 18:15:30,198 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions \"HTTP/1.1 200 OK\"\n",
      "2025-06-15 18:15:30,204 - INFO - Found 3 segments for topic 'AI in Business Operations'.\n",
      "2025-06-15 18:15:30,205 - INFO - Processing topic 2/5: 'Roles in AI Implementation'\n",
      "2025-06-15 18:15:30,206 - INFO - Invoking LLM for segment extraction on topic: 'Roles in AI Implementation'...\n",
      "2025-06-15 18:15:36,724 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions \"HTTP/1.1 200 OK\"\n",
      "2025-06-15 18:15:36,726 - INFO - Found 6 segments for topic 'Roles in AI Implementation'.\n",
      "2025-06-15 18:15:36,727 - INFO - Processing topic 3/5: 'Defining Processes for AI'\n",
      "2025-06-15 18:15:36,728 - INFO - Invoking LLM for segment extraction on topic: 'Defining Processes for AI'...\n",
      "2025-06-15 18:15:42,333 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions \"HTTP/1.1 200 OK\"\n",
      "2025-06-15 18:15:42,335 - INFO - Found 4 segments for topic 'Defining Processes for AI'.\n",
      "2025-06-15 18:15:42,336 - INFO - Processing topic 4/5: 'AI Operator Importance'\n",
      "2025-06-15 18:15:42,337 - INFO - Invoking LLM for segment extraction on topic: 'AI Operator Importance'...\n",
      "2025-06-15 18:15:45,518 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions \"HTTP/1.1 200 OK\"\n",
      "2025-06-15 18:15:45,520 - INFO - Found 2 segments for topic 'AI Operator Importance'.\n",
      "2025-06-15 18:15:45,521 - INFO - Processing topic 5/5: 'Mindset Shift for AI Usage'\n",
      "2025-06-15 18:15:45,522 - INFO - Invoking LLM for segment extraction on topic: 'Mindset Shift for AI Usage'...\n",
      "2025-06-15 18:15:48,617 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions \"HTTP/1.1 200 OK\"\n",
      "2025-06-15 18:15:48,624 - INFO - Found 1 segments for topic 'Mindset Shift for AI Usage'.\n",
      "2025-06-15 18:15:48,625 - INFO - Total segments extracted across all topics: 16\n"
     ]
    }
   ],
   "source": [
    "# --- LLM for Segment Extraction (Looping Per Topic) ---\n",
    "all_extracted_segments = [] # Reset global variable\n",
    "\n",
    "if OPENAI_API_KEY and raw_transcript_data and identified_topics: # Proceed if all prereqs are met\n",
    "    try:\n",
    "        logging.info(\"Initializing LLM for segment extraction...\")\n",
    "        # Using a potentially more powerful model for precision in segmenting\n",
    "        llm_segment_extractor = ChatOpenAI(\n",
    "            api_key=OPENAI_API_KEY,\n",
    "            model=\"gpt-4o-mini\", # or \"gpt-4o-mini\" if cost/speed is a concern and quality is acceptable\n",
    "            temperature=0.5 # Slightly lower temp for more factual extraction\n",
    "        )\n",
    "        structured_llm_segment = llm_segment_extractor.with_structured_output(SegmentResponse)\n",
    "\n",
    "        for topic_index, current_topic in enumerate(identified_topics):\n",
    "            logging.info(f\"Processing topic {topic_index + 1}/{len(identified_topics)}: '{current_topic}'\")\n",
    "\n",
    "            segment_prompt_text = (\n",
    "                f\"From the provided full video transcript (which is a list of dictionaries, each with 'text', 'start' in seconds, and 'duration' in seconds), \"\n",
    "                f\"identify all distinct segments that are specifically and clearly about the topic: '{current_topic}'.\\n\\n\"\n",
    "                f\"For each segment you identify:\\n\"\n",
    "                f\"1. It MUST be between {segment_duration_min_seconds} and {segment_duration_max_seconds} seconds in duration.\\n\"\n",
    "                f\"2. Provide extremely accurate 'start_time' and 'end_time' in whole SECONDS from the beginning of the video.\\n\"\n",
    "                f\"3. Create a concise, engaging 'title' (max 10 words, relevant to the topic and segment content).\\n\"\n",
    "                f\"4. Write a brief 'description' (1-2 sentences) of what this specific segment covers.\\n\\n\"\n",
    "                f\"Carefully review the timestamps in the transcript to ensure accuracy. Prioritize segments that are coherent and self-contained.\\n\\n\"\n",
    "                f\"Respond ONLY with a JSON object containing a single key 'segments', where 'segments' is a list of these segment objects. \"\n",
    "                f\"If no segments match the criteria for this topic, return an empty list for 'segments'.\\n\\n\"\n",
    "                f\"Topic to focus on: {current_topic}\\n\\n\"\n",
    "                f\"Full Video Transcript (list of dictionaries):\\n\\\"\\\"\\\"\\n{raw_transcript_data}\\n\\\"\\\"\\\"\" # Send the structured transcript\n",
    "            )\n",
    "\n",
    "            segment_messages = [\n",
    "                {\"role\": \"system\", \"content\": \"You are a YouTube shorts content producer. Your goal is to find precise video segments for a given topic based on a transcript. Respond ONLY with the requested JSON structure.\"},\n",
    "                {\"role\": \"user\", \"content\": segment_prompt_text}\n",
    "            ]\n",
    "\n",
    "            logging.info(f\"Invoking LLM for segment extraction on topic: '{current_topic}'...\")\n",
    "            try:\n",
    "                segment_extraction_result = structured_llm_segment.invoke(segment_messages)\n",
    "                if segment_extraction_result and segment_extraction_result.segments:\n",
    "                    logging.info(f\"Found {len(segment_extraction_result.segments)} segments for topic '{current_topic}'.\")\n",
    "                    for seg_obj in segment_extraction_result.segments:\n",
    "                        # Add calculated duration and topic to the segment data before storing\n",
    "                        calculated_duration = seg_obj.end_time - seg_obj.start_time\n",
    "                        # Convert Pydantic model to dict for easier appending if needed, or store as objects\n",
    "                        segment_dict = seg_obj.model_dump() # Pydantic V2\n",
    "                        segment_dict['calculated_duration'] = calculated_duration\n",
    "                        segment_dict['topic'] = current_topic # Keep track of which topic it came from\n",
    "                        all_extracted_segments.append(segment_dict)\n",
    "                else:\n",
    "                    logging.info(f\"No segments found by LLM for topic '{current_topic}'.\")\n",
    "            except Exception as e_invoke:\n",
    "                logging.error(f\"LLM invocation failed for topic '{current_topic}': {e_invoke}\", exc_info=True)\n",
    "\n",
    "    except Exception as e:\n",
    "        logging.error(f\"An error occurred during the segment extraction process: {e}\", exc_info=True)\n",
    "else:\n",
    "    if not OPENAI_API_KEY:\n",
    "        logging.warning(\"OpenAI API key not available. Skipping segment extraction.\")\n",
    "    if not raw_transcript_data:\n",
    "        logging.warning(\"Transcript data not available. Skipping segment extraction.\")\n",
    "    if not identified_topics:\n",
    "        logging.warning(\"No topics identified. Skipping segment extraction.\")\n",
    "\n",
    "logging.info(f\"Total segments extracted across all topics: {len(all_extracted_segments)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "d6c8bacf",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-06-15 18:15:52,233 - INFO - Starting FFMPEG clip generation for 16 segments...\n",
      "2025-06-15 18:15:52,234 - INFO - \n",
      "Processing segment 1/16 for topic 'AI in Business Operations':\n",
      "2025-06-15 18:15:52,234 - INFO -   Source Video: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\n",
      "2025-06-15 18:15:52,234 - INFO -   Start: 590s, End: 649s, Duration: 59s\n",
      "2025-06-15 18:15:52,235 - INFO -   Segment Title (Raw): Defining AI Operations in Business\n",
      "2025-06-15 18:15:52,235 - INFO -   Output File: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Defining_AI_Operations_in_Business_59s_0.mp4\n",
      "2025-06-15 18:15:52,235 - INFO -   Executing FFMPEG command: ffmpeg -y -hide_banner -i downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4 -ss 590 -to 649 -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Defining_AI_Operations_in_Business_59s_0.mp4\n",
      "2025-06-15 18:15:55,983 - INFO -   Successfully generated clip: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Defining_AI_Operations_in_Business_59s_0.mp4\n",
      "2025-06-15 18:15:55,984 - INFO - \n",
      "Processing segment 2/16 for topic 'AI in Business Operations':\n",
      "2025-06-15 18:15:55,984 - INFO -   Source Video: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\n",
      "2025-06-15 18:15:55,984 - INFO -   Start: 675s, End: 734s, Duration: 59s\n",
      "2025-06-15 18:15:55,984 - INFO -   Segment Title (Raw): Roles in AI Implementation\n",
      "2025-06-15 18:15:55,985 - INFO -   Output File: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Roles_in_AI_Implementation_59s_1.mp4\n",
      "2025-06-15 18:15:55,985 - INFO -   Executing FFMPEG command: ffmpeg -y -hide_banner -i downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4 -ss 675 -to 734 -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Roles_in_AI_Implementation_59s_1.mp4\n",
      "2025-06-15 18:15:59,801 - INFO -   Successfully generated clip: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Roles_in_AI_Implementation_59s_1.mp4\n",
      "2025-06-15 18:15:59,801 - INFO - \n",
      "Processing segment 3/16 for topic 'AI in Business Operations':\n",
      "2025-06-15 18:15:59,802 - INFO -   Source Video: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\n",
      "2025-06-15 18:15:59,803 - INFO -   Start: 1195s, End: 1255s, Duration: 60s\n",
      "2025-06-15 18:15:59,803 - INFO -   Segment Title (Raw): Crafting AI Processes\n",
      "2025-06-15 18:15:59,803 - INFO -   Output File: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Crafting_AI_Processes_60s_2.mp4\n",
      "2025-06-15 18:15:59,804 - INFO -   Executing FFMPEG command: ffmpeg -y -hide_banner -i downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4 -ss 1195 -to 1255 -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Crafting_AI_Processes_60s_2.mp4\n",
      "2025-06-15 18:16:04,651 - INFO -   Successfully generated clip: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Crafting_AI_Processes_60s_2.mp4\n",
      "2025-06-15 18:16:04,651 - INFO - \n",
      "Processing segment 4/16 for topic 'Roles in AI Implementation':\n",
      "2025-06-15 18:16:04,651 - INFO -   Source Video: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\n",
      "2025-06-15 18:16:04,651 - INFO -   Start: 614s, End: 670s, Duration: 56s\n",
      "2025-06-15 18:16:04,652 - INFO -   Segment Title (Raw): Defining Roles in AI Implementation\n",
      "2025-06-15 18:16:04,652 - INFO -   Output File: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Defining_Roles_in_AI_Implementation_56s_3.mp4\n",
      "2025-06-15 18:16:04,652 - INFO -   Executing FFMPEG command: ffmpeg -y -hide_banner -i downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4 -ss 614 -to 670 -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Defining_Roles_in_AI_Implementation_56s_3.mp4\n",
      "2025-06-15 18:16:08,084 - INFO -   Successfully generated clip: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Defining_Roles_in_AI_Implementation_56s_3.mp4\n",
      "2025-06-15 18:16:08,084 - INFO - \n",
      "Processing segment 5/16 for topic 'Roles in AI Implementation':\n",
      "2025-06-15 18:16:08,085 - INFO -   Source Video: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\n",
      "2025-06-15 18:16:08,085 - INFO -   Start: 670s, End: 724s, Duration: 54s\n",
      "2025-06-15 18:16:08,086 - INFO -   Segment Title (Raw): The Importance of AI Operators\n",
      "2025-06-15 18:16:08,086 - INFO -   Output File: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_The_Importance_of_AI_Operators_54s_4.mp4\n",
      "2025-06-15 18:16:08,087 - INFO -   Executing FFMPEG command: ffmpeg -y -hide_banner -i downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4 -ss 670 -to 724 -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_The_Importance_of_AI_Operators_54s_4.mp4\n",
      "2025-06-15 18:16:11,769 - INFO -   Successfully generated clip: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_The_Importance_of_AI_Operators_54s_4.mp4\n",
      "2025-06-15 18:16:11,769 - INFO - \n",
      "Processing segment 6/16 for topic 'Roles in AI Implementation':\n",
      "2025-06-15 18:16:11,770 - INFO -   Source Video: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\n",
      "2025-06-15 18:16:11,770 - INFO -   Start: 724s, End: 780s, Duration: 56s\n",
      "2025-06-15 18:16:11,770 - INFO -   Segment Title (Raw): Visionary Role in AI Integration\n",
      "2025-06-15 18:16:11,770 - INFO -   Output File: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Visionary_Role_in_AI_Integration_56s_5.mp4\n",
      "2025-06-15 18:16:11,770 - INFO -   Executing FFMPEG command: ffmpeg -y -hide_banner -i downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4 -ss 724 -to 780 -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Visionary_Role_in_AI_Integration_56s_5.mp4\n",
      "2025-06-15 18:16:15,615 - INFO -   Successfully generated clip: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Visionary_Role_in_AI_Integration_56s_5.mp4\n",
      "2025-06-15 18:16:15,615 - INFO - \n",
      "Processing segment 7/16 for topic 'Roles in AI Implementation':\n",
      "2025-06-15 18:16:15,616 - INFO -   Source Video: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\n",
      "2025-06-15 18:16:15,616 - INFO -   Start: 780s, End: 836s, Duration: 56s\n",
      "2025-06-15 18:16:15,617 - INFO -   Segment Title (Raw): Technical Implementation in AI\n",
      "2025-06-15 18:16:15,617 - INFO -   Output File: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Technical_Implementation_in_AI_56s_6.mp4\n",
      "2025-06-15 18:16:15,618 - INFO -   Executing FFMPEG command: ffmpeg -y -hide_banner -i downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4 -ss 780 -to 836 -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Technical_Implementation_in_AI_56s_6.mp4\n",
      "2025-06-15 18:16:19,735 - INFO -   Successfully generated clip: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Technical_Implementation_in_AI_56s_6.mp4\n",
      "2025-06-15 18:16:19,735 - INFO - \n",
      "Processing segment 8/16 for topic 'Roles in AI Implementation':\n",
      "2025-06-15 18:16:19,736 - INFO -   Source Video: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\n",
      "2025-06-15 18:16:19,737 - INFO -   Start: 836s, End: 892s, Duration: 56s\n",
      "2025-06-15 18:16:19,737 - INFO -   Segment Title (Raw): Navigating AI Implementation Challenges\n",
      "2025-06-15 18:16:19,737 - INFO -   Output File: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Navigating_AI_Implementation_Challenges_56s_7.mp4\n",
      "2025-06-15 18:16:19,739 - INFO -   Executing FFMPEG command: ffmpeg -y -hide_banner -i downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4 -ss 836 -to 892 -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Navigating_AI_Implementation_Challenges_56s_7.mp4\n",
      "2025-06-15 18:16:24,016 - INFO -   Successfully generated clip: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Navigating_AI_Implementation_Challenges_56s_7.mp4\n",
      "2025-06-15 18:16:24,016 - INFO - \n",
      "Processing segment 9/16 for topic 'Roles in AI Implementation':\n",
      "2025-06-15 18:16:24,017 - INFO -   Source Video: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\n",
      "2025-06-15 18:16:24,017 - INFO -   Start: 892s, End: 948s, Duration: 56s\n",
      "2025-06-15 18:16:24,017 - INFO -   Segment Title (Raw): Feedback and Continuous Improvement in AI\n",
      "2025-06-15 18:16:24,018 - INFO -   Output File: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Feedback_and_Continuous_Improvement_in_AI_56s_8.mp4\n",
      "2025-06-15 18:16:24,018 - INFO -   Executing FFMPEG command: ffmpeg -y -hide_banner -i downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4 -ss 892 -to 948 -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Feedback_and_Continuous_Improvement_in_AI_56s_8.mp4\n",
      "2025-06-15 18:16:28,312 - INFO -   Successfully generated clip: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Feedback_and_Continuous_Improvement_in_AI_56s_8.mp4\n",
      "2025-06-15 18:16:28,313 - INFO - \n",
      "Processing segment 10/16 for topic 'Defining Processes for AI':\n",
      "2025-06-15 18:16:28,313 - INFO -   Source Video: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\n",
      "2025-06-15 18:16:28,314 - INFO -   Start: 590s, End: 646s, Duration: 56s\n",
      "2025-06-15 18:16:28,314 - INFO -   Segment Title (Raw): Defining AI Operations\n",
      "2025-06-15 18:16:28,315 - INFO -   Output File: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Defining_AI_Operations_56s_9.mp4\n",
      "2025-06-15 18:16:28,315 - INFO -   Executing FFMPEG command: ffmpeg -y -hide_banner -i downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4 -ss 590 -to 646 -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Defining_AI_Operations_56s_9.mp4\n",
      "2025-06-15 18:16:32,039 - INFO -   Successfully generated clip: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Defining_AI_Operations_56s_9.mp4\n",
      "2025-06-15 18:16:32,039 - INFO - \n",
      "Processing segment 11/16 for topic 'Defining Processes for AI':\n",
      "2025-06-15 18:16:32,040 - INFO -   Source Video: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\n",
      "2025-06-15 18:16:32,040 - INFO -   Start: 675s, End: 734s, Duration: 59s\n",
      "2025-06-15 18:16:32,040 - INFO -   Segment Title (Raw): Roles in AI Implementation\n",
      "2025-06-15 18:16:32,040 - INFO -   Output File: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Roles_in_AI_Implementation_59s_10.mp4\n",
      "2025-06-15 18:16:32,041 - INFO -   Executing FFMPEG command: ffmpeg -y -hide_banner -i downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4 -ss 675 -to 734 -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Roles_in_AI_Implementation_59s_10.mp4\n",
      "2025-06-15 18:16:36,339 - INFO -   Successfully generated clip: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Roles_in_AI_Implementation_59s_10.mp4\n",
      "2025-06-15 18:16:36,340 - INFO - \n",
      "Processing segment 12/16 for topic 'Defining Processes for AI':\n",
      "2025-06-15 18:16:36,340 - INFO -   Source Video: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\n",
      "2025-06-15 18:16:36,341 - INFO -   Start: 1735s, End: 1795s, Duration: 60s\n",
      "2025-06-15 18:16:36,341 - INFO -   Segment Title (Raw): Crafting Processes for AI\n",
      "2025-06-15 18:16:36,342 - INFO -   Output File: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Crafting_Processes_for_AI_60s_11.mp4\n",
      "2025-06-15 18:16:36,342 - INFO -   Executing FFMPEG command: ffmpeg -y -hide_banner -i downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4 -ss 1735 -to 1795 -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Crafting_Processes_for_AI_60s_11.mp4\n",
      "2025-06-15 18:16:43,148 - INFO -   Successfully generated clip: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Crafting_Processes_for_AI_60s_11.mp4\n",
      "2025-06-15 18:16:43,148 - INFO - \n",
      "Processing segment 13/16 for topic 'Defining Processes for AI':\n",
      "2025-06-15 18:16:43,149 - INFO -   Source Video: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\n",
      "2025-06-15 18:16:43,149 - INFO -   Start: 1971s, End: 2039s, Duration: 68s\n",
      "2025-06-15 18:16:43,149 - INFO -   Segment Title (Raw): AI Process Documentation\n",
      "2025-06-15 18:16:43,150 - INFO -   Output File: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_AI_Process_Documentation_68s_12.mp4\n",
      "2025-06-15 18:16:43,150 - INFO -   Executing FFMPEG command: ffmpeg -y -hide_banner -i downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4 -ss 1971 -to 2039 -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_AI_Process_Documentation_68s_12.mp4\n",
      "2025-06-15 18:16:50,697 - INFO -   Successfully generated clip: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_AI_Process_Documentation_68s_12.mp4\n",
      "2025-06-15 18:16:50,697 - INFO - \n",
      "Processing segment 14/16 for topic 'AI Operator Importance':\n",
      "2025-06-15 18:16:50,698 - INFO -   Source Video: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\n",
      "2025-06-15 18:16:50,698 - INFO -   Start: 590s, End: 649s, Duration: 59s\n",
      "2025-06-15 18:16:50,698 - INFO -   Segment Title (Raw): The Role of AI Operators\n",
      "2025-06-15 18:16:50,700 - INFO -   Output File: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_The_Role_of_AI_Operators_59s_13.mp4\n",
      "2025-06-15 18:16:50,700 - INFO -   Executing FFMPEG command: ffmpeg -y -hide_banner -i downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4 -ss 590 -to 649 -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_The_Role_of_AI_Operators_59s_13.mp4\n",
      "2025-06-15 18:16:54,474 - INFO -   Successfully generated clip: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_The_Role_of_AI_Operators_59s_13.mp4\n",
      "2025-06-15 18:16:54,475 - INFO - \n",
      "Processing segment 15/16 for topic 'AI Operator Importance':\n",
      "2025-06-15 18:16:54,475 - INFO -   Source Video: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\n",
      "2025-06-15 18:16:54,476 - INFO -   Start: 674s, End: 734s, Duration: 60s\n",
      "2025-06-15 18:16:54,476 - INFO -   Segment Title (Raw): AI Operator Responsibilities\n",
      "2025-06-15 18:16:54,477 - INFO -   Output File: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_AI_Operator_Responsibilities_60s_14.mp4\n",
      "2025-06-15 18:16:54,477 - INFO -   Executing FFMPEG command: ffmpeg -y -hide_banner -i downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4 -ss 674 -to 734 -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_AI_Operator_Responsibilities_60s_14.mp4\n",
      "2025-06-15 18:16:58,421 - INFO -   Successfully generated clip: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_AI_Operator_Responsibilities_60s_14.mp4\n",
      "2025-06-15 18:16:58,422 - INFO - \n",
      "Processing segment 16/16 for topic 'Mindset Shift for AI Usage':\n",
      "2025-06-15 18:16:58,422 - INFO -   Source Video: downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\n",
      "2025-06-15 18:16:58,422 - INFO -   Start: 258s, End: 317s, Duration: 59s\n",
      "2025-06-15 18:16:58,424 - INFO -   Segment Title (Raw): Mindset Shift for AI Usage\n",
      "2025-06-15 18:16:58,424 - INFO -   Output File: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Mindset_Shift_for_AI_Usage_59s_15.mp4\n",
      "2025-06-15 18:16:58,425 - INFO -   Executing FFMPEG command: ffmpeg -y -hide_banner -i downloaded_video\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4 -ss 258 -to 317 -c:v libx264 -crf 18 -preset medium -c:a aac -b:a 192k generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Mindset_Shift_for_AI_Usage_59s_15.mp4\n",
      "2025-06-15 18:17:01,404 - INFO -   Successfully generated clip: generated_clips\\The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods_Mindset_Shift_for_AI_Usage_59s_15.mp4\n",
      "2025-06-15 18:17:01,404 - INFO - \n",
      "FFMPEG processing finished. Successful clips: 16, Failed clips: 0\n",
      "2025-06-15 18:17:01,404 - INFO - --- YouTube Clipper Process Complete ---\n",
      "2025-06-15 18:17:01,405 - INFO - Review your generated clips in the 'generated_clips' directory.\n"
     ]
    }
   ],
   "source": [
    "# --- FFMPEG Snippet Generation ---\n",
    "if not all_extracted_segments:\n",
    "    logging.warning(\"No segments were extracted by the LLM. Skipping FFMPEG processing.\")\n",
    "elif not downloaded_video_path or not os.path.exists(downloaded_video_path):\n",
    "    logging.error(f\"Downloaded video path is not set or file does not exist: {downloaded_video_path}. Cannot generate clips.\")\n",
    "elif not video_base_title:\n",
    "    logging.error(\"Video base title is not set. Cannot generate clips.\")\n",
    "else:\n",
    "    logging.info(f\"Starting FFMPEG clip generation for {len(all_extracted_segments)} segments...\")\n",
    "    successful_clips = 0\n",
    "    failed_clips = 0\n",
    "\n",
    "    for i, segment_data in enumerate(all_extracted_segments):\n",
    "        try:\n",
    "            start_time = segment_data['start_time']\n",
    "            end_time = segment_data['end_time']\n",
    "            segment_title_raw = segment_data['title']\n",
    "            # description = segment_data['description'] # Available if needed\n",
    "            duration = segment_data.get('calculated_duration', end_time - start_time) # Use calculated or re-calculate\n",
    "\n",
    "            # Sanitize segment title for filename\n",
    "            segment_title_sanitized = \"\".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in segment_title_raw).rstrip().replace(\" \", \"_\")\n",
    "            if not segment_title_sanitized: # Handle empty titles\n",
    "                segment_title_sanitized = f\"segment_{i}\"\n",
    "\n",
    "            output_clip_filename = f\"{video_base_title}_{segment_title_sanitized}_{duration}s_{i}.mp4\"\n",
    "            output_clip_path = os.path.join(clips_output_directory, output_clip_filename)\n",
    "\n",
    "            logging.info(f\"\\nProcessing segment {i + 1}/{len(all_extracted_segments)} for topic '{segment_data.get('topic','N/A')}':\")\n",
    "            logging.info(f\"  Source Video: {downloaded_video_path}\")\n",
    "            logging.info(f\"  Start: {start_time}s, End: {end_time}s, Duration: {duration}s\")\n",
    "            logging.info(f\"  Segment Title (Raw): {segment_title_raw}\")\n",
    "            logging.info(f\"  Output File: {output_clip_path}\")\n",
    "\n",
    "            # Construct FFMPEG command (ensure paths with spaces are quoted)\n",
    "            command = [\n",
    "                \"ffmpeg\",\n",
    "                \"-y\",  # Overwrite output files without asking\n",
    "                \"-hide_banner\",\n",
    "                \"-i\", downloaded_video_path,\n",
    "                \"-ss\", str(start_time),\n",
    "                \"-to\", str(end_time),\n",
    "                \"-c:v\", \"libx264\",\n",
    "                \"-crf\", \"18\",      # Constant Rate Factor (lower is better quality, 18 is visually lossless for many)\n",
    "                \"-preset\", \"medium\", # Encoding speed vs. compression (medium is a good balance)\n",
    "                \"-c:a\", \"aac\",\n",
    "                \"-b:a\", \"192k\",    # Audio bitrate\n",
    "                output_clip_path\n",
    "            ]\n",
    "            # Using list for command is safer with subprocess than f-string for complex commands\n",
    "\n",
    "            logging.info(f\"  Executing FFMPEG command: {' '.join(command)}\")\n",
    "            result = subprocess.run(command, capture_output=True, text=True, check=False)\n",
    "\n",
    "            if result.returncode == 0:\n",
    "                logging.info(f\"  Successfully generated clip: {output_clip_path}\")\n",
    "                successful_clips += 1\n",
    "            else:\n",
    "                logging.error(f\"  Error generating clip: {output_clip_path}\")\n",
    "                logging.error(f\"  FFMPEG STDOUT: {result.stdout.strip()}\")\n",
    "                logging.error(f\"  FFMPEG STDERR: {result.stderr.strip()}\")\n",
    "                failed_clips += 1\n",
    "        except KeyError as ke:\n",
    "            logging.error(f\"  Missing expected key in segment_data for segment {i}: {ke}. Segment data: {segment_data}\", exc_info=True)\n",
    "            failed_clips += 1\n",
    "        except FileNotFoundError:\n",
    "            logging.error(\"  Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your system's PATH.\")\n",
    "            break # Stop processing further clips if ffmpeg is not found\n",
    "        except Exception as e:\n",
    "            logging.error(f\"  An unexpected error occurred generating clip for segment {i}: {e}\", exc_info=True)\n",
    "            failed_clips += 1\n",
    "    logging.info(f\"\\nFFMPEG processing finished. Successful clips: {successful_clips}, Failed clips: {failed_clips}\")\n",
    "    \n",
    "logging.info(\"--- YouTube Clipper Process Complete ---\")\n",
    "if successful_clips > 0 :\n",
    "    logging.info(f\"Review your generated clips in the '{clips_output_directory}' directory.\")\n",
    "else:\n",
    "    logging.info(\"No clips were successfully generated. Please review the logs for errors.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "df1d6995",
   "metadata": {},
   "outputs": [],
   "source": [
    "import stable_whisper\n",
    "# Or: import whisper\n",
    "import ffmpeg\n",
    "import json\n",
    "import os\n",
    "\n",
    "def extract_transcript_from_mp4(mp4_filepath, output_json_filepath=None, model_name=\"large\"):\n",
    "    \"\"\"\n",
    "    Extracts transcript with timestamps from an MP4 file using Whisper (via stable-ts).\n",
    "    ... (rest of the docstring) ...\n",
    "    \"\"\"\n",
    "    if not os.path.exists(mp4_filepath):\n",
    "        print(f\"DEBUG: Error - MP4 file not found at {mp4_filepath}\")\n",
    "        return None\n",
    "\n",
    "    print(f\"DEBUG: Loading Whisper model '{model_name}'...\")\n",
    "    try:\n",
    "        model = stable_whisper.load_model(model_name)\n",
    "        # model = whisper.load_model(model_name) # If using original openai-whisper\n",
    "    except Exception as e:\n",
    "        print(f\"DEBUG: Error loading Whisper model: {e}\")\n",
    "        return None\n",
    "\n",
    "    print(f\"DEBUG: Transcribing '{mp4_filepath}'...\")\n",
    "    try:\n",
    "        # For stable-whisper, you might want to explore its specific transcribe options\n",
    "        # result = model.transcribe(mp4_filepath, fp16=False, word_timestamps=True, regroup=False) # Example for stable-whisper\n",
    "        result = model.transcribe(mp4_filepath, fp16=False) # Standard call\n",
    "\n",
    "        # --- DEBUGGING THE RESULT ---\n",
    "        print(\"\\n--- DEBUG: Full Whisper Transcription Result ---\")\n",
    "        if result is None:\n",
    "            print(\"DEBUG: result is None\")\n",
    "        else:\n",
    "            print(f\"DEBUG: type(result) = {type(result)}\")\n",
    "            if isinstance(result, dict):\n",
    "                print(f\"DEBUG: result.keys() = {result.keys()}\")\n",
    "                if 'text' in result:\n",
    "                    print(f\"DEBUG: result['text'] (first 500 chars) = {result['text'][:500]}\")\n",
    "                if 'segments' in result:\n",
    "                    print(f\"DEBUG: type(result['segments']) = {type(result['segments'])}\")\n",
    "                    if result['segments']:\n",
    "                        print(f\"DEBUG: len(result['segments']) = {len(result['segments'])}\")\n",
    "                        print(f\"DEBUG: First segment in result['segments'] = {result['segments'][0] if len(result['segments']) > 0 else 'N/A'}\")\n",
    "                    else:\n",
    "                        print(\"DEBUG: result['segments'] is empty.\")\n",
    "                else:\n",
    "                    print(\"DEBUG: 'segments' key NOT FOUND in result.\")\n",
    "            else:\n",
    "                # If using stable_whisper, the result might be a different object type\n",
    "                # For example, it might be a stable_whisper.result.WhisperResult object\n",
    "                # You'd need to inspect its attributes or methods as per stable_whisper docs\n",
    "                print(f\"DEBUG: result is not a dictionary. Raw result object: {result}\")\n",
    "                # Example for stable_whisper if result is an object:\n",
    "                # try:\n",
    "                #     print(f\"DEBUG: Accessing result.segments directly (for stable_whisper object)\")\n",
    "                #     segments_from_result_object = result.segments\n",
    "                #     if segments_from_result_object:\n",
    "                #         print(f\"DEBUG: len(segments_from_result_object) = {len(segments_from_result_object)}\")\n",
    "                #         print(f\"DEBUG: First segment from result object = {segments_from_result_object[0]}\")\n",
    "                #     else:\n",
    "                #         print(\"DEBUG: result.segments (object attribute) is empty or None\")\n",
    "                # except AttributeError:\n",
    "                #     print(\"DEBUG: result object does not have a .segments attribute directly accessible this way.\")\n",
    "\n",
    "\n",
    "        # --- Processing the result ---\n",
    "        transcript_segments = []\n",
    "        # Adjust access based on what the DEBUG prints show for 'result'\n",
    "        # The original openai-whisper returns a dict. stable-whisper might return an object\n",
    "        # that can be converted to a dict or accessed differently.\n",
    "\n",
    "        # Safely try to access segments\n",
    "        segments_data = None\n",
    "        if isinstance(result, dict) and 'segments' in result:\n",
    "            segments_data = result['segments']\n",
    "        elif hasattr(result, 'segments'): # For objects like stable_whisper.result.WhisperResult\n",
    "             # stable_whisper's .segments is often a generator or list of segment objects\n",
    "             # We might need to iterate and convert them to dicts\n",
    "             print(\"DEBUG: Accessing segments from result object attribute.\")\n",
    "             temp_segments = []\n",
    "             try:\n",
    "                for seg_obj in result.segments: # Iterate if it's a generator/list of objects\n",
    "                    # Convert stable-whisper segment object to a dictionary if needed\n",
    "                    # This depends on the exact structure of seg_obj from stable_whisper\n",
    "                    # A common pattern is:\n",
    "                    temp_segments.append({\n",
    "                        'text': getattr(seg_obj, 'text', '').strip(),\n",
    "                        'start': round(getattr(seg_obj, 'start', 0.0), 2),\n",
    "                        'end': round(getattr(seg_obj, 'end', 0.0), 2)\n",
    "                    })\n",
    "                segments_data = temp_segments\n",
    "                if segments_data:\n",
    "                    print(f\"DEBUG: Successfully extracted {len(segments_data)} segments from stable_whisper result object.\")\n",
    "                else:\n",
    "                    print(\"DEBUG: Failed to extract segments from stable_whisper result object or it was empty.\")\n",
    "\n",
    "             except Exception as e_seg_obj:\n",
    "                 print(f\"DEBUG: Error processing segments from stable_whisper object: {e_seg_obj}\")\n",
    "        \n",
    "        if segments_data: # Check if segments_data was successfully populated\n",
    "            for segment in segments_data:\n",
    "                # Ensure segment is a dict and has the required keys before proceeding\n",
    "                if isinstance(segment, dict) and 'text' in segment and 'start' in segment and 'end' in segment:\n",
    "                    transcript_segments.append({\n",
    "                        \"text\": segment['text'].strip(),\n",
    "                        \"start\": round(segment['start'], 2),\n",
    "                        \"end\": round(segment['end'], 2),\n",
    "                        \"duration\": round(segment['end'] - segment['start'], 2)\n",
    "                    })\n",
    "                else:\n",
    "                    print(f\"DEBUG: Skipping malformed segment: {segment}\")\n",
    "\n",
    "            if transcript_segments:\n",
    "                print(f\"DEBUG: Transcription processing successful. Found {len(transcript_segments)} formatted segments.\")\n",
    "            else:\n",
    "                print(\"DEBUG: No valid segments found after processing the transcription result.\")\n",
    "                # This could happen if segments_data was populated but items were malformed\n",
    "        else:\n",
    "            print(\"DEBUG: 'segments_data' is None or empty. Transcription did not produce expected segments.\")\n",
    "            return None # Exit if no segments were found in the result\n",
    "\n",
    "    except Exception as e:\n",
    "        print(f\"DEBUG: Error during transcription processing: {e}\")\n",
    "        return None\n",
    "\n",
    "    # Save to JSON\n",
    "    if not transcript_segments: # Double check before saving\n",
    "        print(\"DEBUG: No transcript segments to save.\")\n",
    "        return None\n",
    "\n",
    "    if output_json_filepath is None:\n",
    "        base, ext = os.path.splitext(mp4_filepath)\n",
    "        output_json_filepath = f\"{base}_transcript_whisper.json\"\n",
    "\n",
    "    try:\n",
    "        with open(output_json_filepath, \"w\", encoding=\"utf-8\") as f:\n",
    "            json.dump(transcript_segments, f, ensure_ascii=False, indent=4)\n",
    "        print(f\"DEBUG: Transcript saved to: {output_json_filepath}\")\n",
    "    except Exception as e:\n",
    "        print(f\"DEBUG: Error saving transcript to JSON: {e}\")\n",
    "\n",
    "    return transcript_segments\n",
    "\n",
    "# --- Example Usage ---\n",
    "if __name__ == \"__main__\":\n",
    "    video_download_directory_for_test = \"downloaded_video\"\n",
    "    test_video_filename = \"The_High-Paying_AI_Job_Nobody_Knows_About__Yet__ft__Rachel_Woods.mp4\"\n",
    "    mp4_file_to_test = os.path.join(video_download_directory_for_test, test_video_filename)\n",
    "\n",
    "    print(f\"Attempting to test transcription with: {mp4_file_to_test}\")\n",
    "\n",
    "    if not os.path.exists(mp4_file_to_test):\n",
    "        print(f\"Error: The test MP4 file '{mp4_file_to_test}' does not exist. Please check the path.\")\n",
    "    else:\n",
    "        transcript_data = extract_transcript_from_mp4(mp4_file_to_test, model_name=\"large\")\n",
    "\n",
    "        if transcript_data:\n",
    "            print(\"\\n--- First 5 Transcript Segments (from example usage) ---\")\n",
    "            for i, segment_info in enumerate(transcript_data[:5]):\n",
    "                print(f\"Segment {i+1}:\")\n",
    "                print(f\"  Start: {segment_info['start']:.2f}s, End: {segment_info['end']:.2f}s, Duration: {segment_info['duration']:.2f}s\")\n",
    "                print(f\"  Text: {segment_info['text']}\")\n",
    "        else:\n",
    "            print(\"\\nExample usage: Transcription failed or produced no data.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1078de97",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "print(f\"PyTorch version: {torch.__version__}\")\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"CUDA version PyTorch was compiled with: {torch.version.cuda}\") # This should ideally be 12.1\n",
    "    print(f\"Number of GPUs: {torch.cuda.device_count()}\")\n",
    "    print(f\"Current GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}\")\n",
    "else:\n",
    "    print(\"CUDA is NOT available to PyTorch. Check installation and compatibility.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5117e89f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import gc # Garbage Collector interface\n",
    "\n",
    "if 'model' in locals() or 'model' in globals():\n",
    "    del model # Remove your reference to the model object\n",
    "    print(\"Deleted model variable.\")\n",
    "\n",
    "# Explicitly clear PyTorch's CUDA memory cache\n",
    "if torch.cuda.is_available():\n",
    "    torch.cuda.empty_cache()\n",
    "    print(\"Cleared PyTorch CUDA cache.\")\n",
    "    # You can also run garbage collection for good measure,\n",
    "    # though empty_cache is usually the more direct for GPU.\n",
    "    gc.collect()\n",
    "    print(\"Ran Python garbage collection.\")\n",
    "\n",
    "# After this, check nvidia-smi. The memory used by that model should be freed.\n",
    "# Note: Some base CUDA context memory (~few hundred MBs) might remain as long as PyTorch is active."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

```

`video_clipper_streamlit/core\constants.py`:

```py
# core/constants.py

# Directory Names (relative to the project root where app.py runs)
VIDEO_DOWNLOAD_DIR = "downloaded_video"
CLIPS_OUTPUT_DIR = "generated_clips"
TEMP_AUDIO_DIR = "temp_audio" # For storing temporary audio files for OpenAI API

# Default Model Names / Settings
DEFAULT_YT_RESOLUTION = "720p"
DEFAULT_LOCAL_WHISPER_MODEL = "base"
DEFAULT_TOPIC_ID_LLM = "gpt-4o-mini" # Cheaper for topic identification
DEFAULT_SEGMENT_EXTRACT_LLM = "gpt-4o-mini" # Can be changed to "gpt-4o" for higher quality segment extraction
MAX_TOPICS_DEFAULT = 3
MIN_CLIP_DURATION_DEFAULT = 50
MAX_CLIP_DURATION_DEFAULT = 59

# Other constants
TRANSCRIPT_TYPE_YT_API = "youtube_api"
TRANSCRIPT_TYPE_LOCAL_WHISPER = "local_whisper"
TRANSCRIPT_TYPE_OPENAI_API = "openai_api"

# Filename Suffixes
YT_API_TRANSCRIPT_SUFFIX = "_ytapi_transcript.json"
LOCAL_WHISPER_TRANSCRIPT_SUFFIX_FORMAT = "_local_whisper_{model_name}_transcript.json" # Note the format placeholder
OPENAI_API_TRANSCRIPT_SUFFIX = "_openai_api_transcript.json"
TEMP_AUDIO_FILE_SUFFIX = ".mp3"
```

`video_clipper_streamlit/core\models.py`:

```py
# core/models.py

from typing import List, Optional
from pydantic import BaseModel, Field

# --- Pydantic Models ---

class VideoAnalysis(BaseModel):
    summary: str = Field(..., description="A brief summary of the video content, in 2-3 sentences.")
    main_topics: List[str] = Field(
        ...,
        description="A list of distinct main topics or themes discussed in the video. Each topic should be a short phrase (2-5 words) suitable for guiding clip extraction.",
        min_length=1 # For Pydantic v2, actual enforcement might need validator or be on List itself
    )
    target_audience: Optional[str] = Field(None, description="The likely target audience for this video (e.g., 'Tech Enthusiasts', 'Business Leaders', 'Students').")

class Segment(BaseModel):
    start_time: int = Field(..., description="The start time of the segment in SECONDS from the beginning of the video.")
    end_time: int = Field(..., description="The end time of the segment in SECONDS from the beginning of the video.")
    title: str = Field(..., description="A concise, engaging title for this video segment (max 10 words).")
    description: str = Field(..., description="A brief 1-2 sentence description of what this segment is about.")

class SegmentResponse(BaseModel):
    segments: List[Segment] = Field(..., description="A list of identified video segments matching the criteria.")
```

`video_clipper_streamlit/orchestrators\common_steps.py`:

```py
# orchestrators/common_steps.py
import os
import logging
from typing import List, Dict, Any, Callable
from core import models as core_models, constants
from services import system_service, llm_service, ffmpeg_service

def common_llm_and_clipping_pipeline(
    video_file_path: str,
    video_base_title: str,
    transcript_data_list: List[Dict[str, Any]],
    openai_api_key: str,
    max_topics: int,
    min_clip_duration: int,
    max_clip_duration: int,
    status_callback: Callable[[str, str], None]
) -> Dict[str, Any]:
    orchestration_results = {
        "analysis": None, 
        # "all_segments_objects": [], # We will now store segments with their topics
        "segments_with_topics": [], # New: List of tuples (topic_str, Segment_Pydantic_Object)
        "generated_clips_info": [], 
        "success_llm_clipping": False 
    }

    # 1. Identify Topics
    status_callback("Identifying topics with LLM...", "info")
    analysis_result = llm_service.identify_video_topics(
        transcript_data_list, openai_api_key, max_topics, constants.DEFAULT_TOPIC_ID_LLM
    )
    if not analysis_result or not analysis_result.main_topics:
        status_callback("Failed to identify topics or no topics returned by LLM.", "error")
        return orchestration_results
    
    orchestration_results["analysis"] = analysis_result
    identified_topics_list = analysis_result.main_topics
    status_callback(f"LLM identified topics: {', '.join(identified_topics_list)}", "info")

    # 2. Extract Segments and associate with topic immediately
    all_segments_with_associated_topic: List[Dict[str, Any]] = [] # Will store dicts ready for FFMPEG & UI
    
    for topic_idx, topic_str in enumerate(identified_topics_list):
        status_callback(f"Extracting segments for topic {topic_idx+1}/{len(identified_topics_list)}: '{topic_str}'", "info")
        
        # llm_service.extract_segments_for_topic returns Optional[List[core_models.Segment]]
        segments_for_this_topic_objects = llm_service.extract_segments_for_topic(
            transcript_data_list, topic_str, openai_api_key, 
            min_clip_duration, max_clip_duration, constants.DEFAULT_SEGMENT_EXTRACT_LLM
        )
        
        if segments_for_this_topic_objects: 
            status_callback(f"Found {len(segments_for_this_topic_objects)} segments for topic '{topic_str}'.", "info")
            for seg_obj in segments_for_this_topic_objects:
                seg_dict = seg_obj.model_dump() # Convert Pydantic object to dict
                seg_dict['topic'] = topic_str    # Add the current topic
                seg_dict['calculated_duration'] = seg_dict['end_time'] - seg_dict['start_time']
                all_segments_with_associated_topic.append(seg_dict)
        else: 
            status_callback(f"No segments found or LLM error for topic '{topic_str}'.", "warning")
    
    # Store this list of dictionaries, which now includes the topic for each segment.
    # This replaces the old "all_segments_objects" which was just a flat list of Pydantic objects.
    orchestration_results["all_segments_with_topics"] = all_segments_with_associated_topic 
    status_callback(f"Total segments identified by LLM for potential clipping: {len(all_segments_with_associated_topic)}", "info")

    # 3. Generate Clips
    if not all_segments_with_associated_topic: # Check the new list
        status_callback("No valid segments identified by LLM to generate clips for.", "warning")
        orchestration_results["success_llm_clipping"] = True 
        return orchestration_results

    generated_clips_list = []
    status_callback(f"Generating {len(all_segments_with_associated_topic)} video clips...", "info")
    
    for seg_idx, segment_data in enumerate(all_segments_with_associated_topic): # Iterate the list of dicts
        # segment_data is now a dictionary already containing 'topic', 'title', 'start_time', etc.
        clip_title_sanitized = system_service.sanitize_filename(segment_data.get('title', f'segment_{seg_idx}'))
        duration = segment_data.get('calculated_duration', segment_data['end_time'] - segment_data['start_time']) # Use pre-calculated
        
        output_clip_filename = f"{video_base_title}_{clip_title_sanitized}_{duration}s_{seg_idx}.mp4"
        output_clip_path = os.path.join(constants.CLIPS_OUTPUT_DIR, output_clip_filename)

        status_callback(f"Generating clip {seg_idx+1}/{len(all_segments_with_associated_topic)} for topic '{segment_data['topic']}': '{segment_data.get('title', 'Untitled')}'", "info")
        
        success = ffmpeg_service.generate_clip(
            video_file_path, output_clip_path,
            segment_data['start_time'], segment_data['end_time']
        )
        if success:
            # Build the clip_info dictionary for app.py to display
            clip_info = {
                "title": segment_data.get('title', 'Untitled Clip'),
                "topic": segment_data['topic'], # Now correctly available
                "path": output_clip_path, 
                "filename": output_clip_filename,
                "description": segment_data.get('description', 'No description.'),
                "start_time": segment_data['start_time'],
                "end_time": segment_data['end_time'],
                "duration": duration
            }
            generated_clips_list.append(clip_info)
            
    orchestration_results["generated_clips_info"] = generated_clips_list
    status_callback(f"Clip generation finished. {len(generated_clips_list)} clips created.", "info")
    orchestration_results["success_llm_clipping"] = True
    return orchestration_results
```

`video_clipper_streamlit/orchestrators\local_mp4_pipeline.py`:

```py
# orchestrators/local_mp4_pipeline.py
import os
import logging
from typing import Dict, Any, Callable, List, Optional # Ensure Optional and List are imported

from core import constants
from services import system_service, video_processing_service, transcription_service, llm_service, ffmpeg_service
# Assuming common_steps.py is in the same 'orchestrators' package
from . import common_steps 

def run_local_mp4_pipeline(
    uploaded_file_obj: Any, # Streamlit UploadedFile object
    transcription_method_choice: str, # e.g., "Local Whisper Model", "OpenAI Whisper API"
    local_whisper_model_size: str, # e.g., "base", "medium"
    openai_api_key: str,
    max_topics: int,
    min_clip_duration: int,
    max_clip_duration: int,
    status_callback: Callable[[str, str], None] # Function to send status updates to UI
) -> Dict[str, Any]:
    """
    Orchestrates the entire processing pipeline for an uploaded MP4 file.
    Returns a dictionary containing all results and status.
    """
    master_results: Dict[str, Any] = {
        "video_info": None, 
        "transcript_info": None, 
        "analysis": None, # Will hold core_models.VideoAnalysis object
        "all_segments_objects": [], # Will hold List[core_models.Segment] from common_steps
        "generated_clips_info": [], # Will hold List[Dict] with clip details for UI
        "success": False,
        "error_message": None
    }
    
    if uploaded_file_obj is None:
        msg = "No file uploaded."
        status_callback(msg, "error")
        master_results["error_message"] = msg
        return master_results
        
    status_callback(f"Starting Local MP4 pipeline for: {uploaded_file_obj.name}", "info")

    # 1. Save Uploaded Video
    status_callback("Saving uploaded MP4...", "info")
    suggested_base_title = system_service.sanitize_filename(os.path.splitext(uploaded_file_obj.name)[0])
    
    saved_video_path = video_processing_service.save_uploaded_mp4(
        uploaded_file_obj, constants.VIDEO_DOWNLOAD_DIR, suggested_base_title
    )
    if not saved_video_path:
        msg = "Failed to save uploaded MP4."
        status_callback(msg, "error")
        master_results["error_message"] = msg
        return master_results
    
    # Use the sanitized name from the actual saved path for consistency
    video_base_title = system_service.sanitize_filename(os.path.splitext(os.path.basename(saved_video_path))[0])
    master_results["video_info"] = {"path": saved_video_path, "title": video_base_title}
    video_file_path = saved_video_path # This is the path to the processed video
    status_callback(f"Uploaded MP4 saved to: {video_file_path}", "info")

    # 2. Get Transcript
    transcript_data_list: Optional[List[Dict[str, Any]]] = None
    transcript_info_dict: Optional[Dict[str, Any]] = None

    if transcription_method_choice == "Local Whisper Model":
        device = system_service.check_gpu_availability()
        status_callback(f"Transcribing with Local Whisper model '{local_whisper_model_size}' on {device.upper()}...", "info")
        transcript_info_dict = transcription_service.transcribe_with_local_whisper(
            video_file_path, constants.VIDEO_DOWNLOAD_DIR, video_base_title, 
            local_whisper_model_size, device
        )
        # +++ DEBUG LOG for Local Whisper Transcript +++
        if transcript_info_dict and transcript_info_dict.get("type") == constants.TRANSCRIPT_TYPE_LOCAL_WHISPER:
            local_segments = transcript_info_dict.get("data", [])
            status_callback(f"--- Local Whisper Transcript (first 3 of {len(local_segments)}) ---", "debug")
            for i, seg_data in enumerate(local_segments[:3]):
                status_callback(
                    f"  LocalSeg {i+1}: S={seg_data.get('start')}, E={seg_data.get('end')}, D={seg_data.get('duration')}, Txt='{seg_data.get('text', '')[:50]}...'", 
                    "debug"
                )
            if len(local_segments) > 3:
                 status_callback(f"  ... and {len(local_segments)-3} more local segments.", "debug")
        # +++ END DEBUG LOG +++
            
    elif transcription_method_choice == "OpenAI Whisper API":
        status_callback("Transcribing with OpenAI Whisper API...", "info")
        transcript_info_dict = transcription_service.transcribe_with_openai_api(
            video_file_path, constants.VIDEO_DOWNLOAD_DIR, video_base_title, openai_api_key
        )
        # +++ DEBUG LOG for OpenAI API Transcript +++
        if transcript_info_dict and transcript_info_dict.get("type") == constants.TRANSCRIPT_TYPE_OPENAI_API:
            api_segments = transcript_info_dict.get("data", [])
            status_callback(f"--- OpenAI API Transcript (first 3 of {len(api_segments)}) ---", "debug")
            for i, seg_data in enumerate(api_segments[:3]):
                status_callback(
                    f"  APISeg {i+1}: S={seg_data.get('start')}, E={seg_data.get('end')}, D={seg_data.get('duration')}, Txt='{seg_data.get('text', '')[:50]}...'", 
                    "debug"
                )
            if len(api_segments) > 3:
                 status_callback(f"  ... and {len(api_segments)-3} more API segments.", "debug")
        # +++ END DEBUG CODE +++
    else:
        # This case should ideally be prevented by UI constraints if "YouTube API" is not an option for uploads.
        msg = f"Invalid transcription method '{transcription_method_choice}' selected for an uploaded MP4. Please choose Local Whisper or OpenAI API."
        status_callback(msg, "error")
        master_results["error_message"] = msg
        return master_results

    if not transcript_info_dict or not transcript_info_dict.get("data"):
        msg = f"Failed to obtain transcript data using method: {transcription_method_choice}."
        status_callback(msg, "error")
        master_results["error_message"] = msg
        return master_results

    transcript_data_list = transcript_info_dict["data"]
    master_results["transcript_info"] = transcript_info_dict # Store the full dict (path, type, data)
    status_callback(f"Transcription successful. Type: {transcript_info_dict.get('type', 'N/A')}, Segments: {len(transcript_data_list)}", "info")
    
    # 3. Common LLM and Clipping Steps
    llm_clipping_results: Dict[str, Any] = { # Initialize to prevent TypeError
        "analysis": None, "all_segments_objects": [], 
        "generated_clips_info": [], "success_llm_clipping": False
    }

    if transcript_data_list: # Ensure transcript_data_list is not None or empty before proceeding
        llm_clipping_results = common_steps.common_llm_and_clipping_pipeline(
            video_file_path=video_file_path, 
            video_base_title=video_base_title, 
            transcript_data_list=transcript_data_list,
            openai_api_key=openai_api_key, 
            max_topics=max_topics, 
            min_clip_duration=min_clip_duration, 
            max_clip_duration=max_clip_duration,
            status_callback=status_callback
        )
    else: 
        status_callback("Transcript data is empty, skipping LLM and clipping steps.", "warning")
        # llm_clipping_results keeps its initialized state (success_llm_clipping=False)

    # Update master_results with whatever came from llm_clipping_results
    # (which is guaranteed to be a dict now)
    master_results.update(llm_clipping_results) 

    # Determine overall success based on the success of the LLM/clipping stage
    if llm_clipping_results.get("success_llm_clipping", False):
        master_results["success"] = True
    else:
        # If llm_clipping_results failed, an error message should have been logged by common_steps.
        # We can append a general failure message here.
        current_error = master_results.get("error_message")
        additional_error = "LLM processing or clipping stage failed."
        if current_error: # If a previous error (e.g., transcription) occurred
            master_results["error_message"] = f"{current_error}. {additional_error}"
        else: # If this is the first error being recorded at this level
            master_results["error_message"] = additional_error
        master_results["success"] = False

    return master_results
```

`video_clipper_streamlit/orchestrators\youtube_pipeline.py`:

```py
# orchestrators/youtube_pipeline.py
import logging
from typing import Dict, Any, Callable, List, Optional # Ensure all needed types are imported

from core import constants
from services import system_service, video_processing_service, transcription_service 
# llm_service and ffmpeg_service will be called by common_steps

# Import the common pipeline steps
from . import common_steps # Relative import for common_steps.py

def run_youtube_pipeline(
    youtube_url: str,
    desired_yt_resolution: str,
    transcription_method_choice: str,
    local_whisper_model_size: str, # Only used if transcription_method_choice is local
    openai_api_key: str,
    max_topics: int,
    min_clip_duration: int,
    max_clip_duration: int,
    status_callback: Callable[[str, str], None]
) -> Dict[str, Any]:
    """
    Orchestrates the entire processing pipeline for a YouTube URL.
    Returns a dictionary containing all results and status.
    """
    
    master_results: Dict[str, Any] = {
        "video_info": None, 
        "transcript_info": None, 
        "analysis": None, # Will hold core_models.VideoAnalysis object
        # "all_segments_objects": [], # This key is from common_steps, might be renamed there
        "all_segments_with_topics": [], # Expecting this key from common_steps now
        "generated_clips_info": [], 
        "success": False, # Overall success of the entire pipeline
        "error_message": None
    }
    status_callback(f"Starting YouTube pipeline for: {youtube_url}", "info")

    # 1. Download Video
    status_callback("Downloading YouTube video...", "info")
    video_info = video_processing_service.download_youtube_video(
        youtube_url, constants.VIDEO_DOWNLOAD_DIR, desired_yt_resolution
    )
    if not video_info or "path" not in video_info:
        msg = "Failed to download video or video info is incomplete."
        status_callback(msg, "error")
        master_results["error_message"] = msg
        return master_results # Exit early
        
    master_results["video_info"] = video_info
    video_file_path = video_info["path"]
    video_base_title = video_info["title"]
    video_id = video_info["id"] # Used for YouTube API transcript option
    status_callback(f"Video downloaded successfully: {video_file_path}", "info")

    # 2. Get Transcript
    transcript_data_list: Optional[List[Dict[str, Any]]] = None
    transcript_info_dict: Optional[Dict[str, Any]] = None

    if transcription_method_choice == "YouTube API (for YouTube URLs, if available)":
        status_callback(f"Fetching transcript via YouTube API for video ID: {video_id}...", "info")
        transcript_info_dict = transcription_service.fetch_youtube_api_transcript(
            video_id, constants.VIDEO_DOWNLOAD_DIR, video_base_title
        )
        # Debug log for YouTube API transcript
        if transcript_info_dict and transcript_info_dict.get("type") == constants.TRANSCRIPT_TYPE_YT_API:
            yt_api_segments = transcript_info_dict.get("data", [])
            status_callback(f"--- YouTube API Transcript (first 3 of {len(yt_api_segments)}) ---", "debug")
            for i, seg_data in enumerate(yt_api_segments[:3]):
                status_callback(
                    f"  YTAPISeg {i+1}: S={seg_data.get('start')}, E={seg_data.get('end')}, D={seg_data.get('duration')}, Txt='{seg_data.get('text', '')[:50]}...'", "debug"
                )
            if len(yt_api_segments) > 3:
                 status_callback(f"  ... and {len(yt_api_segments)-3} more YouTube API segments.", "debug")
    
    # Fallback or direct choice for Local Whisper
    if transcript_info_dict is None and transcription_method_choice == "Local Whisper Model":
        device = system_service.check_gpu_availability()
        status_callback(f"Transcribing with Local Whisper model '{local_whisper_model_size}' on {device.upper()}...", "info")
        transcript_info_dict = transcription_service.transcribe_with_local_whisper(
            video_file_path, constants.VIDEO_DOWNLOAD_DIR, video_base_title, 
            local_whisper_model_size, device
        )
        if transcript_info_dict and transcript_info_dict.get("type") == constants.TRANSCRIPT_TYPE_LOCAL_WHISPER:
            local_segments = transcript_info_dict.get("data", [])
            status_callback(f"--- Local Whisper Transcript (first 3 of {len(local_segments)}) ---", "debug")
            for i, seg_data in enumerate(local_segments[:3]):
                status_callback(
                    f"  LocalSeg {i+1}: S={seg_data.get('start')}, E={seg_data.get('end')}, D={seg_data.get('duration')}, Txt='{seg_data.get('text', '')[:50]}...'", "debug"
                )
            if len(local_segments) > 3:
                 status_callback(f"  ... and {len(local_segments)-3} more local segments.", "debug")
            
    # Fallback or direct choice for OpenAI API
    if transcript_info_dict is None and transcription_method_choice == "OpenAI Whisper API":
        status_callback("Transcribing with OpenAI Whisper API...", "info")
        transcript_info_dict = transcription_service.transcribe_with_openai_api(
            video_file_path, constants.VIDEO_DOWNLOAD_DIR, video_base_title, openai_api_key
        )
        if transcript_info_dict and transcript_info_dict.get("type") == constants.TRANSCRIPT_TYPE_OPENAI_API:
            api_segments = transcript_info_dict.get("data", [])
            status_callback(f"--- OpenAI API Transcript (first 3 of {len(api_segments)}) ---", "debug")
            for i, seg_data in enumerate(api_segments[:3]):
                status_callback(
                    f"  APISeg {i+1}: S={seg_data.get('start')}, E={seg_data.get('end')}, D={seg_data.get('duration')}, Txt='{seg_data.get('text', '')[:50]}...'", "debug"
                )
            if len(api_segments) > 3:
                 status_callback(f"  ... and {len(api_segments)-3} more API segments.", "debug")

    if not transcript_info_dict or not transcript_info_dict.get("data"):
        msg = f"Failed to obtain transcript data using method: {transcription_method_choice}."
        status_callback(msg, "error")
        master_results["error_message"] = msg
        return master_results 
    
    transcript_data_list = transcript_info_dict["data"]
    master_results["transcript_info"] = transcript_info_dict
    status_callback(f"Transcription successful. Type: {transcript_info_dict.get('type', 'N/A')}, Segments: {len(transcript_data_list)}", "info")
    
    # 3. Common LLM and Clipping Steps
    # Initialize llm_clipping_results to ensure it's always a dict
    llm_clipping_results: Dict[str, Any] = {
        "analysis": None, "all_segments_with_topics": [], # Expecting this key now
        "generated_clips_info": [], "success_llm_clipping": False
    }

    if transcript_data_list: 
        llm_clipping_results = common_steps.common_llm_and_clipping_pipeline(
            video_file_path=video_file_path, 
            video_base_title=video_base_title, 
            transcript_data_list=transcript_data_list,
            openai_api_key=openai_api_key, 
            max_topics=max_topics, 
            min_clip_duration=min_clip_duration, 
            max_clip_duration=max_clip_duration,
            status_callback=status_callback
        )
    else: 
        status_callback("Transcript data is empty, skipping LLM and clipping steps.", "warning")
        # llm_clipping_results retains its initialized state

    master_results.update(llm_clipping_results) 

    if llm_clipping_results.get("success_llm_clipping", False):
        master_results["success"] = True
    else:
        current_error = master_results.get("error_message")
        additional_error = "LLM processing or clipping stage failed." # This message comes if common_steps sets success_llm_clipping to False
        
        if not llm_clipping_results.get("success_llm_clipping", False) and not current_error:
             master_results["error_message"] = additional_error
        elif current_error : 
             master_results["error_message"] = f"{current_error}. {additional_error}"
        master_results["success"] = False

    return master_results
```

`video_clipper_streamlit/services\ffmpeg_service.py`:

```py
# services/ffmpeg_service.py

import os
import subprocess
import logging

# Import from other services or core modules
from services.system_service import ensure_ffmpeg_is_accessible, get_ffmpeg_executable_name
from core import constants # For CLIPS_OUTPUT_DIR if needed, though app.py constructs full path

def generate_clip(
    input_video_path: str, 
    output_clip_path: str, 
    start_time: int, 
    end_time: int,
    video_codec: str = "libx264",
    video_crf: str = "23", 
    video_preset: str = "medium", 
    audio_codec: str = "aac",
    audio_bitrate: str = "128k"
) -> bool:
    """
    Generates a video clip from the input video using FFMPEG.
    """
    if not ensure_ffmpeg_is_accessible():
        logging.error(f"FFMPEG not accessible for generating clip: {os.path.basename(output_clip_path)}")
        return False
        
    ffmpeg_executable = get_ffmpeg_executable_name()

    if not os.path.exists(input_video_path):
        logging.error(f"Input video for FFMPEG not found: {input_video_path}")
        return False
        
    clip_output_dir = os.path.dirname(output_clip_path)
    if not os.path.exists(clip_output_dir):
        try:
            os.makedirs(clip_output_dir)
            logging.info(f"Created output directory for clips: {clip_output_dir}")
        except OSError as e:
            logging.error(f"Failed to create clip output directory {clip_output_dir}: {e}")
            return False

    logging.info(
        f"Generating FFMPEG clip: '{os.path.basename(output_clip_path)}' "
        f"from {start_time}s to {end_time}s using '{ffmpeg_executable}'"
    )
    try:
        command = [
            ffmpeg_executable, "-y", "-hide_banner",
            "-i", input_video_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c:v", video_codec, 
            "-crf", video_crf, 
            "-preset", video_preset,
            "-c:a", audio_codec, 
            "-b:a", audio_bitrate,
            output_clip_path
        ]
        
        logging.debug(f"Executing FFMPEG command: {' '.join(command)}")
        # shell=False is safer if ffmpeg_executable is a full path or resolved by PATH.
        # The ensure_ffmpeg_is_accessible tries to make _ffmpeg_executable_name a full path if found locally.
        result = subprocess.run(command, capture_output=True, text=True, shell=False, check=False) 
        
        if result.returncode == 0:
            logging.info(f"Successfully generated clip: {output_clip_path}")
            return True
        else:
            logging.error(f"FFMPEG failed for clip '{output_clip_path}'. Return code: {result.returncode}")
            logging.error(f"FFMPEG STDOUT: {result.stdout.strip()}")
            logging.error(f"FFMPEG STDERR: {result.stderr.strip()}")
            return False
    except Exception as e: # Catch other potential errors like FileNotFoundError if ffmpeg_executable was bad
        logging.error(f"An unexpected error occurred during FFMPEG clip generation for '{output_clip_path}': {e}", exc_info=True)
        return False
```

`video_clipper_streamlit/services\llm_service.py`:

```py
# services/llm_service.py

import json
import logging
from typing import List, Optional, Dict, Any

from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel, Field 

# Import Pydantic models from the core module
# Use the specific classes directly, or alias 'core.models' if you prefer 'core_models.Class'
from core.models import VideoAnalysis, Segment, SegmentResponse 
from core import constants 

# Remove local SegmentDebug and SegmentResponseDebug as we are using the ones from core.models
# class SegmentDebug(BaseModel): ...
# class SegmentResponseDebug(BaseModel): ...


def identify_video_topics(
    transcript_data: List[Dict[str, Any]], 
    openai_api_key: str, 
    max_topics: int = constants.MAX_TOPICS_DEFAULT, 
    model_name: str = constants.DEFAULT_TOPIC_ID_LLM
) -> Optional[VideoAnalysis]: # Return type is VideoAnalysis from core.models
    """
    Identifies topics from transcript data using an LLM.
    """
    if not transcript_data:
        logging.warning("LLM Service: Cannot identify topics: transcript data is empty or None.")
        return None
    
    full_transcript_text = " ".join([s.get('text', '') for s in transcript_data if isinstance(s, dict)])
    if not full_transcript_text.strip():
        logging.warning("LLM Service: Cannot identify topics: concatenated transcript text is empty after processing.")
        return None

    logging.info(f"LLM Service: Identifying up to {max_topics} topics using LLM ({model_name})...")
    try:
        llm = ChatOpenAI(api_key=openai_api_key, model=model_name, temperature=0.3)
        
        class DynamicVideoAnalysis(BaseModel): 
            summary: str = Field(..., description="A brief summary of the video content, in 2-3 sentences.")
            main_topics: List[str] = Field(
                ...,
                description=f"A list of up to {max_topics} distinct main topics or themes. Each topic: short phrase (2-5 words).",
                min_length=0, 
                max_length=max_topics 
            )
            target_audience: Optional[str] = Field(None, description="The likely target audience (e.g., 'Tech Enthusiasts').")

        truncated_transcript_text = full_transcript_text[:15000]
        logging.debug(f"LLM Service: Transcript text for topic ID (first 100 chars): {truncated_transcript_text[:100]}")

        prompt_text = (
            f"Analyze the provided video transcript. Your tasks are:\n"
            f"1. Provide a brief overall summary of the video's content (2-3 sentences).\n"
            f"2. Identify up to {max_topics} distinct main topics or key themes evident in the transcript. Each topic should be a concise phrase (2-5 words) suitable for later use in creating short video clips.\n"
            f"3. Suggest a likely target audience for this video (e.g., 'Tech Enthusiasts', 'Business Leaders', 'Students').\n\n"
            f"Respond ONLY in the requested JSON structure.\n\n"
            f"Video Transcript (excerpt):\n\"\"\"\n{truncated_transcript_text}\n\"\"\""
        )
        messages = [
            {"role": "system", "content": "You are an expert video content analyst. Your response must strictly adhere to the requested JSON output structure."},
            {"role": "user", "content": prompt_text}
        ]
        
        structured_llm = llm.with_structured_output(DynamicVideoAnalysis)
        logging.info(f"LLM Service: Invoking LLM for topic identification with model {model_name}...")
        analysis_result_dynamic = structured_llm.invoke(messages)

        if analysis_result_dynamic:
            final_analysis = VideoAnalysis( # Use VideoAnalysis from core.models
                summary=analysis_result_dynamic.summary,
                main_topics=analysis_result_dynamic.main_topics,
                target_audience=analysis_result_dynamic.target_audience
            )
            logging.info(f"LLM Service: Topic ID successful. Summary: '{final_analysis.summary[:100]}...', Topics: {final_analysis.main_topics}")
            return final_analysis
        else:
            logging.warning("LLM Service: Topic identification returned no result from LLM.")
            return None

    except Exception as e:
        logging.error(f"LLM Service: Error identifying topics with LLM: {e}", exc_info=True)
        return None

def extract_segments_for_topic(
    full_transcript_data: List[Dict[str, Any]], 
    topic: str, 
    openai_api_key: str, 
    min_duration: int, 
    max_duration: int,
    model_name: str = constants.DEFAULT_SEGMENT_EXTRACT_LLM
) -> Optional[List[Segment]]: # CORRECTED: Return type is List of Segment from core.models
    """
    Extracts video segments for a given topic from the full transcript using an LLM.
    """
    if not full_transcript_data:
        logging.warning(f"LLM Service: Cannot extract segments for topic '{topic}': transcript data is empty or None.")
        return None

    logging.info(f"LLM Service: --- Input Transcript for Topic '{topic}' (extract_segments_for_topic) ---")
    if isinstance(full_transcript_data, list):
        for i, seg_data in enumerate(full_transcript_data[:5]):
            logging.info(
                f"  LLM Service: API/Local Segment {i+1}: Start={seg_data.get('start', 'N/A')}, "
                f"End={seg_data.get('end', 'N/A')}, "
                f"Duration={seg_data.get('duration', 'N/A')}, "
                f"Text='{seg_data.get('text', '')[:70]}...'"
            )
        if len(full_transcript_data) > 5:
            logging.info(f"  LLM Service: ... and {len(full_transcript_data) - 5} more API/Local segments.")
    else:
        logging.warning("LLM Service: full_transcript_data is not a list as expected in extract_segments_for_topic.")
    logging.info(f"LLM Service: --- End of Input Transcript for Topic '{topic}' ---")

    logging.info(f"LLM Service: Extracting segments for topic: '{topic}' using LLM ({model_name}). Target duration: {min_duration}-{max_duration}s.")
    try:
        llm = ChatOpenAI(api_key=openai_api_key, model=model_name, temperature=0.5)
        
        transcript_json_str = json.dumps(full_transcript_data, indent=2)
        MAX_TRANSCRIPT_CHARS_FOR_PROMPT = 300000 
        if len(transcript_json_str) > MAX_TRANSCRIPT_CHARS_FOR_PROMPT:
            logging.warning(
                f"LLM Service: Transcript for topic '{topic}' is very long ({len(transcript_json_str)} chars). "
                f"Truncating to {MAX_TRANSCRIPT_CHARS_FOR_PROMPT} chars for LLM prompt."
            )
            cutoff_point = transcript_json_str.rfind("},", 0, MAX_TRANSCRIPT_CHARS_FOR_PROMPT)
            if cutoff_point != -1:
                transcript_json_str = transcript_json_str[:cutoff_point+2] + "\n  ...\n]" 
            else: 
                transcript_json_str = transcript_json_str[:MAX_TRANSCRIPT_CHARS_FOR_PROMPT] + "\n... (transcript truncated due to length) ...\n]}" 
        
        prompt_text = (
            f"From the provided full video transcript (a JSON string representing a list of dictionaries, each with 'text', 'start' in seconds, 'end' in seconds, and 'duration' in seconds), "
            f"identify all distinct segments that are specifically and clearly about the topic: '{topic}'.\n\n"
            f"For each segment you identify:\n"
            f"1. CRITICAL: The segment's total duration (calculated as 'end_time' - 'start_time') MUST be strictly between {min_duration} and {max_duration} seconds. This is a primary requirement.\n"
            f"2. To achieve the target duration, you may need to COMBINE consecutive relevant sentences or short transcript entries. The 'start_time' should be the start of the first combined entry, and 'end_time' should be the end of the last combined entry for the generated clip.\n"
            f"3. Provide 'start_time' and 'end_time' as integers representing whole SECONDS from the beginning of the video, derived from the input transcript's timestamps.\n"
            f"4. Create a concise, engaging 'title' (max 10 words) for the combined segment, relevant to its content and the given topic.\n"
            f"5. Write a brief 'description' (1-2 sentences) summarizing what this combined segment covers.\n\n"
            f"Prioritize segments that are coherent, make sense as standalone clips, and STRICTLY meet the duration criteria. If you combine transcript entries, ensure the combined text flows well.\n\n"
            f"Respond ONLY with a JSON object containing a single key 'segments'. The value of 'segments' must be a list of segment objects. "
            f"Each segment object must have the keys: 'start_time', 'end_time', 'title', and 'description'.\n"
            f"If no segments can be formed that meet ALL criteria (especially topic relevance AND the {min_duration}-{max_duration}s duration), return an empty list for 'segments'.\n\n"
            f"Topic to focus on: {topic}\n\n"
            f"Full Video Transcript (JSON string):\n\"\"\"\n{transcript_json_str}\n\"\"\""
        )
        messages = [
            {"role": "system", "content": "You are a precise video segment extraction assistant. Your response must strictly be a JSON object with a 'segments' key, containing a list of segment objects as specified."},
            {"role": "user", "content": prompt_text}
        ]
        
        logging.debug(f"LLM Service: --- PROMPT for Segment Extraction (Topic: '{topic}') ---")
        logging.debug(f"{prompt_text[:1000]} ... (prompt potentially truncated for log)")
        logging.debug(f"--- END OF PROMPT ---")

        structured_llm = llm.with_structured_output(SegmentResponse) # Using SegmentResponse from core.models
        logging.info(f"LLM Service: Invoking LLM for segment extraction on topic '{topic}' with model {model_name}...")
        response_model = structured_llm.invoke(messages)
        
        if response_model and response_model.segments is not None:
            logging.info(f"LLM Service: LLM returned {len(response_model.segments)} segments for topic '{topic}'.")
            valid_segments = []
            for seg in response_model.segments: # seg is now core.models.Segment
                actual_duration = seg.end_time - seg.start_time
                if not (min_duration <= actual_duration <= max_duration):
                    logging.warning(
                        f"LLM Service: Segment '{seg.title}' for topic '{topic}' "
                        f"has duration {actual_duration}s, outside target range {min_duration}-{max_duration}s. Skipping."
                    )
                    continue
                valid_segments.append(seg) 
            
            if valid_segments:
                 logging.info(f"LLM Service: After duration validation, {len(valid_segments)} segments remain for topic '{topic}'.")
            else:
                logging.info(f"LLM Service: No segments remained after duration validation for topic '{topic}'.")
            return valid_segments # List of core.models.Segment objects
        else:
            logging.warning(f"LLM Service: LLM did not return valid segments structure for topic '{topic}'. Response: {response_model}")
            return [] 

    except Exception as e:
        logging.error(f"LLM Service: Error extracting segments for topic '{topic}' with LLM: {e}", exc_info=True)
        return None
```

`video_clipper_streamlit/services\system_service.py`:

```py
# services/system_service.py

import os
import sys
import subprocess
import logging
import torch
import gc
from typing import Optional

# Module-level flags to track FFMPEG status
_ffmpeg_path_confirmed = False
_ffmpeg_dir_added_to_path = False
_ffmpeg_executable_name = "ffmpeg" # Default, might become full path

def sanitize_filename(name: str) -> str:
    """Cleans a string to be a safe filename component."""
    if not isinstance(name, str):
        name = str(name) 
    sanitized = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name).rstrip()
    sanitized = sanitized.replace(" ", "_")
    # Ensure it doesn't just become underscores or empty
    if not sanitized.strip('_'):
        return "untitled_segment"
    return sanitized if sanitized else "untitled_segment"

def get_openai_api_key() -> Optional[str]:
    """Loads OpenAI API key from environment."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        logging.warning("OpenAI API key not found in environment variables.")
    return key

def check_gpu_availability() -> str:
    """Checks for CUDA availability and returns device string ('cuda' or 'cpu')."""
    if torch.cuda.is_available():
        try:
            # Basic check first
            torch.cuda.device_count() 
            logging.info(f"CUDA is available. PyTorch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
            logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
            return "cuda"
        except RuntimeError as e: # Handles cases like "No CUDA GPUs are available" despite torch.cuda.is_available() being true initially
            logging.warning(f"PyTorch reported CUDA as available, but encountered RuntimeError checking details: {e}. Defaulting to CPU.")
            return "cpu"
        except Exception as e: # Catch any other unexpected errors
            logging.warning(f"CUDA reported as available, but an unexpected error occurred checking details: {e}. Defaulting to CPU.")
            return "cpu"
    else:
        logging.info("CUDA not available. Using CPU.")
        return "cpu"

def get_nvidia_smi_output() -> str:
    """Runs nvidia-smi and returns its output or an error message."""
    try:
        is_windows = os.name == 'nt'
        cmd_to_try = ["nvidia-smi"]
        
        # On Windows, try a common explicit path if the simple command fails
        if is_windows:
            program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
            explicit_path = os.path.join(program_files, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")
            if os.path.exists(explicit_path):
                cmd_to_try = [explicit_path] # Prioritize explicit path if found
            else: # If explicit path not found, rely on PATH and shell=True might be needed
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, shell=True, check=False) # Try with shell=True for PATH resolution
                if result.returncode == 0:
                    return result.stdout
                # If still fails, the generic error below will be caught
        
        # For non-Windows or if explicit Windows path failed/not found
        result = subprocess.run(cmd_to_try, capture_output=True, text=True, shell=False, check=False) # shell=False preferred for safety
        if result.returncode == 0:
            return result.stdout
        else:
            logging.warning(f"nvidia-smi (command: {' '.join(cmd_to_try)}) failed or not found. Return code: {result.returncode}. Error: {result.stderr.strip()}")
            return "nvidia-smi command not found or failed. Is NVIDIA driver installed and nvidia-smi in system PATH?"

    except FileNotFoundError:
        logging.warning("nvidia-smi command not found. Is NVIDIA driver installed and in system PATH?")
        return "nvidia-smi command not found. Ensure NVIDIA drivers are installed and nvidia-smi is in your system PATH."
    except Exception as e:
        logging.error(f"An unexpected error occurred while running nvidia-smi: {e}", exc_info=True)
        return f"An unexpected error occurred while running nvidia-smi: {str(e)}"


def get_ffmpeg_executable_name() -> str:
    """Returns the command name or full path for ffmpeg after ensuring it's accessible."""
    global _ffmpeg_executable_name
    ensure_ffmpeg_is_accessible() # This will set _ffmpeg_executable_name if found locally
    return _ffmpeg_executable_name


def ensure_ffmpeg_is_accessible() -> bool:
    """
    Ensures ffmpeg is accessible.
    Checks for ffmpeg.exe and ffprobe.exe in the app's root directory first.
    If found, prepends that directory to os.environ['PATH'] for the current session.
    Then verifies by trying to run 'ffmpeg -version'.
    Updates a module-level flag `_ffmpeg_path_confirmed`.
    """
    global _ffmpeg_path_confirmed, _ffmpeg_dir_added_to_path, _ffmpeg_executable_name
    if _ffmpeg_path_confirmed:
        return True

    # Determine app root: Assumes services/ is one level below app.py's directory
    current_service_script_dir = os.path.dirname(os.path.abspath(__file__))
    app_root_dir = os.path.dirname(current_service_script_dir)

    local_ffmpeg_exe = "ffmpeg.exe" if os.name == 'nt' else "ffmpeg"
    local_ffprobe_exe = "ffprobe.exe" if os.name == 'nt' else "ffprobe"
    
    path_to_local_ffmpeg = os.path.join(app_root_dir, local_ffmpeg_exe)
    path_to_local_ffprobe = os.path.join(app_root_dir, local_ffprobe_exe)

    ffmpeg_dir_to_add = None

    if os.path.exists(path_to_local_ffmpeg) and os.path.exists(path_to_local_ffprobe):
        logging.info(f"Found local ffmpeg/ffprobe in: {app_root_dir}")
        ffmpeg_dir_to_add = app_root_dir
        _ffmpeg_executable_name = path_to_local_ffmpeg # Use full path if found locally
    else:
        logging.debug(f"Local ffmpeg/ffprobe not found in {app_root_dir}. Will rely on system PATH.")
        _ffmpeg_executable_name = "ffmpeg" # Rely on system PATH

    # Add to PATH if a local directory was found and not already added
    if ffmpeg_dir_to_add and not _ffmpeg_dir_added_to_path:
        logging.info(f"Prepending FFMPEG directory to PATH for this session: {ffmpeg_dir_to_add}")
        os.environ["PATH"] = ffmpeg_dir_to_add + os.pathsep + os.environ.get("PATH", "")
        _ffmpeg_dir_added_to_path = True
    
    # Verify FFMPEG accessibility
    try:
        cmd_to_verify = [_ffmpeg_executable_name, "-version"]
        logging.debug(f"Verifying FFMPEG with command: {' '.join(cmd_to_verify)}")
        # Use shell=True for Windows if just "ffmpeg" is used and it's in PATH,
        # but if _ffmpeg_executable_name is a full path, shell=False is better.
        use_shell = (os.name == 'nt' and _ffmpeg_executable_name == "ffmpeg")

        result = subprocess.run(cmd_to_verify, capture_output=True, text=True, shell=use_shell, check=True)
        logging.info(f"FFMPEG is accessible. Version info (first 100 chars): {result.stdout[:100].strip()}...")
        _ffmpeg_path_confirmed = True
        return True
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logging.error(
            f"FFMPEG command ('{_ffmpeg_executable_name}') failed verification. "
            f"Ensure ffmpeg.exe and ffprobe.exe are in '{app_root_dir}' (if bundled) OR in your system PATH. Error: {e}"
        )
        if isinstance(e, subprocess.CalledProcessError):
            logging.error(f"FFMPEG STDERR: {e.stderr.strip()}")
        _ffmpeg_path_confirmed = False
        return False
    except Exception as e_unexp:
        logging.error(f"Unexpected error verifying FFMPEG: {e_unexp}", exc_info=True)
        _ffmpeg_path_confirmed = False
        return False


def clear_gpu_mem(model_ref=None):
    """Clears GPU memory by deleting model reference and emptying cache."""
    if model_ref is not None:
        try:
            del model_ref
            logging.info("Model reference deleted for GPU memory clearing.")
        except NameError: # model_ref might not exist if an error occurred before its assignment
            logging.debug("Model reference for clearing GPU mem was not defined or already deleted.")

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            logging.info("PyTorch CUDA cache cleared.")
        except Exception as e:
            logging.error(f"Error clearing CUDA cache: {e}", exc_info=True)
    try:
        gc.collect()
        logging.info("Python garbage collection ran.")
    except Exception as e:
        logging.error(f"Error running garbage collection: {e}", exc_info=True)
```

`video_clipper_streamlit/services\transcription_service.py`:

```py
# services/transcription_service.py

import os
import json
import logging
import subprocess
from typing import Optional, Dict, Any, List

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI as OpenAIClient

# Import from other services or core modules
from services.system_service import clear_gpu_mem, sanitize_filename, ensure_ffmpeg_is_accessible, get_ffmpeg_executable_name
from core import constants

# Conditional import for stable_whisper
try:
    import stable_whisper
except ImportError:
    stable_whisper = None
    logging.info("stable_whisper library not found. Local Whisper transcription will not be available if selected.")


def _extract_audio_for_api(video_file_path: str, base_title: str) -> Optional[str]:
    """
    Private helper to extract audio from video for API transcription.
    Saves audio to the TEMP_AUDIO_DIR defined in constants.
    """
    if not ensure_ffmpeg_is_accessible():
        logging.error("FFMPEG not accessible for audio extraction.")
        return None
    
    ffmpeg_executable = get_ffmpeg_executable_name()

    if not os.path.exists(constants.TEMP_AUDIO_DIR):
        try:
            os.makedirs(constants.TEMP_AUDIO_DIR)
            logging.info(f"Created temporary audio directory: {constants.TEMP_AUDIO_DIR}")
        except OSError as e_mkdir:
            logging.error(f"Failed to create temporary audio directory {constants.TEMP_AUDIO_DIR}: {e_mkdir}")
            return None

    temp_audio_filename = f"{sanitize_filename(base_title)}{constants.TEMP_AUDIO_FILE_SUFFIX}"
    temp_audio_file_path = os.path.join(constants.TEMP_AUDIO_DIR, temp_audio_filename)
    
    try:
        logging.info(f"Extracting audio from '{video_file_path}' to '{temp_audio_file_path}' using '{ffmpeg_executable}'...")
        ffmpeg_command = [
            ffmpeg_executable, "-y",
            "-i", video_file_path,
            "-vn", 
            "-acodec", "libmp3lame", 
            "-q:a", "2", # MP3 quality for LAME (0-9, lower is better)
            temp_audio_file_path
        ]
        # shell=False is generally safer, especially if ffmpeg_executable is a full path.
        # ensure_ffmpeg_is_accessible tries to make it a full path if found locally.
        result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, shell=False)
        
        if os.path.exists(temp_audio_file_path) and os.path.getsize(temp_audio_file_path) > 0:
            logging.info(f"Audio extracted successfully to {temp_audio_file_path}")
            return temp_audio_file_path
        else:
            logging.error(f"Audio extraction command ran, but output file '{temp_audio_file_path}' is missing or empty.")
            if result: logging.debug(f"FFMPEG STDOUT for audio extraction: {result.stdout}")
            return None

    except subprocess.CalledProcessError as e:
        logging.error(f"FFMPEG audio extraction failed. Return code: {e.returncode}")
        logging.error(f"FFMPEG STDERR for audio extraction: {e.stderr}")
        logging.error(f"FFMPEG STDOUT for audio extraction: {e.stdout}")
        return None
    except Exception as e: # Catch other potential errors like FileNotFoundError if ffmpeg_executable was bad despite checks
        logging.error(f"Unexpected error during audio extraction: {e}", exc_info=True)
        return None


def fetch_youtube_api_transcript(
    video_id: str, output_dir: str, base_title: str
) -> Optional[Dict[str, Any]]:
    """Fetches transcript using YouTubeTranscriptApi and saves it."""
    transcript_filename_json = f"{sanitize_filename(base_title)}{constants.YT_API_TRANSCRIPT_SUFFIX}"
    transcript_path = os.path.join(output_dir, transcript_filename_json)
    try:
        logging.info(f"Fetching YouTube API transcript for video ID: {video_id}...")
        if os.path.exists(transcript_path):
            logging.info(f"YouTube API transcript already exists: {transcript_path}. Loading from file.")
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
            for entry in transcript_data: # Ensure 'end' key consistency
                if 'start' in entry and 'duration' in entry and 'end' not in entry:
                    entry['end'] = round(entry['start'] + entry['duration'], 2)
            return {"data": transcript_data, "path": transcript_path, "type": constants.TRANSCRIPT_TYPE_YT_API}

        transcript_list_api = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_data = []
        for entry in transcript_list_api:
            start = round(entry.get("start", 0.0), 2)
            duration = round(entry.get("duration", 0.0), 2)
            transcript_data.append({
                "text": entry.get("text", "").strip(),
                "start": start,
                "duration": duration,
                "end": round(start + duration, 2)
            })
        
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved YouTube API transcript to: {transcript_path}")
        
        return {"data": transcript_data, "path": transcript_path, "type": constants.TRANSCRIPT_TYPE_YT_API}
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        logging.warning(f"Could not fetch YouTube API transcript for video ID {video_id}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error fetching YouTube API transcript for {video_id}: {e}", exc_info=True)
        return None


def transcribe_with_local_whisper(
    video_file_path: str, output_dir: str, base_title: str, model_name: str, device: str
) -> Optional[Dict[str, Any]]:
    """Transcribes video using local Whisper model (stable_whisper) and saves it."""
    if stable_whisper is None:
        logging.error("stable_whisper library is not installed/imported. Cannot perform local Whisper transcription.")
        return None
    
    if not ensure_ffmpeg_is_accessible(): # Whisper needs ffprobe (part of ffmpeg)
        logging.error("FFMPEG/FFPROBE not accessible for local Whisper transcription.")
        return None

    transcript_filename_json = f"{sanitize_filename(base_title)}{constants.LOCAL_WHISPER_TRANSCRIPT_SUFFIX_FORMAT.format(model_name=model_name)}"
    transcript_path = os.path.join(output_dir, transcript_filename_json)
    model_instance = None
    try:
        logging.info(
            f"Local Whisper: Processing '{os.path.basename(video_file_path)}', model '{model_name}', device '{device.upper()}'"
        )
        if os.path.exists(transcript_path):
            logging.info(f"Local Whisper transcript already exists: {transcript_path}. Loading.")
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
            return {"data": transcript_data, "path": transcript_path, "type": constants.TRANSCRIPT_TYPE_LOCAL_WHISPER}

        logging.info(f"Loading local Whisper model '{model_name}' onto {device.upper()}...")
        model_instance = stable_whisper.load_model(model_name, device=device)
        
        use_fp16 = (device == "cuda")
        logging.info(f"Transcribing with fp16={use_fp16}...")
        result = model_instance.transcribe(video_file_path, fp16=use_fp16) 

        transcript_data = []
        segments_from_result = None
        if hasattr(result, 'segments'): 
            segments_from_result = result.segments
        elif isinstance(result, dict) and 'segments' in result: 
            segments_from_result = result['segments']

        if segments_from_result:
            for seg_obj_or_dict in segments_from_result:
                start = round(getattr(seg_obj_or_dict, 'start', seg_obj_or_dict.get('start', 0.0) if isinstance(seg_obj_or_dict,dict) else 0.0), 2)
                end = round(getattr(seg_obj_or_dict, 'end', seg_obj_or_dict.get('end', 0.0) if isinstance(seg_obj_or_dict,dict) else 0.0), 2)
                transcript_data.append({
                    "text": getattr(seg_obj_or_dict, 'text', seg_obj_or_dict.get('text', '') if isinstance(seg_obj_or_dict,dict) else '').strip(),
                    "start": start,
                    "end": end,
                    "duration": round(end - start, 2)
                })
            logging.info(f"Local Whisper transcription successful. Found {len(transcript_data)} segments.")
        else:
            logging.error("Local Whisper transcription result did not contain 'segments'.")
            return None

        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved local Whisper transcript to: {transcript_path}")
        return {"data": transcript_data, "path": transcript_path, "type": constants.TRANSCRIPT_TYPE_LOCAL_WHISPER}

    except Exception as e:
        logging.error(f"Error during local Whisper transcription: {e}", exc_info=True)
        return None
    finally:
        if model_instance: 
            clear_gpu_mem(model_instance)


def transcribe_with_openai_api(
    video_file_path: str, output_dir: str, base_title: str, openai_api_key: str
) -> Optional[Dict[str, Any]]:
    """Transcribes video using OpenAI Whisper API after extracting audio, and saves transcript."""
    temp_audio_file_path = None 
    transcript_filename_json = f"{sanitize_filename(base_title)}{constants.OPENAI_API_TRANSCRIPT_SUFFIX}"
    transcript_path = os.path.join(output_dir, transcript_filename_json)
    
    try:
        logging.info(f"OpenAI API Transcription: Processing video '{os.path.basename(video_file_path)}'")
        if os.path.exists(transcript_path):
            logging.info(f"OpenAI API transcript already exists: {transcript_path}. Loading.")
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
            return {"data": transcript_data, "path": transcript_path, "type": constants.TRANSCRIPT_TYPE_OPENAI_API}

        temp_audio_file_path = _extract_audio_for_api(video_file_path, base_title) # output_dir for temp audio is now constants.TEMP_AUDIO_DIR
        if not temp_audio_file_path:
            logging.error("Failed to extract audio, cannot proceed with OpenAI API transcription.")
            return None
        
        logging.info(f"Transcribing extracted audio '{temp_audio_file_path}' with OpenAI Whisper API...")
        client = OpenAIClient(api_key=openai_api_key)
        with open(temp_audio_file_path, "rb") as audio_file_obj:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file_obj,
                response_format="verbose_json", 
                timestamp_granularities=["segment"]
            )
        
        transcript_data = []
        # Access segments from the response object (OpenAI SDK v1.x+)
        api_segments = response.segments if hasattr(response, 'segments') and response.segments else []

        if api_segments:
            for segment_from_api in api_segments:
                start = round(segment_from_api.start, 2) # Direct attribute access
                end = round(segment_from_api.end, 2)     # Direct attribute access
                transcript_data.append({
                    "text": segment_from_api.text.strip(), # Direct attribute access
                    "start": start,
                    "end": end,
                    "duration": round(end - start, 2)
                })
            logging.info(f"OpenAI API transcription successful. Found {len(transcript_data)} segments.")
        elif hasattr(response, 'text') and response.text and not api_segments : 
             logging.warning("OpenAI API returned full text but no segments. Creating a single segment with unknown timestamps.")
             # Estimate duration if possible, otherwise 0. For a long text, this is not ideal.
             # This part might need refinement if full text without segments is common.
             estimated_duration = len(response.text.split()) / 3 # Rough estimate: 3 words per second
             transcript_data.append({
                 "text": response.text.strip(), 
                 "start": 0.0, 
                 "end": round(estimated_duration, 2), # Placeholder
                 "duration": round(estimated_duration, 2) # Placeholder
             })
        else:
            logging.error(f"OpenAI API transcription did not return expected segments or text. Response: {response}")
            return None

        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved OpenAI API transcript to: {transcript_path}")
        return {"data": transcript_data, "path": transcript_path, "type": constants.TRANSCRIPT_TYPE_OPENAI_API}

    except Exception as e:
        logging.error(f"Error during OpenAI API transcription process: {e}", exc_info=True)
        return None
    finally:
        if temp_audio_file_path and os.path.exists(temp_audio_file_path):
            try:
                os.remove(temp_audio_file_path)
                logging.info(f"Deleted temporary audio file: {temp_audio_file_path}")
            except Exception as e_del:
                logging.error(f"Failed to delete temporary audio file {temp_audio_file_path}: {e_del}")
```

`video_clipper_streamlit/services\video_processing_service.py`:

```py
# services/video_processing_service.py

import os
import logging
from typing import Optional, Dict, Any
from pytubefix import YouTube
# If you were to handle Streamlit's UploadedFile directly here, you'd import:
# from streamlit.runtime.uploaded_file_manager import UploadedFile
# But it's better to pass file-like objects or paths from app.py

from services.system_service import sanitize_filename # Use sanitize_filename from system_service

def download_youtube_video(
    youtube_url: str, 
    output_dir: str, 
    desired_resolution: Optional[str] = "720p"
) -> Optional[Dict[str, str]]:
    """
    Downloads a YouTube video to the specified output directory.

    Args:
        youtube_url: The URL of the YouTube video.
        output_dir: The directory to save the downloaded video.
        desired_resolution: Preferred resolution (e.g., "720p", "360p", "best_progressive").

    Returns:
        A dictionary with "path", "title" (sanitized), and "id" of the video, or None on failure.
    """
    try:
        logging.info(f"Fetching video info for YouTube URL: {youtube_url}")
        yt = YouTube(youtube_url)
        video_title_raw = yt.title
        video_base_title = sanitize_filename(video_title_raw)
        if not video_base_title: # Fallback if title sanitization results in empty string
            video_base_title = f"youtube_video_{yt.video_id}"

        video_filename_mp4 = f"{video_base_title}.mp4"
        download_path = os.path.join(output_dir, video_filename_mp4)

        logging.info(f"Original Video Title: {video_title_raw}")
        logging.info(f"Sanitized Base Title for Filename: {video_base_title}")
        logging.info(f"Attempting to download to: {download_path}")

        if os.path.exists(download_path):
            logging.info(f"Video already exists at {download_path}. Using existing file.")
            return {"path": download_path, "title": video_base_title, "id": yt.video_id}

        streams_query = yt.streams.filter(progressive=True, file_extension='mp4')
        stream_to_download = None

        if desired_resolution and desired_resolution != "best_progressive":
            stream_to_download = streams_query.filter(res=desired_resolution).first()
            if not stream_to_download:
                logging.warning(
                    f"Desired resolution '{desired_resolution}' not found as progressive MP4. "
                    "Attempting to get the best available progressive stream."
                )
                stream_to_download = streams_query.order_by('resolution').desc().first()
        else: # "best_progressive" or None
            stream_to_download = streams_query.order_by('resolution').desc().first()

        if stream_to_download:
            logging.info(
                f"Found video stream: Resolution {stream_to_download.resolution}, "
                f"MIME Type {stream_to_download.mime_type}"
            )
            stream_to_download.download(output_path=output_dir, filename=video_filename_mp4)
            logging.info(f"Successfully downloaded video: {download_path}")
            return {"path": download_path, "title": video_base_title, "id": yt.video_id}
        else:
            logging.error("No suitable progressive MP4 stream found for this YouTube video.")
            return None
    except Exception as e:
        logging.error(f"Error downloading YouTube video from {youtube_url}: {e}", exc_info=True)
        return None

def save_uploaded_mp4(
    uploaded_file_obj: Any, # Should be Streamlit's UploadedFile object passed from app.py
    output_dir: str,
    base_title_suggestion: str # A suggested base title from the original filename
) -> Optional[str]:
    """
    Saves a Streamlit UploadedFile object (MP4) to a local path.

    Args:
        uploaded_file_obj: The file object from st.file_uploader().
        output_dir: The directory to save the file.
        base_title_suggestion: A sanitized base title derived from the uploaded file's name.

    Returns:
        The full path to the saved file, or None on failure.
    """
    try:
        if not uploaded_file_obj:
            logging.warning("No file object provided to save_uploaded_mp4.")
            return None

        # Ensure the base title is safe and has an extension
        final_base_title = sanitize_filename(base_title_suggestion)
        if not final_base_title:
            final_base_title = "uploaded_video" # Default fallback
        
        # Ensure output_dir exists (though app.py should also do this)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created directory for uploaded file: {output_dir}")

        # Construct a unique filename if needed, or just use the base title
        # For simplicity, we'll use the base title directly. Add timestamp or UUID for uniqueness if many uploads expected.
        output_filename = f"{final_base_title}.mp4"
        output_filepath = os.path.join(output_dir, output_filename)
        
        logging.info(f"Saving uploaded MP4 '{uploaded_file_obj.name}' as '{output_filepath}'")
        with open(output_filepath, "wb") as f:
            f.write(uploaded_file_obj.getbuffer()) # getbuffer() is for Streamlit UploadedFile
        
        logging.info(f"Successfully saved uploaded file to: {output_filepath}")
        return output_filepath
    except Exception as e:
        logging.error(f"Error saving uploaded MP4 file: {e}", exc_info=True)
        return None
```

`video_clipper_streamlit/test_cuda.py`:

```py
# test_cuda.py
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version PyTorch built with: {torch.version.cuda}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

`video_clipper_streamlit/ui_components\results_display.py`:

```py
# ui_components/results_display.py
import streamlit as st
import os
import time
# Assuming utils re-exports sanitize_filename if needed here, or import from system_service
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
```

`video_clipper_streamlit/ui_components\sidebar.py`:

```py
# ui_components/sidebar.py
import streamlit as st
from core import constants
# utils might be needed if get_openai_api_key is still there, or import directly from system_service
# For this structure, let's assume app.py passes the key or utils.py (facade) is used
import utils # Assuming utils re-exports system_service.get_openai_api_key

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
```

`video_clipper_streamlit/ui_components\status_logger.py`:

```py
# ui_components/status_logger.py
import streamlit as st
import time

def add_log_entry(message: str, level: str = "info"):
    """Appends a new message to the processing log in session state."""
    timestamp = time.strftime("%H:%M:%S")
    if "processing_log" not in st.session_state: # Initialize if called early
        st.session_state.processing_log = []
    st.session_state.processing_log.append({"time": timestamp, "level": level.upper(), "message": message})

def display_log(log_container_placeholder): # Pass the placeholder
    """Displays the processing log from session state."""
    if "processing_log" in st.session_state and st.session_state.processing_log:
        log_html = ""
        for entry in reversed(st.session_state.processing_log): # Newest first
            color = {"INFO": "#555555", "WARNING": "orange", "ERROR": "red", "DEBUG": "#888888"}.get(entry["level"], "inherit")
            # Basic HTML escaping for message
            escaped_message = st.markdown._html.escape(entry['message'])
            log_html += f"<div style='color: {color}; font-family: monospace; font-size: 0.9em;'>{entry['time']} [{entry['level']}] {escaped_message}</div>"
        
        with log_container_placeholder: # Use the passed placeholder
             st.markdown(f"""
             <div style="background-color: #f0f2f6; border: 1px solid #cccccc; padding: 10px; border-radius: 5px; max-height: 300px; overflow-y: auto; overflow-x: hidden; word-wrap: break-word;">
                 {log_html}
             </div>
             """, unsafe_allow_html=True)
```

`video_clipper_streamlit/utils.py`:

```py
# video_clipper_streamlit/utils.py (Thin Facade)

from typing import Dict, Any, Callable

# Assuming 'orchestrators' folder is at the same level as this utils.py
from orchestrators import youtube_pipeline, local_mp4_pipeline 

def process_youtube_url(*args, **kwargs) -> Dict[str, Any]:
    return youtube_pipeline.run_youtube_pipeline(*args, **kwargs)

def process_uploaded_mp4(*args, **kwargs) -> Dict[str, Any]:
    return local_mp4_pipeline.run_local_mp4_pipeline(*args, **kwargs)

# Re-export necessary functions from services directly through utils
from services.system_service import get_nvidia_smi_output, check_gpu_availability, get_openai_api_key
from services.system_service import sanitize_filename # If app.py needs it (currently it doesn't directly)
# Add other re-exports if app.py calls more utils.service_name.function_name
```