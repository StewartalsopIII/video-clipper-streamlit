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

