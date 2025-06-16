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