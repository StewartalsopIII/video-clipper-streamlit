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