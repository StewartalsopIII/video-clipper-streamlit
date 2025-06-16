# orchestrators/youtube_pipeline.py
import logging
from typing import Dict, Any, Callable, List, Optional # Ensure all needed types are imported

from core import constants
from services import system_service, video_processing_service, transcription_service 
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