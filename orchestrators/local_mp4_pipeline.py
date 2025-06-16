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