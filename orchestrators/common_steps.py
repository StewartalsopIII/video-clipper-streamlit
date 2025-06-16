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