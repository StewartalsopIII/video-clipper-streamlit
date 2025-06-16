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