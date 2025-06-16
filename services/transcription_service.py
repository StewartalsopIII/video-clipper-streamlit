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