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