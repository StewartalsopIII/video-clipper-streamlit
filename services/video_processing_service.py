# services/video_processing_service.py

import os
import logging
from typing import Optional, Dict, Any
from pytubefix import YouTube

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