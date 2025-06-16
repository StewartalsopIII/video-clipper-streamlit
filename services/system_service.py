# services/system_service.py

import os
import sys
import subprocess
import logging
import torch
import gc
from typing import Optional

# Module-level flags to track FFMPEG status
_ffmpeg_path_confirmed = False
_ffmpeg_dir_added_to_path = False
_ffmpeg_executable_name = "ffmpeg" # Default, might become full path

def sanitize_filename(name: str) -> str:
    """Cleans a string to be a safe filename component."""
    if not isinstance(name, str):
        name = str(name) 
    sanitized = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name).rstrip()
    sanitized = sanitized.replace(" ", "_")
    # Ensure it doesn't just become underscores or empty
    if not sanitized.strip('_'):
        return "untitled_segment"
    return sanitized if sanitized else "untitled_segment"

def get_openai_api_key() -> Optional[str]:
    """Loads OpenAI API key from environment."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        logging.warning("OpenAI API key not found in environment variables.")
    return key

def check_gpu_availability() -> str:
    """Checks for CUDA availability and returns device string ('cuda' or 'cpu')."""
    if torch.cuda.is_available():
        try:
            # Basic check first
            torch.cuda.device_count() 
            logging.info(f"CUDA is available. PyTorch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
            logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
            return "cuda"
        except RuntimeError as e: # Handles cases like "No CUDA GPUs are available" despite torch.cuda.is_available() being true initially
            logging.warning(f"PyTorch reported CUDA as available, but encountered RuntimeError checking details: {e}. Defaulting to CPU.")
            return "cpu"
        except Exception as e: # Catch any other unexpected errors
            logging.warning(f"CUDA reported as available, but an unexpected error occurred checking details: {e}. Defaulting to CPU.")
            return "cpu"
    else:
        logging.info("CUDA not available. Using CPU.")
        return "cpu"

def get_nvidia_smi_output() -> str:
    """Runs nvidia-smi and returns its output or an error message."""
    try:
        is_windows = os.name == 'nt'
        cmd_to_try = ["nvidia-smi"]
        
        # On Windows, try a common explicit path if the simple command fails
        if is_windows:
            program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
            explicit_path = os.path.join(program_files, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")
            if os.path.exists(explicit_path):
                cmd_to_try = [explicit_path] # Prioritize explicit path if found
            else: # If explicit path not found, rely on PATH and shell=True might be needed
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, shell=True, check=False) # Try with shell=True for PATH resolution
                if result.returncode == 0:
                    return result.stdout
                # If still fails, the generic error below will be caught
        
        # For non-Windows or if explicit Windows path failed/not found
        result = subprocess.run(cmd_to_try, capture_output=True, text=True, shell=False, check=False) # shell=False preferred for safety
        if result.returncode == 0:
            return result.stdout
        else:
            logging.warning(f"nvidia-smi (command: {' '.join(cmd_to_try)}) failed or not found. Return code: {result.returncode}. Error: {result.stderr.strip()}")
            return "nvidia-smi command not found or failed. Is NVIDIA driver installed and nvidia-smi in system PATH?"

    except FileNotFoundError:
        logging.warning("nvidia-smi command not found. Is NVIDIA driver installed and in system PATH?")
        return "nvidia-smi command not found. Ensure NVIDIA drivers are installed and nvidia-smi is in your system PATH."
    except Exception as e:
        logging.error(f"An unexpected error occurred while running nvidia-smi: {e}", exc_info=True)
        return f"An unexpected error occurred while running nvidia-smi: {str(e)}"


def get_ffmpeg_executable_name() -> str:
    """Returns the command name or full path for ffmpeg after ensuring it's accessible."""
    global _ffmpeg_executable_name
    ensure_ffmpeg_is_accessible() # This will set _ffmpeg_executable_name if found locally
    return _ffmpeg_executable_name


def ensure_ffmpeg_is_accessible() -> bool:
    """
    Ensures ffmpeg is accessible.
    Checks for ffmpeg.exe and ffprobe.exe in the app's root directory first.
    If found, prepends that directory to os.environ['PATH'] for the current session.
    Then verifies by trying to run 'ffmpeg -version'.
    Updates a module-level flag `_ffmpeg_path_confirmed`.
    """
    global _ffmpeg_path_confirmed, _ffmpeg_dir_added_to_path, _ffmpeg_executable_name
    if _ffmpeg_path_confirmed:
        return True

    # Determine app root: Assumes services/ is one level below app.py's directory
    current_service_script_dir = os.path.dirname(os.path.abspath(__file__))
    app_root_dir = os.path.dirname(current_service_script_dir)

    local_ffmpeg_exe = "ffmpeg.exe" if os.name == 'nt' else "ffmpeg"
    local_ffprobe_exe = "ffprobe.exe" if os.name == 'nt' else "ffprobe"
    
    path_to_local_ffmpeg = os.path.join(app_root_dir, local_ffmpeg_exe)
    path_to_local_ffprobe = os.path.join(app_root_dir, local_ffprobe_exe)

    ffmpeg_dir_to_add = None

    if os.path.exists(path_to_local_ffmpeg) and os.path.exists(path_to_local_ffprobe):
        logging.info(f"Found local ffmpeg/ffprobe in: {app_root_dir}")
        ffmpeg_dir_to_add = app_root_dir
        _ffmpeg_executable_name = path_to_local_ffmpeg # Use full path if found locally
    else:
        logging.debug(f"Local ffmpeg/ffprobe not found in {app_root_dir}. Will rely on system PATH.")
        _ffmpeg_executable_name = "ffmpeg" # Rely on system PATH

    # Add to PATH if a local directory was found and not already added
    if ffmpeg_dir_to_add and not _ffmpeg_dir_added_to_path:
        logging.info(f"Prepending FFMPEG directory to PATH for this session: {ffmpeg_dir_to_add}")
        os.environ["PATH"] = ffmpeg_dir_to_add + os.pathsep + os.environ.get("PATH", "")
        _ffmpeg_dir_added_to_path = True
    
    # Verify FFMPEG accessibility
    try:
        cmd_to_verify = [_ffmpeg_executable_name, "-version"]
        logging.debug(f"Verifying FFMPEG with command: {' '.join(cmd_to_verify)}")
        # Use shell=True for Windows if just "ffmpeg" is used and it's in PATH,
        # but if _ffmpeg_executable_name is a full path, shell=False is better.
        use_shell = (os.name == 'nt' and _ffmpeg_executable_name == "ffmpeg")

        result = subprocess.run(cmd_to_verify, capture_output=True, text=True, shell=use_shell, check=True)
        logging.info(f"FFMPEG is accessible. Version info (first 100 chars): {result.stdout[:100].strip()}...")
        _ffmpeg_path_confirmed = True
        return True
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logging.error(
            f"FFMPEG command ('{_ffmpeg_executable_name}') failed verification. "
            f"Ensure ffmpeg.exe and ffprobe.exe are in '{app_root_dir}' (if bundled) OR in your system PATH. Error: {e}"
        )
        if isinstance(e, subprocess.CalledProcessError):
            logging.error(f"FFMPEG STDERR: {e.stderr.strip()}")
        _ffmpeg_path_confirmed = False
        return False
    except Exception as e_unexp:
        logging.error(f"Unexpected error verifying FFMPEG: {e_unexp}", exc_info=True)
        _ffmpeg_path_confirmed = False
        return False


def clear_gpu_mem(model_ref=None):
    """Clears GPU memory by deleting model reference and emptying cache."""
    if model_ref is not None:
        try:
            del model_ref
            logging.info("Model reference deleted for GPU memory clearing.")
        except NameError: # model_ref might not exist if an error occurred before its assignment
            logging.debug("Model reference for clearing GPU mem was not defined or already deleted.")

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            logging.info("PyTorch CUDA cache cleared.")
        except Exception as e:
            logging.error(f"Error clearing CUDA cache: {e}", exc_info=True)
    try:
        gc.collect()
        logging.info("Python garbage collection ran.")
    except Exception as e:
        logging.error(f"Error running garbage collection: {e}", exc_info=True)