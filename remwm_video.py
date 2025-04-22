import sys
import click
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
import torch
import tqdm
from loguru import logger
import subprocess
import math
import os
import shutil

# Import necessary functions from remwm.py
# Assuming remwm.py is in the same directory
try:
    from remwm import get_watermark_mask, process_image_with_lama, TaskType
except ImportError:
    logger.error("Could not import functions from remwm.py. Make sure it's in the same directory.")
    sys.exit(1)

# --- Constants ---
MODELS_DIR = Path("./models")
TEMP_DIR = Path("./temp")
# Note: Set the TORCH_HOME environment variable to MODELS_DIR before running
# export TORCH_HOME=$(pwd)/models
os.environ['TORCH_HOME'] = str(MODELS_DIR.resolve())

# --- Helper Functions ---

def run_ffmpeg_command(command_list):
    """Executes an ffmpeg command and logs output."""
    logger.info(f"Running ffmpeg command: {' '.join(command_list)}")
    try:
        process = subprocess.run(command_list, check=True, capture_output=True, text=True)
        logger.debug(f"ffmpeg stdout:\n{process.stdout}")
        logger.debug(f"ffmpeg stderr:\n{process.stderr}")
        logger.info("ffmpeg command completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg command failed with exit code {e.returncode}")
        logger.error(f"ffmpeg stderr:\n{e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("ffmpeg command not found. Make sure ffmpeg is installed and in your PATH.")
        raise

def get_video_properties(video_path: Path):
    """Gets video properties using ffprobe."""
    logger.info(f"Getting properties for video: {video_path}")
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output = process.stdout.strip().split('\n')
        if len(output) < 4:
            raise ValueError(f"Could not parse ffprobe output for {video_path}: {output}")

        width = int(output[0])
        height = int(output[1])
        fps_str = output[2]
        duration = float(output[3])

        # Evaluate fps fraction (e.g., "30000/1001")
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = float(num) / den
        else:
            fps = float(fps_str)

        logger.info(f"Video Properties: Width={width}, Height={height}, FPS={fps:.2f}, Duration={duration:.2f}s")
        return width, height, fps, duration

    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe command failed for {video_path}: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("ffprobe command not found. Make sure ffmpeg (which includes ffprobe) is installed and in your PATH.")
        raise
    except Exception as e:
        logger.error(f"Error getting video properties for {video_path}: {e}")
        raise


# --- Main Processing Logic ---

@click.command()
@click.argument("input_video", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument("output_video", type=click.Path(resolve_path=True))
@click.option("--chunk-minutes", type=float, default=5.0, help="Duration of video chunks to process in minutes.")
@click.option("--max-bbox-percent", default=10.0, help="Maximum percentage of the image that a watermark bounding box can cover.")
# Add other options from remwm.py if needed, e.g., --transparent
def main(input_video: str, output_video: str, chunk_minutes: float, max_bbox_percent: float):
    """
    Removes watermarks from a VIDEO file by processing it in chunks.
    Requires ffmpeg and ffprobe to be installed and in the system PATH.
    Ensure TORCH_HOME is set to ./models if LaMa model needs downloading.
    """
    input_path = Path(input_video)
    output_path = Path(output_video)
    chunk_seconds = chunk_minutes * 60

    logger.info(f"Starting watermark removal for video: {input_path}")
    logger.info(f"Output video will be saved to: {output_path}")
    logger.info(f"Processing in chunks of {chunk_minutes} minutes ({chunk_seconds} seconds)")

    # --- Setup ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if TEMP_DIR.exists():
        logger.warning(f"Temporary directory {TEMP_DIR} already exists. Removing it.")
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True)
    logger.info(f"Created temporary directory: {TEMP_DIR}")
    logger.info(f"Models will be stored in: {MODELS_DIR}")
    logger.info(f"Make sure TORCH_HOME is set to {MODELS_DIR.resolve()} if LaMa needs downloading.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # --- Load Models ---
    try:
        logger.info("Loading Florence-2 model...")
        # Use cache_dir to specify download/load location
        florence_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True,
            cache_dir=str(MODELS_DIR.resolve())
        ).to(device).eval()
        florence_processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True,
            cache_dir=str(MODELS_DIR.resolve())
        )
        logger.info("Florence-2 model loaded successfully.")

        # Assuming --transparent is not used for now, load LaMa
        logger.info("Loading LaMa model via iopaint...")
        # ModelManager might respect TORCH_HOME or HF_HOME set earlier
        model_manager = ModelManager(name="lama", device=device)
        logger.info("LaMa model loaded successfully.")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.error("Ensure you have internet connectivity if models need downloading.")
        logger.error(f"Check if models exist in {MODELS_DIR} or if TORCH_HOME is set correctly.")
        sys.exit(1)

    # --- Get Video Info ---
    try:
        width, height, fps, duration = get_video_properties(input_path)
        if duration <= 0:
             raise ValueError("Video duration is zero or negative.")
    except Exception as e:
        logger.error(f"Could not get properties for video {input_path}: {e}")
        sys.exit(1)

    # --- Extract Audio ---
    audio_path = TEMP_DIR / f"{input_path.stem}_audio.aac"
    try:
        run_ffmpeg_command([
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vn",           # No video
            "-acodec", "copy", # Copy audio stream directly
            str(audio_path)
        ])
        logger.info(f"Extracted audio to {audio_path}")
    except Exception as e:
        logger.warning(f"Could not extract audio using 'copy': {e}. Trying AAC encoding.")
        try:
            # Fallback to encoding if copy fails (e.g., incompatible codec)
            run_ffmpeg_command([
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-vn",
                "-acodec", "aac", # Encode to AAC
                "-b:a", "192k",   # Reasonable bitrate
                str(audio_path)
            ])
            logger.info(f"Extracted and encoded audio to {audio_path}")
        except Exception as e_enc:
            logger.error(f"Failed to extract audio even with encoding: {e_enc}")
            # Decide if processing should continue without audio
            logger.warning("Proceeding without audio track.")
            audio_path = None # Signal that audio is not available


    # --- Process Video in Chunks ---
    num_chunks = math.ceil(duration / chunk_seconds)
    processed_chunk_files = []
    logger.info(f"Video will be processed in {num_chunks} chunks.")

    for i in range(num_chunks):
        chunk_start_time = i * chunk_seconds
        chunk_duration = min(chunk_seconds, duration - chunk_start_time)
        chunk_num = i + 1
        logger.info(f"--- Processing Chunk {chunk_num}/{num_chunks} (Start: {chunk_start_time:.2f}s, Duration: {chunk_duration:.2f}s) ---")

        # Define chunk paths
        raw_chunk_path = TEMP_DIR / f"chunk_{chunk_num}.mp4"
        processed_chunk_path = TEMP_DIR / f"processed_chunk_{chunk_num}.mp4"

        # 1. Extract video chunk (no audio) using ffmpeg
        try:
            run_ffmpeg_command([
                "ffmpeg", "-y",
                "-ss", str(chunk_start_time), # Start time
                "-i", str(input_path),
                "-t", str(chunk_duration),   # Duration of chunk
                "-map", "0:v",              # Select video stream
                "-c:v", "copy",             # Copy video stream directly (fast)
                str(raw_chunk_path)
            ])
            logger.info(f"Extracted raw video chunk to {raw_chunk_path}")
        except Exception as e:
            logger.error(f"Failed to extract chunk {chunk_num}: {e}")
            # Consider how to handle chunk failure (skip? abort?)
            logger.error("Aborting processing due to chunk extraction failure.")
            shutil.rmtree(TEMP_DIR)
            sys.exit(1)

        # 2. Process frames in the chunk
        try:
            cap = cv2.VideoCapture(str(raw_chunk_path))
            if not cap.isOpened():
                raise IOError(f"Cannot open video chunk: {raw_chunk_path}")

            # Use FFV1 for high quality intermediate, or libx264 with low CRF
            # fourcc = cv2.VideoWriter_fourcc(*'FFV1') # Lossless, HUGE files
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Common, good quality
            # Adjust quality settings as needed, e.g., CRF for x264 if available
            out = cv2.VideoWriter(str(processed_chunk_path), fourcc, fps, (width, height))
            if not out.isOpened():
                 raise IOError(f"Cannot open video writer for: {processed_chunk_path}")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Processing {frame_count} frames in chunk {chunk_num}...")

            for frame_idx in tqdm.tqdm(range(frame_count), desc=f"Chunk {chunk_num}"):
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Could not read frame {frame_idx} from chunk {chunk_num}. Stopping chunk processing.")
                    break

                # Convert frame BGR (OpenCV) to RGB (PIL)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # Get watermark mask
                mask_image = get_watermark_mask(pil_image, florence_model, florence_processor, device, max_bbox_percent)

                # Apply LaMa inpainting (assuming not transparent mode)
                # Convert PIL image and mask to NumPy arrays for LaMa
                lama_result_rgb = process_image_with_lama(np.array(pil_image), np.array(mask_image), model_manager)

                # Convert result back to BGR for OpenCV VideoWriter
                processed_frame_bgr = cv2.cvtColor(lama_result_rgb, cv2.COLOR_RGB2BGR)

                out.write(processed_frame_bgr)

            cap.release()
            out.release()
            logger.info(f"Finished processing frames for chunk {chunk_num}.")
            processed_chunk_files.append(str(processed_chunk_path.resolve()))

        except Exception as e:
            logger.error(f"Error processing frames in chunk {chunk_num}: {e}")
            # Clean up potentially corrupted chunk file?
            if processed_chunk_path.exists():
                processed_chunk_path.unlink()
            logger.error("Aborting processing due to frame processing failure.")
            shutil.rmtree(TEMP_DIR)
            sys.exit(1)
        finally:
            # Ensure capture and writer are released
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            if 'out' in locals() and out.isOpened():
                out.release()
            # 3. Delete raw chunk
            if raw_chunk_path.exists():
                raw_chunk_path.unlink()
                logger.info(f"Deleted raw chunk: {raw_chunk_path}")


    # --- Concatenate Processed Chunks ---
    if not processed_chunk_files:
        logger.error("No chunks were processed successfully. Aborting.")
        shutil.rmtree(TEMP_DIR)
        sys.exit(1)

    concat_list_path = TEMP_DIR / "concat_list.txt"
    with open(concat_list_path, "w") as f:
        for chunk_file in processed_chunk_files:
            # Need to escape special characters for ffmpeg concat demuxer if any
            # For simplicity, assuming standard paths for now.
            f.write(f"file '{chunk_file}'\n")
    logger.info(f"Created concatenation list: {concat_list_path}")

    final_video_no_audio_path = TEMP_DIR / "final_video_no_audio.mp4"
    try:
        run_ffmpeg_command([
            "ffmpeg", "-y",
            "-f", "concat",      # Use concat demuxer
            "-safe", "0",        # Allow absolute paths in list
            "-i", str(concat_list_path),
            "-c", "copy",        # Copy streams directly (no re-encoding)
            str(final_video_no_audio_path)
        ])
        logger.info(f"Concatenated processed chunks to {final_video_no_audio_path}")
    except Exception as e:
        logger.error(f"Failed to concatenate video chunks: {e}")
        shutil.rmtree(TEMP_DIR)
        sys.exit(1)


    # --- Merge Video and Audio ---
    logger.info("Merging final video with audio...")
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    merge_cmd = [
        "ffmpeg", "-y",
        "-i", str(final_video_no_audio_path), # Input video
    ]
    if audio_path and audio_path.exists():
         merge_cmd.extend(["-i", str(audio_path)]) # Input audio
         merge_cmd.extend([
             "-map", "0:v:0",    # Map video from first input
             "-map", "1:a:0",    # Map audio from second input
             "-c:v", "copy",     # Copy video stream
             "-c:a", "aac",      # Encode audio (copy might fail if formats differ)
             "-b:a", "192k",
             "-shortest"         # Finish when the shorter stream ends
         ])
    else:
        logger.warning("No audio track found or extracted. Final video will be silent.")
        merge_cmd.extend([
             "-map", "0:v:0",
             "-c:v", "copy",
        ])

    merge_cmd.append(str(output_path))

    try:
        run_ffmpeg_command(merge_cmd)
        logger.info(f"Successfully created final output video: {output_path}")
    except Exception as e:
        logger.error(f"Failed to merge video and audio: {e}")
        shutil.rmtree(TEMP_DIR)
        sys.exit(1)


    # --- Cleanup ---
    logger.info(f"Cleaning up temporary directory: {TEMP_DIR}")
    shutil.rmtree(TEMP_DIR)

    logger.info("Video processing complete!")


if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO") # Log INFO and above to console
    # logger.add("remwm_video.log", level="DEBUG") # Optional: Log DEBUG to file

    main()
