import functools
import math
from typing import Literal

from sam3.__globals import (
    DEFAULT_MIN_CHUNK_OVERLAP,
    DEFAULT_MIN_VIDEO_FRAMES,
    DEVICE,
    IMAGE_INFERENCE_MB,
    RAM_USAGE_PERCENT,
    VIDEO_INFERENCE_MB,
    VRAM_USAGE_PERCENT,
    logger,
)
from sam3.memory_optimizer import (
    clear_memory,
    estimate_per_frame_bytes,
)
from sam3.utils.ffmpeglib import ffmpeg_lib
from sam3.utils.helpers import ram_stat, vram_stat


class MemoryError(Exception):
    """Custom exception for memory-related errors."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"MemoryError: {self.message}"

    def __repr__(self):
        return f"MemoryError(message={self.message})"


class MemoryManager:
    def __init__(self):
        pass

    def compute_memory_safe_frames(
        self,
        width: int,
        height: int,
        device: str = DEVICE.type,
        type: Literal["video", "image"] = "video",
        max_memory_bytes: int = None,
    ):
        """Compute maximum frames that fit safely in available memory.

        Uses the calibrated ``estimate_per_frame_bytes()`` from
        ``sam3.memory_optimizer`` which accounts for model state overhead
        (feature maps, cached masks, attention KV caches) — not just raw
        pixel size.

        Parameters
        ----------
        max_memory_bytes : int, optional
            Simulate a smaller device by capping total memory.  When set,
            ``free`` is computed as ``max_memory_bytes − used`` so the
            chunk planner thinks it only has this much VRAM/RAM.
        """
        # Clear stale CUDA cache before measuring so we get accurate free memory
        clear_memory(device, full_gc=True)

        # Calibrated per-frame estimate (includes model state overhead)
        bytes_per_frame = estimate_per_frame_bytes(width, height, device)

        logger.debug(f"Frame size in MB: {bytes_per_frame / (1024**2):.2f} MB")

        if device == "cpu":
            memory_info = ram_stat()
            percent = RAM_USAGE_PERCENT
            available_key = "available"
        elif device == "cuda":
            memory_info = vram_stat()
            percent = VRAM_USAGE_PERCENT
            available_key = "free"
        else:
            raise ValueError(f"Unsupported device type: {device}")

        # Apply simulated memory cap (for testing with smaller devices)
        if max_memory_bytes is not None and max_memory_bytes > 0:
            real_total = memory_info["total"]
            real_used = real_total - memory_info[available_key]
            sim_total = max_memory_bytes
            sim_free = max(sim_total - real_used, 0)
            memory_info = dict(memory_info)  # copy
            memory_info["total"] = sim_total
            memory_info[available_key] = sim_free
            logger.debug(
                f"Simulated memory cap: {max_memory_bytes / (1024**3):.1f} GB "
                f"(real total: {real_total / (1024**3):.1f} GB, "
                f"sim free: {sim_free / (1024**3):.1f} GB)"
            )

        if type == "video":
            available_bytes = memory_info[available_key] - VIDEO_INFERENCE_MB * 1024**2
        else:
            available_bytes = memory_info[available_key] - IMAGE_INFERENCE_MB * 1024**2

        if available_bytes <= 0:
            least_extra_mb = DEFAULT_MIN_VIDEO_FRAMES * (
                VIDEO_INFERENCE_MB if type == "video" else IMAGE_INFERENCE_MB
            ) + bytes_per_frame // (1024**2)
            logger.warning(
                f"Not enough available memory for inference. Available: {memory_info[available_key]} bytes, Required: {least_extra_mb} MB. Consider freeing up memory or reducing video resolution."
            )
            return 0

        usable_bytes = int(available_bytes * percent)
        max_frames = usable_bytes // bytes_per_frame
        if max_frames < DEFAULT_MIN_VIDEO_FRAMES:
            logger.warning(
                f"Estimated max frames ({max_frames}) is below the minimum threshold ({DEFAULT_MIN_VIDEO_FRAMES}). Consider freeing up memory or reducing video resolution."
            )
            return 0
        logger.debug(
            f"Estimated max frames that fit in RAM: {max_frames} frames ({max_frames * bytes_per_frame / (1024**2):.2f} MB)"
        )
        return max_frames

    def generate_chunks(
        self,
        total_frames: int,
        chunk_size: int,
        chunk_spread: Literal["even", "default"] = "default",
        overlap: int = None,
    ):
        """Generate chunk index ranges."""
        if overlap is None or overlap < DEFAULT_MIN_CHUNK_OVERLAP:
            overlap = DEFAULT_MIN_CHUNK_OVERLAP

        if chunk_spread == "even":
            chunk_size = math.ceil(total_frames / math.ceil(total_frames / chunk_size))
            logger.debug(f"Adjusted chunk size for even spread: {chunk_size} frames")

        stride = chunk_size - overlap
        chunks = []

        start = 0
        idx = 0

        while start < total_frames:
            end = min(start + chunk_size - 1, total_frames - 1)
            if end > start:
                chunks.append({"chunk": idx, "start": start, "end": end})
            start += stride
            idx += 1

        return chunks

    def chunk_plan_video(
        self,
        video_file: str,
        device: str = DEVICE.type,
        chunk_spread: Literal["even", "default"] = "default",
        max_memory_bytes: int = None,
    ):
        # Placeholder for actual chunk planning logic
        logger.info(f"Planning memory chunks for video: {video_file}")

        video_info = ffmpeg_lib.get_video_info(video_file)
        if video_info is None:
            logger.warning(f"Could not retrieve video info for: {video_file}")
            return []

        logger.info(f"Video info: {video_info}")

        frames_per_chunk = self.compute_memory_safe_frames(
            video_info["width"],
            video_info["height"],
            device,
            type="video",
            max_memory_bytes=max_memory_bytes,
        )

        fps = round(video_info.get("fps", 25))  # Default to 25 if FPS info is missing
        if frames_per_chunk == 0:
            logger.warning(
                f"Memory constraints are too tight to process any frames for video: {video_file}. Consider freeing up memory or reducing video resolution."
            )
            raise MemoryError("Insufficient memory to process video frames.")

        frames_per_chunk = fps * (frames_per_chunk // fps)  # Adjust to be a multiple of FPS for better chunking

        logger.info(f"Estimated frames per chunk: {frames_per_chunk}")

        video_chunks = self.generate_chunks(video_info["nb_frames"], frames_per_chunk, chunk_spread=chunk_spread)

        metadata = {
            "video": video_file,
            "width": video_info["width"],
            "height": video_info["height"],
            "duration": video_info["duration"],
            "nb_frames": video_info["nb_frames"],
            "fps": video_info["fps"],
            "frames_per_chunk": frames_per_chunk,
            "chunks": video_chunks,
        }

        logger.info(f"Generated {len(video_chunks)} chunks for video processing.")
        logger.info(f"Metadata: {metadata}")

        return metadata, video_chunks


# Singleton instance of MemoryManager
memory_manager = MemoryManager()


# Decorator to check memory before executing a function
def mem_check(device=DEVICE):
    """Profile decorator with global enable/disable control via sam3.__globals.ENABLE_PROFILING"""

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pass

        return wrapper

    return decorator
