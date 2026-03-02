"""
SAM3 Utilities Package

Centralised utility modules for logging, profiling, system info,
video probing, visualisation, and general helpers.
"""

from sam3.utils.ffmpeglib import FFMpegLib, ffmpeg_lib
from sam3.utils.helpers import ram_stat, run_cmd, sanitize_filename, vram_stat
from sam3.utils.logger import LOG_LEVELS, get_logger
from sam3.utils.profiler import profile
from sam3.utils.system_info import (
    available_ram,
    cpu_cores,
    cpu_usage,
    get_system_info,
    total_ram,
)
from sam3.utils.visualization import (
    draw_box_on_image,
    load_frame,
    normalize_bbox,
    plot_results,
    prepare_masks_for_visualization,
    show_box,
    show_mask,
    show_points,
    visualize_formatted_frame_output,
)

__all__ = [
    # logger
    "get_logger",
    "LOG_LEVELS",
    # helpers
    "run_cmd",
    "sanitize_filename",
    "vram_stat",
    "ram_stat",
    # profiler
    "profile",
    # system_info
    "available_ram",
    "total_ram",
    "cpu_usage",
    "cpu_cores",
    "get_system_info",
    # ffmpeglib
    "FFMpegLib",
    "ffmpeg_lib",
    # visualization
    "normalize_bbox",
    "draw_box_on_image",
    "plot_results",
    "show_box",
    "show_mask",
    "show_points",
    "load_frame",
    "prepare_masks_for_visualization",
    "visualize_formatted_frame_output",
]
