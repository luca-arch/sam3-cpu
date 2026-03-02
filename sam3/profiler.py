"""Backward-compatible re-export — moved to sam3.utils.profiler."""

from sam3.utils.profiler import clear_profiling_results, get_profiling_results, profile

__all__ = ["profile", "get_profiling_results", "clear_profiling_results"]
