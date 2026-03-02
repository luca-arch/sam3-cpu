"""Backward-compatible re-export — moved to sam3.utils.system_info."""

from sam3.utils.system_info import available_ram, cpu_cores, cpu_usage, get_system_info, total_ram

__all__ = ["available_ram", "total_ram", "cpu_usage", "cpu_cores", "get_system_info"]
