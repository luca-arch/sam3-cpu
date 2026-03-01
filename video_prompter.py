#!/usr/bin/env python3
"""
SAM3 Video Prompter

Process a single video with text prompts, click points, and/or mask images
for segmentation.  Automatically chunks the video based on available memory,
runs cross-chunk IoU-based ID remapping for object continuity, and produces
stitched mask videos and overlay.

Usage:
    # Text prompt
    python video_prompter.py --video clip.mp4 --prompts person ball

    # Click points on first frame
    python video_prompter.py --video clip.mp4 --points 320,240 --point-labels 1

    # Mask image as prompt
    python video_prompter.py --video clip.mp4 --masks mask.png

    # Process a specific segment (frames or time)
    python video_prompter.py --video clip.mp4 --prompts player --frame-range 100 500
    python video_prompter.py --video clip.mp4 --prompts player --time-range 4.0 20.0
    python video_prompter.py --video clip.mp4 --prompts player --time-range 00:01:30 00:03:00

    # Custom options
    python video_prompter.py --video clip.mp4 --prompts player \\
        --output results/match --alpha 0.4 --device cpu --keep-temp
"""

# ---- Force-CPU guard (must run before ANY torch/sam3 import) ----
import os as _os, sys as _sys
if '--device' in _sys.argv:
    _i = _sys.argv.index('--device')
    if _i + 1 < len(_sys.argv) and _sys.argv[_i + 1].lower() == 'cpu':
        _os.environ['CUDA_VISIBLE_DEVICES'] = ''
# -----------------------------------------------------------------

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
from PIL import Image


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(b: float) -> str:
    """Human-readable byte size."""
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} PB"


def _table(rows: List[List[str]]):
    """Print a simple ASCII table."""
    if not rows:
        return
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(sep)
    for i, row in enumerate(rows):
        print("|" + "|".join(f" {str(c).ljust(w)} " for c, w in zip(row, widths)) + "|")
        if i == 0:
            print(sep)
    print(sep)


# ---------------------------------------------------------------------------
# Time / frame range parsing
# ---------------------------------------------------------------------------

def _parse_timestamp(value: str) -> float:
    """Parse a timestamp string to seconds.

    Accepts:
        - Plain float/int:  "4.5"  -> 4.5
        - MM:SS:            "1:30" -> 90.0
        - HH:MM:SS:        "0:01:30" -> 90.0
    """
    try:
        return float(value)
    except ValueError:
        pass
    parts = value.split(":")
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    raise ValueError(f"Cannot parse timestamp: '{value}'")


def _resolve_range(
    video_path: Path,
    frame_range: Optional[Tuple[int, int]],
    time_range: Optional[Tuple[str, str]],
) -> Optional[Tuple[int, int]]:
    """Convert frame_range or time_range to an (start_frame, end_frame) tuple.

    Returns *None* when the full video should be processed.
    """
    if frame_range is not None:
        return (frame_range[0], frame_range[1])
    if time_range is not None:
        from sam3.utils.ffmpeglib import ffmpeg_lib
        info = ffmpeg_lib.get_video_info(str(video_path))
        fps = info["fps"]
        start_sec = _parse_timestamp(time_range[0])
        end_sec = _parse_timestamp(time_range[1])
        return (int(start_sec * fps), min(int(end_sec * fps), info["nb_frames"] - 1))
    return None


def _extract_subclip(
    video_path: Path, start_frame: int, end_frame: int, temp_dir: Path
) -> Path:
    """Extract a sub-clip from *video_path* covering [start_frame, end_frame]."""
    from sam3.utils.ffmpeglib import ffmpeg_lib
    temp_dir.mkdir(parents=True, exist_ok=True)
    out = temp_dir / f"subclip_{start_frame}_{end_frame}.mp4"
    ffmpeg_lib.create_video_chunk(str(video_path), str(out), start_frame, end_frame)
    return out


# ---------------------------------------------------------------------------
# Memory validation
# ---------------------------------------------------------------------------

def _validate_video_memory(
    video_path: Path, device: str, max_memory_bytes: int = None,
) -> Dict[str, Any]:
    """Check if there is enough memory to process at least MIN_VIDEO_FRAMES.

    Returns a dict with memory stats, chunk plan, and ``can_process`` flag.

    Parameters
    ----------
    max_memory_bytes : int, optional
        Simulate a smaller device (for testing).
    """
    from sam3.memory_manager import MemoryManager
    from sam3.utils.helpers import ram_stat, vram_stat
    from sam3.utils.ffmpeglib import ffmpeg_lib
    from sam3.__globals import (
        VIDEO_INFERENCE_MB,
        RAM_USAGE_PERCENT,
        VRAM_USAGE_PERCENT,
        DEFAULT_MIN_VIDEO_FRAMES,
    )

    video_info = ffmpeg_lib.get_video_info(str(video_path))
    if video_info is None:
        return {"can_process": False, "error": "Could not read video metadata"}

    mm = MemoryManager()
    max_frames = mm.compute_memory_safe_frames(
        video_info["width"], video_info["height"], device, type="video",
        max_memory_bytes=max_memory_bytes,
    )

    if device == "cuda":
        mem = vram_stat()
        pct = VRAM_USAGE_PERCENT
        available = mem["free"]
    else:
        mem = ram_stat()
        pct = RAM_USAGE_PERCENT
        available = mem["available"]

    # Apply simulated cap to the reported values
    if max_memory_bytes is not None and max_memory_bytes > 0:
        real_used = mem["total"] - available
        sim_total = max_memory_bytes
        sim_available = max(sim_total - real_used, 0)
        mem = dict(mem)
        mem["total"] = sim_total
        available = sim_available

    info: Dict[str, Any] = {
        "video": str(video_path),
        "resolution": f"{video_info['width']}x{video_info['height']}",
        "total_frames": video_info["nb_frames"],
        "fps": round(video_info.get("fps", 25), 2),
        "duration_s": round(video_info.get("duration", 0), 2),
        "device": device,
        "total_memory": mem["total"],
        "available_memory": available,
        "inference_overhead_mb": VIDEO_INFERENCE_MB,
        "max_frames_per_chunk": max_frames,
        "can_process": max_frames >= DEFAULT_MIN_VIDEO_FRAMES,
    }

    if not info["can_process"]:
        frame_bytes = video_info["width"] * video_info["height"] * 3
        needed = (
            VIDEO_INFERENCE_MB * 1024**2
            + DEFAULT_MIN_VIDEO_FRAMES * frame_bytes
        )
        info["deficit_bytes"] = max(needed - available, 0)
    else:
        info["deficit_bytes"] = 0

    return info


def _show_video_memory_table(info: Dict[str, Any]):
    """Print video memory validation table."""
    rows = [
        ["Metric", "Value"],
        ["Video", info.get("video", "?")],
        ["Resolution", info.get("resolution", "?")],
        ["Frames / FPS", f"{info.get('total_frames', '?')} / {info.get('fps', '?')}"],
        ["Duration", f"{info.get('duration_s', '?')} s"],
        ["Device", info.get("device", "?")],
        ["Total memory", _fmt(info.get("total_memory", 0))],
        ["Available memory", _fmt(info.get("available_memory", 0))],
        ["Inference overhead", f"{info.get('inference_overhead_mb', 0)} MB"],
        ["Max frames/chunk", str(info.get("max_frames_per_chunk", 0))],
    ]
    if info["can_process"]:
        rows.append(["Status", "\033[92m✓ Sufficient memory\033[0m"])
    else:
        deficit = info.get("deficit_bytes", 0)
        rows.append(["Status", f"\033[91m✗ Insufficient memory (need {_fmt(deficit)} more)\033[0m"])
    _table(rows)


# ---------------------------------------------------------------------------
# Chunk plan
# ---------------------------------------------------------------------------

def _make_chunk_plan(
    video_path: Path, device: str, chunk_spread: str = "default",
    max_memory_bytes: int = None,
) -> tuple:
    """Create a memory-safe chunk plan.

    Returns (video_metadata, chunk_list).
    """
    from sam3.memory_manager import memory_manager

    metadata, chunks = memory_manager.chunk_plan_video(
        str(video_path), device=device, chunk_spread=chunk_spread,
        max_memory_bytes=max_memory_bytes,
    )
    return metadata, chunks


# ---------------------------------------------------------------------------
# Mask extraction helpers
# ---------------------------------------------------------------------------

def _extract_last_frame_masks(
    result_prompt: dict,
    object_ids: set,
) -> Dict[int, np.ndarray]:
    """Extract masks from the last frame of propagation results."""
    if not result_prompt:
        return {}
    last_idx = max(result_prompt.keys())
    output = result_prompt[last_idx]
    out_ids = output.get("out_obj_ids", [])
    if isinstance(out_ids, np.ndarray):
        out_ids = out_ids.tolist()
    masks = {}
    for oid in object_ids:
        if oid in out_ids:
            idx = out_ids.index(oid)
            m = output["out_binary_masks"][idx]
            masks[oid] = (m.astype(np.uint8) * 255)
    return masks


# ---------------------------------------------------------------------------
# IoU + ID remapping (same algorithm as ChunkProcessor)
# ---------------------------------------------------------------------------

def _compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    ma = a > 127 if a.dtype == np.uint8 else a.astype(bool)
    mb = b > 127 if b.dtype == np.uint8 else b.astype(bool)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter / union) if union > 0 else 0.0


def _match_and_remap(
    result_prompt: dict,
    object_ids: set,
    prev_masks: Dict[int, np.ndarray],
    global_next_id: int,
    iou_threshold: float = 0.25,
):
    """Greedy IoU matching and ID remapping.

    Returns (remapped_result, remapped_ids, id_mapping, updated_next_id, iou_matrix).

    ``iou_matrix`` is a nested dict ``{new_id: {prev_id: iou}}`` with ALL
    pairwise comparisons (not just those above the threshold).  It is ``{}``
    for the first chunk where no previous masks exist.
    """
    iou_matrix: Dict[int, Dict[int, float]] = {}

    if not result_prompt:
        mapping = {}
        for oid in sorted(object_ids):
            mapping[oid] = global_next_id
            global_next_id += 1
        return {}, set(mapping.values()), mapping, global_next_id, iou_matrix

    first_idx = min(result_prompt.keys())
    first_out = result_prompt.get(first_idx)
    if first_out is None:
        mapping = {o: o for o in object_ids}
        return result_prompt, object_ids, mapping, global_next_id, iou_matrix

    out_ids = first_out.get("out_obj_ids", [])
    if isinstance(out_ids, np.ndarray):
        out_ids = out_ids.tolist()

    first_masks = {}
    for oid in out_ids:
        idx = out_ids.index(oid)
        first_masks[oid] = (first_out["out_binary_masks"][idx].astype(np.uint8) * 255)

    if not prev_masks:
        # First chunk — identity mapping
        mapping = {}
        for oid in sorted(object_ids):
            mapping[oid] = global_next_id
            global_next_id += 1
    else:
        pairs = []
        for nid, nm in first_masks.items():
            iou_matrix[nid] = {}
            for pid, pm in prev_masks.items():
                iou = _compute_iou(nm, pm)
                iou_matrix[nid][pid] = round(iou, 6)
                if iou >= iou_threshold:
                    pairs.append((iou, nid, pid))
        pairs.sort(reverse=True)

        mapping = {}
        used = set()
        for iou, nid, pid in pairs:
            if nid in mapping or pid in used:
                continue
            mapping[nid] = pid
            used.add(pid)
            print(f"      Matched obj_{nid} → global_{pid} (IoU={iou:.3f})")

        for oid in sorted(object_ids):
            if oid not in mapping:
                mapping[oid] = global_next_id
                print(f"      New obj_{oid} → global_{global_next_id}")
                global_next_id += 1

    # Apply mapping to all frames
    remapped = {}
    for fidx, output in result_prompt.items():
        ids = output.get("out_obj_ids", [])
        ids_list = ids.tolist() if isinstance(ids, np.ndarray) else list(ids)
        new_out = dict(output)
        new_out["out_obj_ids"] = np.array(
            [mapping.get(o, o) for o in ids_list], dtype=np.int64
        )
        remapped[fidx] = new_out

    return remapped, set(mapping.values()), mapping, global_next_id, iou_matrix


# ---------------------------------------------------------------------------
# Mask saving helpers
# ---------------------------------------------------------------------------

def _save_chunk_masks(
    result_prompt: dict,
    object_ids: set,
    masks_dir: Path,
    width: int,
    height: int,
    total_frames: int,
):
    """Save per-object per-frame PNG masks."""
    for oid in object_ids:
        obj_dir = masks_dir / f"object_{oid}"
        obj_dir.mkdir(parents=True, exist_ok=True)

    for fidx in range(total_frames):
        output = result_prompt.get(fidx)
        for oid in object_ids:
            mask_u8 = np.zeros((height, width), dtype=np.uint8)
            if output is not None:
                out_ids = output.get("out_obj_ids", [])
                if isinstance(out_ids, np.ndarray):
                    out_ids = out_ids.tolist()
                if oid in out_ids:
                    idx = out_ids.index(oid)
                    m = output["out_binary_masks"][idx]
                    if m.any():
                        mask_u8 = (m.astype(np.uint8) * 255)
            png = masks_dir / f"object_{oid}" / f"frame_{fidx:06d}.png"
            Image.fromarray(mask_u8, mode="L").save(png, compress_level=1)


# ---------------------------------------------------------------------------
# Intra-chunk monitoring helpers
# ---------------------------------------------------------------------------

def _ensure_cpu_masks(result: dict) -> None:
    """Convert any GPU tensors in result dict to CPU numpy arrays in-place.

    This must be called before submitting results to the ``AsyncIOWorker``
    to ensure the background thread can access the data safely after CUDA
    memory is released.
    """
    for frame_data in result.values():
        if frame_data is None:
            continue
        masks = frame_data.get("out_binary_masks")
        if masks is not None and hasattr(masks, "cpu"):
            frame_data["out_binary_masks"] = masks.cpu().numpy()
        obj_ids = frame_data.get("out_obj_ids")
        if obj_ids is not None and hasattr(obj_ids, "cpu"):
            frame_data["out_obj_ids"] = obj_ids.cpu().numpy()


def _propagate_with_monitoring(
    driver,
    session_id: str,
    monitor,
    propagation_direction: str = "both",
) -> Tuple[dict, set, dict, bool]:
    """Stream-process video propagation with per-frame memory monitoring.

    Iterates through ``driver.propagate_in_video_streaming()`` frame by
    frame, calling ``monitor.check()`` at each step.  If the monitor
    signals a stop (memory pressure), the generator is broken and partial
    results are returned.

    Parameters
    ----------
    driver : Sam3VideoDriver
        The loaded video driver.
    session_id : str
        Active session from ``driver.start_session()``.
    monitor : IntraChunkMonitor
        Pre-initialised monitor (call ``monitor.start()`` before this).
    propagation_direction : str
        ``"both"``, ``"forward"``, or ``"backward"``.

    Returns
    -------
    result : dict
        ``{frame_idx: outputs}`` for processed frames.
    object_ids : set
        All unique object IDs found.
    frame_objects : dict
        ``{frame_idx: [obj_ids]}`` per frame.
    early_stopped : bool
        ``True`` if monitor triggered an early stop.
    """
    from sam3.memory_optimizer import clear_memory

    result = {}
    object_ids = set()
    frame_objects = {}
    early_stopped = False

    try:
        for frame_idx, outputs, frame_obj_ids in driver.propagate_in_video_streaming(
            session_id, propagation_direction=propagation_direction
        ):
            if not monitor.check(frame_idx):
                early_stopped = True
                break
            result[frame_idx] = outputs
            frame_objects[frame_idx] = frame_obj_ids
            object_ids.update(frame_obj_ids)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            early_stopped = True
            monitor._stop_reason = "oom_exception"
            clear_memory(monitor.device, full_gc=True)
        else:
            raise

    return result, object_ids, frame_objects, early_stopped


# ---------------------------------------------------------------------------
# Stitching and overlay
# ---------------------------------------------------------------------------

def _stitch_masks_to_video(
    chunks_dir: Path,
    prompt_name: str,
    object_ids: set,
    chunk_infos: list,
    overlap: int,
    output_dir: Path,
    fps: float,
    width: int,
    height: int,
):
    """Stitch per-chunk PNG masks into per-object mask videos."""
    output_dir.mkdir(parents=True, exist_ok=True)

    black = np.zeros((height, width), dtype=np.uint8)

    for oid in sorted(object_ids):
        out_path = output_dir / f"object_{oid}_mask.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height), False)

        for ci, cinfo in enumerate(chunk_infos):
            chunk_id = cinfo["chunk"]
            skip = overlap if ci > 0 else 0
            obj_mask_dir = (
                chunks_dir / f"chunk_{chunk_id}" / "masks" / prompt_name / f"object_{oid}"
            )
            if not obj_mask_dir.exists():
                # Object not present in this chunk — write black frames
                chunk_len = cinfo["end"] - cinfo["start"] + 1
                for _ in range(chunk_len - skip):
                    writer.write(black)
                continue
            pngs = sorted(obj_mask_dir.glob("frame_*.png"))
            for png in pngs[skip:]:
                frame = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
                if frame is not None:
                    writer.write(frame)

        writer.release()
        print(f"    Saved mask video: {out_path.name}")


def _create_overlay_video(
    video_path: Path,
    mask_videos: List[Path],
    output_path: Path,
    alpha: float = 0.5,
):
    """Overlay coloured masks onto the original video."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h), True)

    # Open mask video readers
    mask_caps = [cv2.VideoCapture(str(p)) for p in mask_videos if p.exists()]

    # Colour palette
    colours = [
        (30, 144, 255), (255, 50, 50), (50, 205, 50),
        (255, 165, 0), (148, 103, 189), (255, 215, 0),
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.copy()
        for i, mc in enumerate(mask_caps):
            ret_m, mask_frame = mc.read()
            if ret_m and mask_frame is not None:
                if mask_frame.ndim == 3:
                    mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                binary = mask_frame > 127
                c = colours[i % len(colours)]
                for ch in range(3):
                    overlay[:, :, ch][binary] = (
                        (1 - alpha) * frame[:, :, ch][binary]
                        + alpha * c[ch]
                    ).astype(np.uint8)
        writer.write(overlay)

    for mc in mask_caps:
        mc.release()
    cap.release()
    writer.release()
    print(f"    Saved overlay video: {output_path.name}")


# ---------------------------------------------------------------------------
# Per-object tracking metadata
# ---------------------------------------------------------------------------

def _build_object_tracking(
    mask_dir: Path,
    object_ids: set,
    fps: float,
    frame_offset: int = 0,
    min_active_pixels: int = 50,
    gap_tolerance: int = 2,
) -> List[Dict[str, Any]]:
    """Scan stitched mask videos and compute per-object presence info.

    For each object, determines **all intervals** where the mask is active
    (has at least *min_active_pixels* bright pixels), handling objects that
    appear, disappear, and reappear multiple times.

    Short gaps of up to *gap_tolerance* inactive frames between two active
    regions are bridged so that tiny dips (e.g. from mp4 compression or a
    single missed frame) do not fragment intervals.

    Args:
        mask_dir: Directory containing ``object_{id}_mask.mp4`` files.
        object_ids: Set of global object IDs to scan.
        fps: Video FPS (used for timestamp conversion).
        frame_offset: If a sub-clip was extracted, offset added to frame
            numbers so they refer to the *original* video's timeline.
        min_active_pixels: Minimum number of pixels > 127 for a frame to
            be considered "active".  Default 50 (≈ noise-only filter).
        gap_tolerance: Maximum gap (in frames) between active frames that
            will be bridged into a single interval.  Default 2.

    Returns:
        List of dicts, one per object, sorted by object ID.
    """
    tracking: List[Dict[str, Any]] = []

    def _ts(sec: float) -> str:
        m, s = divmod(sec, 60)
        h, m = divmod(int(m), 60)
        return f"{h:02d}:{int(m):02d}:{s:06.3f}"

    for oid in sorted(object_ids):
        mp4 = mask_dir / f"object_{oid}_mask.mp4"
        if not mp4.exists():
            tracking.append({
                "object_id": oid,
                "intervals": [],
                "num_intervals": 0,
                "total_frames_active": 0,
                "total_frames": 0,
                "total_duration_s": 0.0,
                "first_frame": None,
                "last_frame": None,
                "first_timestamp": None,
                "last_timestamp": None,
                "first_timecode": None,
                "last_timecode": None,
                "mean_mask_area_pct": None,
                "max_mask_area_pct": None,
            })
            continue

        cap = cv2.VideoCapture(str(mp4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_pixels = max(w * h, 1)

        # Scan all frames — collect active frame indices and per-frame areas
        active_frames: List[int] = []
        area_fractions: List[float] = []          # only for active frames
        for fidx in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            active_px = int((frame > 127).sum())
            if active_px >= min_active_pixels:
                active_frames.append(fidx)
                area_fractions.append(active_px / total_pixels)
        cap.release()

        # ----- Group active frames into intervals with gap tolerance -----
        # Two active regions separated by <= gap_tolerance inactive frames
        # are merged into one interval.
        raw_intervals: List[tuple] = []           # (start, end) pairs
        if active_frames:
            iv_start = active_frames[0]
            iv_end = active_frames[0]
            for fidx in active_frames[1:]:
                if fidx <= iv_end + 1 + gap_tolerance:
                    iv_end = fidx         # extend (possibly bridging a gap)
                else:
                    raw_intervals.append((iv_start, iv_end))
                    iv_start = fidx
                    iv_end = fidx
            raw_intervals.append((iv_start, iv_end))

        intervals: List[Dict[str, Any]] = []
        for s, e in raw_intervals:
            abs_s = s + frame_offset
            abs_e = e + frame_offset
            intervals.append({
                "start_frame": abs_s,
                "end_frame": abs_e,
                "start_time": round(abs_s / fps, 3),
                "end_time": round(abs_e / fps, 3),
                "start_timecode": _ts(abs_s / fps),
                "end_timecode": _ts(abs_e / fps),
                "duration_frames": abs_e - abs_s + 1,
                "duration_s": round((abs_e - abs_s + 1) / fps, 3),
            })

        active_count = len(active_frames)
        abs_first = (active_frames[0] + frame_offset) if active_frames else None
        abs_last = (active_frames[-1] + frame_offset) if active_frames else None
        total_dur = round(active_count / fps, 3) if active_count else 0.0

        mean_area = round(sum(area_fractions) / len(area_fractions) * 100, 4) if area_fractions else None
        max_area = round(max(area_fractions) * 100, 4) if area_fractions else None

        entry: Dict[str, Any] = {
            "object_id": oid,
            "intervals": intervals,
            "num_intervals": len(intervals),
            "total_frames_active": active_count,
            "total_frames": total,
            "total_duration_s": total_dur,
            "first_frame": abs_first,
            "last_frame": abs_last,
            "first_timestamp": round(abs_first / fps, 3) if abs_first is not None else None,
            "last_timestamp": round(abs_last / fps, 3) if abs_last is not None else None,
            "first_timecode": _ts(abs_first / fps) if abs_first is not None else None,
            "last_timecode": _ts(abs_last / fps) if abs_last is not None else None,
            "mean_mask_area_pct": mean_area,
            "max_mask_area_pct": max_area,
        }

        tracking.append(entry)

    return tracking


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def _process_video(
    video_path: Path,
    prompts: Optional[List[str]],
    points: Optional[List[List[float]]],
    point_labels: Optional[List[int]],
    mask_paths: Optional[List[Path]],
    output_dir: Path,
    device: str,
    alpha: float,
    chunk_spread: str,
    keep_temp: bool,
    frame_range: Optional[Tuple[int, int]] = None,
    time_range: Optional[Tuple[str, str]] = None,
    max_vram_gb: Optional[float] = None,
    max_ram_gb: Optional[float] = None,
):
    """Full video processing pipeline with adaptive dynamic chunking."""
    from datetime import datetime
    import torch
    from sam3.utils.ffmpeglib import ffmpeg_lib
    from sam3.utils.helpers import sanitize_filename
    from sam3.__globals import TEMP_DIR, DEFAULT_MIN_CHUNK_OVERLAP
    from sam3.utils.memory_sampler import MemorySampler
    from sam3.memory_predictor import MemoryPredictor, StopLevel
    from sam3.memory_optimizer import clear_memory, AdaptiveChunkManager, IntraChunkMonitor
    from sam3.async_io import AsyncIOWorker

    # Convert simulated limits to bytes
    max_vram_bytes = int(max_vram_gb * 1024**3) if max_vram_gb else None
    max_ram_bytes = int(max_ram_gb * 1024**3) if max_ram_gb else None

    # ── Clear stale CUDA cache from any previous runs ──
    clear_memory(device, full_gc=True)

    # ── Timing & memory instrumentation ──
    pipeline_start = time.time()
    pipeline_start_iso = datetime.now().isoformat()
    mem_sampler = MemorySampler(interval=1.0, device=device)
    mem_sampler.start()

    # ── OOM predictor (async, non-blocking) ──
    _oom_stop_requested = False

    def _on_soft():
        nonlocal _oom_stop_requested
        print("\033[93m⚠ Memory warning: headroom < 20% — consider reducing chunk size\033[0m")

    def _on_hard():
        nonlocal _oom_stop_requested
        _oom_stop_requested = True
        print("\033[91m✗ Memory critical: headroom < 5% — requesting early stop\033[0m")

    mem_predictor = MemoryPredictor(
        device=device,
        safety_factor=0.85,
        on_soft_stop=_on_soft,
        on_hard_stop=_on_hard,
    )
    mem_predictor.start()
    frames_processed = 0  # running counter for record_frame

    video_name = video_path.stem

    # ----- Resolve range & extract sub-clip if needed -----
    resolved = _resolve_range(video_path, frame_range, time_range)
    original_video = video_path
    frame_offset = 0  # offset into the original video for metadata
    if resolved is not None:
        sf, ef = resolved
        frame_offset = sf
        print(f"Extracting segment: frames {sf}–{ef} ...")
        temp_subclip_dir = Path(TEMP_DIR) / video_name / "subclip"
        video_path = _extract_subclip(original_video, sf, ef, temp_subclip_dir)
        print(f"  Sub-clip: {video_path}  ({ef - sf + 1} frames)\n")

    # ----- Memory check -----
    # Use simulated VRAM limit if testing with a smaller GPU
    _mem_cap = max_vram_bytes if device == "cuda" else max_ram_bytes
    mem_info = _validate_video_memory(video_path, device, max_memory_bytes=_mem_cap)
    if _mem_cap:
        mem_info["simulated_limit_gb"] = round(_mem_cap / (1024**3), 1)
    _show_video_memory_table(mem_info)
    print()

    if not mem_info["can_process"]:
        deficit = mem_info.get("deficit_bytes", 0)
        print(f"\033[91m✗ Cannot process video — need {_fmt(deficit)} more memory.\033[0m")
        sys.exit(1)

    # ----- Chunk plan -----
    print("Creating chunk plan...")
    video_metadata, chunk_list = _make_chunk_plan(
        video_path, device, chunk_spread, max_memory_bytes=_mem_cap,
    )
    initial_chunk_size = mem_info.get("max_frames_per_chunk", 200)
    n_chunks = len(chunk_list)
    print(f"  {n_chunks} chunk(s), {initial_chunk_size} frames/chunk")

    # ----- Adaptive chunk manager -----
    adaptive = AdaptiveChunkManager(
        initial_chunk_size=initial_chunk_size,
        device=device,
        vram_limit_bytes=max_vram_bytes,
        ram_limit_bytes=max_ram_bytes,
    )
    if _mem_cap:
        print(f"  Simulated memory limit: {_mem_cap / (1024**3):.1f} GB")
    print()

    # ----- Directories -----
    temp_base = Path(TEMP_DIR) / video_name
    chunks_dir = temp_base / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    video_output = output_dir / video_name
    video_output.mkdir(parents=True, exist_ok=True)

    # Save video metadata
    meta_dir = video_output / "metadata"
    meta_dir.mkdir(exist_ok=True)
    with open(meta_dir / "video_metadata.json", "w") as f:
        json.dump(video_metadata, f, indent=2)
    with open(meta_dir / "memory_info.json", "w") as f:
        json.dump(mem_info, f, indent=2, default=str)

    # ----- Load model once -----
    print("Loading SAM3 video model...")
    t_model_start = time.time()
    from sam3.drivers import Sam3VideoDriver
    driver = Sam3VideoDriver(device=device)
    model_load_s = round(time.time() - t_model_start, 3)
    print(f"Model loaded in {model_load_s:.1f}s.\n")
    mem_predictor.record_frame(0)  # baseline after model load

    # ----- Async I/O worker for overlapping writes -----
    async_worker = AsyncIOWorker()
    async_worker.start()

    # ----- Validate mask dimensions (if masks provided) -----
    if mask_paths:
        vid_info = ffmpeg_lib.get_video_info(str(video_path))
        for mp in mask_paths:
            m_img = Image.open(mp)
            if m_img.size != (vid_info["width"], vid_info["height"]):
                print(
                    f"\033[91m✗ Mask {mp.name} size {m_img.size} does not match "
                    f"video {vid_info['width']}x{vid_info['height']}\033[0m"
                )
                sys.exit(1)

    # ----- Process chunks -----
    overlap = DEFAULT_MIN_CHUNK_OVERLAP
    # Carry-forward state across chunks (per prompt)
    carry: Dict[str, Dict[int, np.ndarray]] = {}
    global_next_ids: Dict[str, int] = {}
    all_object_ids: Dict[str, set] = {}

    # ── Collectors for enriched metadata ──
    chunk_timing: List[Dict[str, Any]] = []
    chunk_meta_list: List[Dict[str, Any]] = []
    cross_chunk_iou: Dict[str, Dict[str, Any]] = {}

    total_frames_in_video = video_metadata.get("nb_frames", 0)

    t_chunks_start = time.time()

    # ── Dynamic chunk loop with OOM recovery ──
    # Instead of iterating a fixed list, we maintain a cursor and can
    # replan remaining chunks after each chunk completes (or on OOM).
    ci = 0  # chunk counter
    chunk_cursor = 0  # index into chunk_list

    while chunk_cursor < len(chunk_list):
        # ── OOM predictor: check if we should bail ──
        if _oom_stop_requested:
            print(f"\033[91m✗ Stopping after chunk {ci} — OOM predictor triggered hard stop\033[0m")
            break

        cinfo = chunk_list[chunk_cursor]
        chunk_id = ci
        start_frame = cinfo["start"]
        end_frame = cinfo["end"]
        current_chunk_frames = end_frame - start_frame + 1
        t_chunk_start = time.time()

        print(f"── Chunk {ci + 1}/{len(chunk_list)} (frames {start_frame}–{end_frame}, size={current_chunk_frames}) ──")

        chunk_dir = chunks_dir / f"chunk_{chunk_id}"
        chunk_dir.mkdir(exist_ok=True)

        # Extract chunk video
        if len(chunk_list) == 1 and chunk_cursor == 0:
            chunk_video = video_path  # use original
        else:
            chunk_video = chunk_dir / f"chunk_{chunk_id}.mp4"
            ffmpeg_lib.create_video_chunk(
                str(video_path), str(chunk_video), start_frame, end_frame
            )

        # Get chunk frame count
        cap = cv2.VideoCapture(str(chunk_video))
        chunk_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Reset peak memory stats for this chunk
        if device.startswith("cuda"):
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.reset_peak_memory_stats()

        # Start session
        session_id = driver.start_session(video_path=str(chunk_video))

        chunk_objects: Dict[str, Any] = {}  # per-prompt objects for this chunk
        chunk_n_objects = 0  # total objects across all prompts
        _chunk_oom = False  # flag for OOM during this chunk
        _chunk_early_stop = False  # flag for proactive early stop
        _early_stop_monitor = None  # monitor that triggered early stop
        _chunk_monitor_results: List[Dict] = []  # monitor metadata per prompt

        try:
            # ----- Text prompts -----
            if prompts:
                for prompt in prompts:
                    safe = sanitize_filename(prompt)
                    print(f"  Prompt: '{prompt}'")

                    driver.reset_session(session_id)
                    driver.add_prompt(session_id, prompt)

                    # ── Monitored streaming propagation ──
                    expected_iters = chunk_frames * 2  # "both" → forward + backward
                    monitor = IntraChunkMonitor(
                        expected_iterations=expected_iters,
                        device=device,
                        vram_limit_bytes=max_vram_bytes,
                    )
                    monitor.start()

                    result, obj_ids, frame_objs, early_stopped = _propagate_with_monitoring(
                        driver, session_id, monitor, propagation_direction="both"
                    )
                    _chunk_monitor_results.append(monitor.to_dict())

                    if early_stopped:
                        mon_result = monitor.finalize()
                        if mon_result.stop_reason == "oom_exception":
                            _chunk_oom = True
                            print(f"\033[91m    ✗ CUDA OOM during '{prompt}' propagation\033[0m")
                        else:
                            _chunk_early_stop = True
                            _early_stop_monitor = monitor
                            print(f"\033[93m    ⚠ Proactive stop during '{prompt}': "
                                  f"{mon_result.stop_reason} at iter {mon_result.iterations_completed}/{expected_iters}\033[0m")
                        clear_memory(device, full_gc=True)
                        break

                    frames_processed += chunk_frames
                    mem_predictor.record_frame(frames_processed)

                    prev_masks = carry.get(prompt, {})
                    gnid = global_next_ids.get(prompt, 0)

                    result, obj_ids, mapping, gnid, iou_mat = _match_and_remap(
                        result, obj_ids, prev_masks, gnid
                    )
                    global_next_ids[prompt] = gnid

                    # Store IoU matrix for cross-chunk metadata
                    if iou_mat and ci > 0:
                        key = f"chunk_{ci-1}_to_{ci}"
                        cross_chunk_iou.setdefault(key, {})[prompt] = {
                            "matrix": {str(k): {str(pk): v for pk, v in row.items()} for k, row in iou_mat.items()},
                            "matched": {str(k): v for k, v in mapping.items()},
                            "threshold": 0.25,
                        }

                    # Track all IDs
                    all_object_ids.setdefault(prompt, set()).update(obj_ids)
                    chunk_n_objects += len(obj_ids)

                    chunk_objects[prompt] = {
                        "object_ids": sorted(obj_ids),
                        "num_objects": len(obj_ids),
                        "id_mapping": {str(k): v for k, v in mapping.items()},
                    }

                    # Save masks (async — overlap I/O with next prompt's GPU work)
                    masks_dir = chunk_dir / "masks" / safe
                    masks_dir.mkdir(parents=True, exist_ok=True)
                    _ensure_cpu_masks(result)
                    async_worker.submit(
                        _save_chunk_masks,
                        result, obj_ids, masks_dir,
                        video_metadata["width"], video_metadata["height"],
                        chunk_frames,
                    )

                    # Extract carry-forward
                    carry[prompt] = _extract_last_frame_masks(result, obj_ids)
                    print(f"    {len(obj_ids)} object(s)")

                    # Free result tensors so GPU memory is available for next prompt
                    del result, obj_ids, frame_objs, mapping, iou_mat
                    clear_memory(device, full_gc=False)

            # ----- Point prompts -----
            if points and not _chunk_oom and not _chunk_early_stop:
                prompt_key = "__points__"
                safe = "points"
                vid_info = video_metadata
                print(f"  Points: {points}")

                driver.reset_session(session_id)

                for pi, (pt, lbl) in enumerate(zip(points, point_labels)):
                    driver.add_object_with_points_prompt(
                        session_id,
                        frame_idx=0,
                        object_id=pi,
                        frame_width=vid_info["width"],
                        frame_height=vid_info["height"],
                        points=[pt],
                        point_labels=[lbl],
                    )

                # ── Monitored streaming propagation ──
                expected_iters = chunk_frames * 2
                monitor = IntraChunkMonitor(
                    expected_iterations=expected_iters,
                    device=device,
                    vram_limit_bytes=max_vram_bytes,
                )
                monitor.start()

                result, obj_ids, frame_objs, early_stopped = _propagate_with_monitoring(
                    driver, session_id, monitor, propagation_direction="both"
                )
                _chunk_monitor_results.append(monitor.to_dict())

                if early_stopped:
                    mon_result = monitor.finalize()
                    if mon_result.stop_reason == "oom_exception":
                        _chunk_oom = True
                        print(f"\033[91m    ✗ CUDA OOM during points propagation\033[0m")
                    else:
                        _chunk_early_stop = True
                        _early_stop_monitor = monitor
                        print(f"\033[93m    ⚠ Proactive stop during points: "
                              f"{mon_result.stop_reason} at iter {mon_result.iterations_completed}/{expected_iters}\033[0m")
                    clear_memory(device, full_gc=True)

                if not _chunk_oom and not _chunk_early_stop:
                    frames_processed += chunk_frames
                    mem_predictor.record_frame(frames_processed)

                    prev_masks = carry.get(prompt_key, {})
                    gnid = global_next_ids.get(prompt_key, 0)
                    result, obj_ids, mapping, gnid, iou_mat = _match_and_remap(
                        result, obj_ids, prev_masks, gnid
                    )
                    global_next_ids[prompt_key] = gnid
                    all_object_ids.setdefault(prompt_key, set()).update(obj_ids)
                    chunk_n_objects += len(obj_ids)

                    if iou_mat and ci > 0:
                        key = f"chunk_{ci-1}_to_{ci}"
                        cross_chunk_iou.setdefault(key, {})[prompt_key] = {
                            "matrix": {str(k): {str(pk): v for pk, v in row.items()} for k, row in iou_mat.items()},
                            "matched": {str(k): v for k, v in mapping.items()},
                            "threshold": 0.25,
                        }

                    chunk_objects[prompt_key] = {
                        "object_ids": sorted(obj_ids),
                        "num_objects": len(obj_ids),
                        "id_mapping": {str(k): v for k, v in mapping.items()},
                    }

                    masks_dir = chunk_dir / "masks" / safe
                    masks_dir.mkdir(parents=True, exist_ok=True)
                    _ensure_cpu_masks(result)
                    async_worker.submit(
                        _save_chunk_masks,
                        result, obj_ids, masks_dir,
                        vid_info["width"], vid_info["height"],
                        chunk_frames,
                    )
                    carry[prompt_key] = _extract_last_frame_masks(result, obj_ids)
                    print(f"    {len(obj_ids)} object(s)")

                    # Free result tensors so GPU memory is available for next prompt
                    del result, obj_ids, frame_objs, mapping, iou_mat
                    clear_memory(device, full_gc=False)

            # ----- Mask prompts -----
            if mask_paths and not _chunk_oom and not _chunk_early_stop:
                prompt_key = "__masks__"
                safe = "masks"
                vid_info = video_metadata
                print(f"  Masks: {[m.name for m in mask_paths]}")

                driver.reset_session(session_id)

                # Inject each mask as a separate object
                mask_dict = {}
                obj_id_list = []
                for mi, mp in enumerate(mask_paths):
                    m = np.array(Image.open(mp).convert("L"))
                    mask_dict[mi] = m
                    obj_id_list.append(mi)

                driver.inject_masks(session_id, frame_idx=0, masks=mask_dict, object_ids=obj_id_list)

                # ── Monitored streaming propagation ──
                expected_iters = chunk_frames * 2
                monitor = IntraChunkMonitor(
                    expected_iterations=expected_iters,
                    device=device,
                    vram_limit_bytes=max_vram_bytes,
                )
                monitor.start()

                result, obj_ids, frame_objs, early_stopped = _propagate_with_monitoring(
                    driver, session_id, monitor, propagation_direction="both"
                )
                _chunk_monitor_results.append(monitor.to_dict())

                if early_stopped:
                    mon_result = monitor.finalize()
                    if mon_result.stop_reason == "oom_exception":
                        _chunk_oom = True
                        print(f"\033[91m    ✗ CUDA OOM during mask propagation\033[0m")
                    else:
                        _chunk_early_stop = True
                        _early_stop_monitor = monitor
                        print(f"\033[93m    ⚠ Proactive stop during masks: "
                              f"{mon_result.stop_reason} at iter {mon_result.iterations_completed}/{expected_iters}\033[0m")
                    clear_memory(device, full_gc=True)

                if not _chunk_oom and not _chunk_early_stop:
                    frames_processed += chunk_frames
                    mem_predictor.record_frame(frames_processed)

                    prev_masks_cf = carry.get(prompt_key, {})
                    gnid = global_next_ids.get(prompt_key, 0)
                    result, obj_ids, mapping, gnid, iou_mat = _match_and_remap(
                        result, obj_ids, prev_masks_cf, gnid
                    )
                    global_next_ids[prompt_key] = gnid
                    all_object_ids.setdefault(prompt_key, set()).update(obj_ids)
                    chunk_n_objects += len(obj_ids)

                    if iou_mat and ci > 0:
                        key = f"chunk_{ci-1}_to_{ci}"
                        cross_chunk_iou.setdefault(key, {})[prompt_key] = {
                            "matrix": {str(k): {str(pk): v for pk, v in row.items()} for k, row in iou_mat.items()},
                            "matched": {str(k): v for k, v in mapping.items()},
                            "threshold": 0.25,
                        }

                    chunk_objects[prompt_key] = {
                        "object_ids": sorted(obj_ids),
                        "num_objects": len(obj_ids),
                        "id_mapping": {str(k): v for k, v in mapping.items()},
                    }

                    masks_dir = chunk_dir / "masks" / safe
                    masks_dir.mkdir(parents=True, exist_ok=True)
                    _ensure_cpu_masks(result)
                    async_worker.submit(
                        _save_chunk_masks,
                        result, obj_ids, masks_dir,
                        vid_info["width"], vid_info["height"],
                        chunk_frames,
                    )
                    carry[prompt_key] = _extract_last_frame_masks(result, obj_ids)
                    print(f"    {len(obj_ids)} object(s)")

                    # Free result tensors
                    del result, obj_ids, frame_objs, mapping, iou_mat
                    clear_memory(device, full_gc=False)

        finally:
            driver.close_session(session_id)

        # ── Measure peak memory for this chunk ──
        peak_vram = 0
        peak_ram = 0
        if device.startswith("cuda"):
            import torch as _t
            if _t.cuda.is_available():
                peak_vram = _t.cuda.max_memory_allocated()
        try:
            peak_ram = psutil.Process().memory_info().rss
        except Exception:
            pass

        # ── Handle OOM: rechunk remaining frames with smaller chunks ──
        if _chunk_oom:
            print(f"\033[93m  ⚠ OOM on chunk {ci + 1} (size={current_chunk_frames}). "
                  f"Reducing chunk size and replanning...\033[0m")
            try:
                new_size = adaptive.handle_oom(chunk_id, current_chunk_frames)
            except RuntimeError as exc:
                print(f"\033[91m  ✗ {exc}\033[0m")
                break
            # Replan from this chunk's start_frame with reduced size
            adaptive.current_chunk_size = new_size
            new_chunks = adaptive.replan_remaining(
                start_frame, total_frames_in_video, overlap
            )
            if not new_chunks:
                print(f"\033[91m  ✗ No viable chunks after OOM reduction\033[0m")
                break
            # Replace remaining chunk_list
            chunk_list = chunk_list[:chunk_cursor] + new_chunks
            n_chunks = len(chunk_list)
            print(f"  Replanned: {len(new_chunks)} chunk(s) remaining, "
                  f"new chunk size = {new_size} frames")
            # Don't advance chunk_cursor — retry from same position
            ci += 1  # but increment chunk counter for directory naming
            continue

        # ── Handle proactive early stop: use calibration for smart replan ──
        if _chunk_early_stop and _early_stop_monitor is not None:
            mon_result = _early_stop_monitor.finalize()
            cal = mon_result.calibration

            # Use calibration to pick a safe chunk size
            if cal and cal.safe_iterations > 0:
                # safe_iterations is for total iterations; "both" = 2× frames
                safe_frames = cal.safe_iterations // 2
                safe_frames = int(safe_frames * 0.8)  # 80% safety margin
            else:
                safe_frames = current_chunk_frames // 2  # fallback: halve

            safe_frames = max(safe_frames, adaptive.min_chunk_frames)

            print(f"\033[93m  ⚠ Proactive stop on chunk {ci + 1}: {mon_result.stop_reason}. "
                  f"Calibration → {safe_frames} frames/chunk (was {current_chunk_frames})\033[0m")

            adaptive.current_chunk_size = safe_frames
            adaptive.rechunk_events.append({
                "chunk_id": chunk_id,
                "from_size": current_chunk_frames,
                "to_size": safe_frames,
                "reason": f"proactive_{mon_result.stop_reason}",
                "calibration": {
                    "growth_rate_mb": round(cal.growth_rate_per_iter / (1024**2), 3) if cal else None,
                    "safe_iters": cal.safe_iterations if cal else None,
                    "confidence": cal.confidence if cal else None,
                },
            })

            # Replan from this chunk's start_frame with calibrated size
            new_chunks = adaptive.replan_remaining(
                start_frame, total_frames_in_video, overlap
            )
            if not new_chunks:
                print(f"\033[91m  ✗ No viable chunks after proactive resizing\033[0m")
                break
            chunk_list = chunk_list[:chunk_cursor] + new_chunks
            n_chunks = len(chunk_list)
            print(f"  Replanned: {len(new_chunks)} chunk(s) remaining, "
                  f"new chunk size = {safe_frames} frames")
            ci += 1
            continue

        # ── Adaptive sizing: evaluate pressure and adjust for next chunk ──
        rec = adaptive.record_chunk(
            chunk_id=chunk_id,
            chunk_size=current_chunk_frames,
            peak_vram_bytes=peak_vram,
            peak_ram_bytes=peak_ram,
            n_objects=chunk_n_objects,
        )

        if rec.action == "SHRINK":
            print(f"\033[93m  ⚠ Memory pressure {rec.pressure} "
                  f"({rec.vram_usage_pct:.0f}% of limit). "
                  f"Reducing next chunk: {current_chunk_frames} → {rec.adjusted_chunk_size} frames\033[0m")
            # Replan remaining chunks with smaller size
            next_start = end_frame + 1 - overlap
            if next_start < total_frames_in_video:
                new_chunks = adaptive.replan_remaining(
                    next_start, total_frames_in_video, overlap
                )
                chunk_list = chunk_list[:chunk_cursor + 1] + new_chunks
                n_chunks = len(chunk_list)
                print(f"  Replanned: {len(new_chunks)} chunk(s) remaining")
        elif rec.action == "GROW":
            print(f"\033[92m  ↑ Under-utilised ({rec.vram_usage_pct:.0f}% of limit). "
                  f"Growing next chunk: {current_chunk_frames} → {rec.adjusted_chunk_size} frames\033[0m")
            next_start = end_frame + 1 - overlap
            if next_start < total_frames_in_video:
                new_chunks = adaptive.replan_remaining(
                    next_start, total_frames_in_video, overlap
                )
                chunk_list = chunk_list[:chunk_cursor + 1] + new_chunks
                n_chunks = len(chunk_list)

        chunk_dur = round(time.time() - t_chunk_start, 3)
        chunk_timing.append({
            "chunk_id": chunk_id,
            "frames": chunk_frames,
            "duration_s": chunk_dur,
            "s_per_frame": round(chunk_dur / max(chunk_frames, 1), 3),
            "peak_vram_mb": round(peak_vram / (1024**2), 1),
            "peak_ram_mb": round(peak_ram / (1024**2), 1),
            "n_objects": chunk_n_objects,
            "pressure": rec.pressure,
            "action": rec.action,
            "intra_chunk_monitors": _chunk_monitor_results,
        })
        chunk_meta_list.append({
            "chunk_id": chunk_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_frame_original": start_frame + frame_offset,
            "end_frame_original": end_frame + frame_offset,
            "total_frames": chunk_frames,
            "processing_time_s": chunk_dur,
            "peak_vram_mb": round(peak_vram / (1024**2), 1),
            "peak_ram_mb": round(peak_ram / (1024**2), 1),
            "n_objects": chunk_n_objects,
            "memory_pressure": rec.pressure,
            "adaptive_action": rec.action,
            "next_chunk_size": rec.adjusted_chunk_size,
            "objects_by_prompt": chunk_objects,
            "intra_chunk_monitors": _chunk_monitor_results,
        })
        print(f"  Chunk {ci + 1} done in {chunk_dur:.1f}s "
              f"({chunk_dur / max(chunk_frames, 1):.2f} s/frame, "
              f"peak VRAM: {peak_vram / (1024**2):.0f} MB, "
              f"pressure: {rec.pressure})\n")

        chunk_cursor += 1
        ci += 1

    chunk_processing_s = round(time.time() - t_chunks_start, 3)

    # ----- Drain async I/O before stitching -----
    io_errors = async_worker.drain()
    if io_errors:
        print(f"\033[93m  ⚠ {io_errors} async I/O error(s) during mask writing\033[0m")

    # ----- Stitch masks and create overlay -----
    print("Stitching masks...")
    t_stitch_start = time.time()
    fps = video_metadata.get("fps", 25)
    w = video_metadata["width"]
    h = video_metadata["height"]

    prompt_keys = []
    if prompts:
        prompt_keys.extend(prompts)
    if points:
        prompt_keys.append("__points__")
    if mask_paths:
        prompt_keys.append("__masks__")

    for pk in prompt_keys:
        safe = sanitize_filename(pk) if not pk.startswith("__") else pk.strip("_")
        oids = all_object_ids.get(pk, set())
        out = video_output / "masks" / safe
        _stitch_masks_to_video(
            chunks_dir, safe, oids, chunk_list, overlap, out, fps, w, h
        )

        # Overlay
        mask_vids = sorted(out.glob("object_*_mask.mp4"))
        if mask_vids:
            overlay_path = video_output / f"overlay_{safe}.mp4"
            _create_overlay_video(video_path, mask_vids, overlay_path, alpha)

    stitching_s = round(time.time() - t_stitch_start, 3)

    # ----- Per-object temporal analysis -----
    print("Analysing object presence...")
    t_analysis_start = time.time()
    objects_tracking: Dict[str, List[Dict[str, Any]]] = {}
    for pk in prompt_keys:
        safe = sanitize_filename(pk) if not pk.startswith("__") else pk.strip("_")
        oids = all_object_ids.get(pk, set())
        mask_out = video_output / "masks" / safe
        if oids and mask_out.exists():
            objects_tracking[pk] = _build_object_tracking(
                mask_out, oids, fps, frame_offset
            )

    analysis_s = round(time.time() - t_analysis_start, 3)

    # ----- Cleanup -----
    driver.cleanup()
    async_worker.shutdown()

    if keep_temp:
        dest = video_output / "temp_files"
        if temp_base.exists():
            shutil.copytree(temp_base, dest, dirs_exist_ok=True)
            print(f"  Temp files preserved at: {dest}")

    if temp_base.exists():
        shutil.rmtree(temp_base)

    # ── Stop memory sampler & predictor ──
    mem_sampler.stop()
    mem_summary = mem_sampler.summary()
    mem_predictor.stop()
    predictor_summary = mem_predictor.summary()

    # ── Timing summary ──
    pipeline_end = time.time()
    pipeline_end_iso = datetime.now().isoformat()
    total_s = round(pipeline_end - pipeline_start, 3)

    # ── Thread config ──
    thread_config = {
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
    }

    # ── Read SAM3 version ──
    sam3_version = "unknown"
    try:
        ver_file = Path(__file__).resolve().parent / "VERSION"
        if ver_file.exists():
            sam3_version = ver_file.read_text().strip()
    except Exception:
        pass

    # ----- Save enriched final metadata -----
    adaptive_summary = adaptive.to_dict()
    async_io_summary = async_worker.to_dict()

    final_meta = {
        "schema_version": "2.2.0",
        "sam3_version": sam3_version,
        "video": str(original_video),
        "video_name": video_name,
        "output_dir": str(video_output),
        "resolution": f"{w}x{h}",
        "total_frames": video_metadata.get("nb_frames"),
        "fps": fps,
        "duration_s": video_metadata.get("duration"),
        "frame_offset": frame_offset,
        "frame_range": list(frame_range) if frame_range else None,
        "time_range": list(time_range) if time_range else None,
        "device": device,
        "thread_config": thread_config,
        "prompts": prompts,
        "points": points,
        "mask_paths": [str(p) for p in mask_paths] if mask_paths else None,
        "num_chunks": n_chunks,
        "num_chunks_processed": ci,
        "overlap_frames": overlap,
        "simulated_limits": {
            "max_vram_gb": max_vram_gb,
            "max_ram_gb": max_ram_gb,
        } if (max_vram_gb or max_ram_gb) else None,
        "timing": {
            "pipeline_start": pipeline_start_iso,
            "pipeline_end": pipeline_end_iso,
            "total_s": total_s,
            "model_load_s": model_load_s,
            "chunk_processing_s": chunk_processing_s,
            "stitching_s": stitching_s,
            "analysis_s": analysis_s,
            "per_chunk": chunk_timing,
        },
        "memory": {
            "pre_run": mem_info,
            **mem_summary,
        },
        "adaptive_chunking": adaptive_summary,
        "async_io": async_io_summary,
        "chunks": chunk_meta_list,
        "cross_chunk_iou": cross_chunk_iou if cross_chunk_iou else None,
        "memory_predictor": predictor_summary,
        "objects": objects_tracking,
    }
    with open(video_output / "metadata.json", "w") as f:
        json.dump(final_meta, f, indent=2, default=str)

    # Also save the detailed cross-chunk IoU separately for evaluation
    if cross_chunk_iou:
        with open(meta_dir / "cross_chunk_iou.json", "w") as f:
            json.dump(cross_chunk_iou, f, indent=2, default=str)

    # Also save object tracking separately for easy access
    if objects_tracking:
        with open(meta_dir / "object_tracking.json", "w") as f:
            json.dump(objects_tracking, f, indent=2, default=str)

    # Save memory predictor data separately for analysis
    with open(meta_dir / "memory_predictor.json", "w") as f:
        json.dump(predictor_summary, f, indent=2, default=str)

    # Save adaptive chunking data separately for analysis
    with open(meta_dir / "adaptive_chunking.json", "w") as f:
        json.dump(adaptive_summary, f, indent=2, default=str)

    print()
    print("=" * 70)
    print(f"  ✓ Video processing complete → {video_output}")
    print(f"    Total: {total_s:.1f}s  (model: {model_load_s:.1f}s, chunks: {chunk_processing_s:.1f}s, stitch: {stitching_s:.1f}s)")
    if adaptive.rechunk_events:
        print(f"    Adaptive rechunks: {len(adaptive.rechunk_events)}  "
              f"(initial: {adaptive.initial_chunk_size}, final: {adaptive.current_chunk_size} frames/chunk)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SAM3 Video Prompter — segment videos with text, click points, or masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_prompter.py --video clip.mp4 --prompts person ball
  python video_prompter.py --video clip.mp4 --points 320,240 --point-labels 1
  python video_prompter.py --video clip.mp4 --masks mask.png
  python video_prompter.py --video clip.mp4 --prompts player --device cpu --keep-temp
  python video_prompter.py --video clip.mp4 --prompts player --frame-range 100 500
  python video_prompter.py --video clip.mp4 --prompts player --time-range 0:05 0:30
  python video_prompter.py --video clip.mp4 --prompts player --time-range 10.0 45.5
        """,
    )

    parser.add_argument("--video", type=str, required=True, help="Input video file")
    parser.add_argument(
        "--prompts", nargs="+", default=None,
        help="Text prompts (e.g. person ball)",
    )
    parser.add_argument(
        "--points", nargs="+", default=None,
        help="Click points as x,y pairs (e.g. 320,240 500,300)",
    )
    parser.add_argument(
        "--point-labels", nargs="+", type=int, default=None,
        help="Labels for each point (1=positive, 0=negative)",
    )
    parser.add_argument(
        "--masks", nargs="+", default=None,
        help="Mask image file(s) for initial object prompts",
    )
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Overlay alpha (0.0–1.0, default: 0.5)",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda"],
        help="Force device (auto-detected if omitted)",
    )
    parser.add_argument(
        "--chunk-spread", type=str, default="default", choices=["default", "even"],
        help="Chunk size strategy (default or even)",
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="Preserve intermediate chunk files in output",
    )
    parser.add_argument(
        "--frame-range", nargs=2, type=int, metavar=("START", "END"),
        help="Process only frames START..END (0-based, inclusive)",
    )
    parser.add_argument(
        "--time-range", nargs=2, type=str, metavar=("START", "END"),
        help="Process a time segment (seconds, MM:SS, or HH:MM:SS)",
    )
    parser.add_argument(
        "--max-vram-gb", type=float, default=None,
        help="Simulate a smaller GPU by capping VRAM (GB). For testing adaptive chunking.",
    )
    parser.add_argument(
        "--max-ram-gb", type=float, default=None,
        help="Simulate a smaller system by capping RAM (GB). For testing adaptive chunking.",
    )

    args = parser.parse_args()

    # Validate: frame-range and time-range are mutually exclusive
    if args.frame_range and args.time_range:
        print("\033[91m✗ --frame-range and --time-range are mutually exclusive.\033[0m")
        sys.exit(1)

    # Validate: at least one prompt type
    if not args.prompts and not args.points and not args.masks:
        print("\033[91m✗ At least one of --prompts, --points, or --masks must be provided.\033[0m")
        sys.exit(1)

    # Validate video exists
    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"\033[91m✗ Video file not found: {args.video}\033[0m")
        sys.exit(1)

    # Parse points
    points = None
    point_labels = None
    if args.points:
        points = []
        for p in args.points:
            parts = p.replace(" ", "").split(",")
            if len(parts) != 2:
                print(f"\033[91m✗ Each point must be x,y — got '{p}'\033[0m")
                sys.exit(1)
            points.append([float(parts[0]), float(parts[1])])
        point_labels = args.point_labels or [1] * len(points)
        if len(point_labels) != len(points):
            print(f"\033[91m✗ --point-labels count must match --points count\033[0m")
            sys.exit(1)

    # Parse masks
    mask_paths = None
    if args.masks:
        mask_paths = [Path(m) for m in args.masks]
        for mp in mask_paths:
            if not mp.is_file():
                print(f"\033[91m✗ Mask file not found: {mp}\033[0m")
                sys.exit(1)

    # Device
    from sam3.__globals import DEVICE as DEFAULT_DEVICE
    device = args.device or DEFAULT_DEVICE.type

    # Header
    print()
    print("=" * 70)
    print("  SAM3 Video Prompter")
    print("=" * 70)
    print(f"  Video   : {video_path}")
    if args.prompts:
        print(f"  Prompts : {', '.join(args.prompts)}")
    if points:
        print(f"  Points  : {points}")
    if mask_paths:
        print(f"  Masks   : {[m.name for m in mask_paths]}")
    if args.frame_range:
        print(f"  Frames  : {args.frame_range[0]}–{args.frame_range[1]}")
    if args.time_range:
        print(f"  Time    : {args.time_range[0]} → {args.time_range[1]}")
    print(f"  Device  : {device}")
    print(f"  Output  : {args.output}")
    print(f"  Alpha   : {args.alpha}")
    print(f"  Chunking: {args.chunk_spread}")
    if args.max_vram_gb:
        print(f"  Max VRAM: {args.max_vram_gb} GB (simulated)")
    if args.max_ram_gb:
        print(f"  Max RAM : {args.max_ram_gb} GB (simulated)")
    print("=" * 70)
    print()

    _process_video(
        video_path=video_path,
        prompts=args.prompts,
        points=points,
        point_labels=point_labels,
        mask_paths=mask_paths,
        output_dir=Path(args.output),
        device=device,
        alpha=args.alpha,
        chunk_spread=args.chunk_spread,
        keep_temp=args.keep_temp,
        frame_range=tuple(args.frame_range) if args.frame_range else None,
        time_range=tuple(args.time_range) if args.time_range else None,
        max_vram_gb=args.max_vram_gb,
        max_ram_gb=args.max_ram_gb,
    )


if __name__ == "__main__":
    main()
