"""Memory optimisation utilities for SAM3 inference.

This module provides a unified, device-agnostic toolkit for:
  - Querying available memory (GPU via ``torch.cuda``, CPU via ``psutil``)
  - Clearing stale allocator caches (CUDA empty_cache / glibc malloc_trim)
  - Estimating per-frame GPU cost for video segmentation
  - Calibrating per-frame cost from a small probe run
  - Context managers that enforce ``torch.inference_mode`` + cleanup

Design principles
-----------------
- **Zero external dependencies** beyond PyTorch and psutil (no pynvml / nvidia-smi).
- **Modular** — every function is independently usable.
- **Device-agnostic** — same API for ``cuda`` and ``cpu``.
"""

from __future__ import annotations

import gc
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import psutil

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Empirically measured overhead multiplier for SAM3 model state per frame.
# raw pixel bytes (width*height*3) capture only the decoded video frame;
# the model also stores per-frame feature maps, cached masks for N tracked
# objects, positional encodings, attention KV caches, and memory-bank
# features.  The multiplier accounts for this additional state.
#
# Calibrated on an A100-80GB with SAM3 video inference:
#   480p  (~410k pixels): measured ~40 MB/frame → multiplier ≈ 4.3×
#   1080p (~2.07M pixels): measured ~70 MB/frame → multiplier ≈ 4.5×
MODEL_STATE_MULTIPLIER: float = 4.5

# Fixed per-frame overhead (bytes) independent of resolution.
# The SAM3 model stores memory-bank features, attention KV caches, and
# positional encodings at *model* internal resolution (1024×1024 typically),
# so there is a substantial baseline cost per frame even for small videos.
# Calibrated: 480p measured ~40 MB/frame, of which ~8 MB is resolution-
# dependent → ~32 MB is fixed model state.
MODEL_FIXED_PER_FRAME_BYTES: int = 32 * 1024 * 1024  # 32 MB

# Minimum bytes per frame to prevent unrealistically small estimates for
# low-resolution videos (e.g. tiny thumbnails).
MIN_PER_FRAME_BYTES: int = 8 * 1024 * 1024  # 8 MB


# ---------------------------------------------------------------------------
# Dataclass for memory snapshot
# ---------------------------------------------------------------------------

@dataclass
class MemorySnapshot:
    """Device memory snapshot at a point in time."""
    total: int = 0
    used: int = 0
    free: int = 0
    percent_used: float = 0.0
    device: str = "cpu"
    source: str = "unknown"  # "torch.cuda" | "psutil" | "nvidia-smi"


# ---------------------------------------------------------------------------
# Core memory queries (no subprocess, no pynvml)
# ---------------------------------------------------------------------------

def get_gpu_memory(device_index: int = 0) -> MemorySnapshot:
    """Query GPU VRAM using ``torch.cuda.mem_get_info`` (zero-overhead).

    Falls back gracefully if CUDA is unavailable.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return MemorySnapshot(device=f"cuda:{device_index}", source="unavailable")
        free, total = torch.cuda.mem_get_info(device_index)
        used = total - free
        return MemorySnapshot(
            total=total,
            used=used,
            free=free,
            percent_used=round((used / total) * 100, 2) if total else 0,
            device=f"cuda:{device_index}",
            source="torch.cuda",
        )
    except Exception:
        return MemorySnapshot(device=f"cuda:{device_index}", source="error")


def get_cpu_memory() -> MemorySnapshot:
    """Query system RAM using ``psutil`` (lightweight)."""
    mem = psutil.virtual_memory()
    return MemorySnapshot(
        total=mem.total,
        used=mem.used,
        free=mem.available,
        percent_used=mem.percent,
        device="cpu",
        source="psutil",
    )


def get_memory(device: str = "cuda") -> MemorySnapshot:
    """Unified memory query — dispatches to GPU or CPU based on *device*."""
    if device.startswith("cuda"):
        idx = 0
        if ":" in device:
            try:
                idx = int(device.split(":")[1])
            except ValueError:
                idx = 0
        return get_gpu_memory(idx)
    return get_cpu_memory()


# ---------------------------------------------------------------------------
# Cache clearing
# ---------------------------------------------------------------------------

def clear_memory(device: str = "cuda", *, full_gc: bool = True) -> None:
    """Aggressively release cached memory.

    For CUDA: calls ``torch.cuda.empty_cache()`` to return reserved memory
    to the CUDA driver so it becomes visible as *free*.

    For CPU (Linux): calls ``malloc_trim`` + Python GC.

    Parameters
    ----------
    device : str
        ``"cuda"`` or ``"cpu"``.
    full_gc : bool
        If *True* (default), also run ``gc.collect()`` to release Python
        objects that may be holding tensor references.
    """
    if full_gc:
        gc.collect()

    if device.startswith("cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    else:
        # Linux-specific: return freed memory to OS
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Per-frame memory estimation
# ---------------------------------------------------------------------------

def estimate_per_frame_bytes(
    width: int,
    height: int,
    device: str = "cuda",
    *,
    multiplier: Optional[float] = None,
) -> int:
    """Estimate the *total* GPU/RAM cost per video frame during SAM3 inference.

    This is much larger than ``width × height × 3`` because the model stores
    per-frame feature maps, cached masks (for all tracked objects), memory
    bank tensors, and attention state in its inference state dict.

    Parameters
    ----------
    width, height : int
        Video resolution.
    device : str
        ``"cuda"`` or ``"cpu"``.
    multiplier : float, optional
        Override the default ``MODEL_STATE_MULTIPLIER``.

    Returns
    -------
    int
        Estimated bytes consumed per frame (including model state).
    """
    mult = multiplier if multiplier is not None else MODEL_STATE_MULTIPLIER

    # Raw decoded frame (uint8 BGR)
    raw_bytes = width * height * 3

    # Resolution-dependent model overhead: feature maps + per-object masks.
    # Factor in float32 (×4) for internal tensor storage.
    variable_bytes = int(width * height * 4 * mult)

    # Fixed model-internal overhead per frame (at model resolution, not video res)
    fixed_bytes = MODEL_FIXED_PER_FRAME_BYTES

    per_frame = raw_bytes + variable_bytes + fixed_bytes
    return max(per_frame, MIN_PER_FRAME_BYTES)


# ---------------------------------------------------------------------------
# Live calibration (process a few frames, measure actual growth)
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Result of a per-frame memory calibration probe."""
    measured_bytes_per_frame: int = 0
    baseline_bytes: int = 0
    peak_bytes: int = 0
    n_calibration_frames: int = 0
    device: str = "cuda"
    confidence: float = 0.0  # 0-1; higher = more samples


def calibrate_per_frame_cost(
    device: str = "cuda",
    n_frames: int = 3,
) -> CalibrationResult:
    """Measure actual per-frame GPU memory growth from the current session.

    This is intended to be called *after* the model is loaded and a session
    is started, but *before* full propagation.  The caller should process
    ``n_frames`` frames and call ``record_sample`` between each.

    Returns a ``CalibrationResult`` that can override the heuristic estimate
    in ``compute_memory_safe_frames``.

    Note: This is a data-class factory.  The actual sampling must be driven
    externally because we don't own the inference loop.
    """
    snap = get_memory(device)
    return CalibrationResult(
        baseline_bytes=snap.used,
        device=device,
        n_calibration_frames=0,
    )


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------

@contextmanager
def inference_context(device: str = "cuda"):
    """Context manager that wraps a block in ``torch.inference_mode``
    and performs memory cleanup on exit.

    Usage::

        with inference_context("cuda"):
            result = driver.propagate_in_video(session_id)
    """
    try:
        import torch
        ctx = torch.inference_mode()
        ctx.__enter__()
    except Exception:
        ctx = None

    try:
        yield
    finally:
        # Cleanup on exit
        if ctx is not None:
            ctx.__exit__(None, None, None)
        clear_memory(device, full_gc=False)


@contextmanager
def memory_cleanup_context(device: str = "cuda"):
    """Lightweight context that only clears memory on exit (no inference_mode)."""
    try:
        yield
    finally:
        clear_memory(device)


# ---------------------------------------------------------------------------
# Memory tier auto-detection
# ---------------------------------------------------------------------------

def get_memory_tier(vram_bytes: int = 0, ram_bytes: int = 0) -> dict:
    """Auto-select threshold profile based on hardware capabilities.

    Returns a dict of recommended growth-strategy parameters for the
    detected memory tier.  The tier system ensures safe, efficient
    operation across a wide range of hardware — from 8 GB consumer GPUs
    to 80 GB data-centre cards.

    Tiers
    -----
    * **S**  (≤ 12 GB VRAM): Conservative.  Model weights (~3 GB) eat a
      large fraction; headroom is precious.
    * **M**  (12–24 GB): Balanced.  RTX 3090 / 4090 class.
    * **L**  (24–48 GB): Generous.  A6000, RTX 6000 Ada class.
    * **XL** (> 48 GB): Aggressive.  A100, H100 — ample headroom.
    * **CPU_S / CPU_M / CPU_L**: CPU-only tiers keyed on system RAM.

    The tiers control the *adaptive chunk manager* (growth factor,
    min/max chunk sizes).  Runtime safety thresholds (soft/hard limits)
    remain under user control in ``__globals.py``.
    """
    vram_gb = vram_bytes / (1024**3) if vram_bytes > 0 else 0.0
    ram_gb = ram_bytes / (1024**3) if ram_bytes > 0 else 0.0

    # ── GPU tiers ──
    if vram_gb > 0:
        if vram_gb <= 12:
            return {
                "tier": "S", "vram_gb": round(vram_gb, 1),
                "min_chunk_frames": 15, "grow_factor": 1.30,
                "max_growth_factor": 2.0, "grow_threshold": 0.50,
            }
        elif vram_gb <= 24:
            return {
                "tier": "M", "vram_gb": round(vram_gb, 1),
                "min_chunk_frames": 25, "grow_factor": 1.40,
                "max_growth_factor": 2.5, "grow_threshold": 0.55,
            }
        elif vram_gb <= 48:
            return {
                "tier": "L", "vram_gb": round(vram_gb, 1),
                "min_chunk_frames": 50, "grow_factor": 1.50,
                "max_growth_factor": 3.0, "grow_threshold": 0.60,
            }
        else:
            return {
                "tier": "XL", "vram_gb": round(vram_gb, 1),
                "min_chunk_frames": 100, "grow_factor": 1.50,
                "max_growth_factor": 3.0, "grow_threshold": 0.65,
            }

    # ── CPU-only tiers (keyed on system RAM) ──
    if ram_gb <= 16:
        return {
            "tier": "CPU_S", "vram_gb": 0, "ram_gb": round(ram_gb, 1),
            "min_chunk_frames": 15, "grow_factor": 1.20,
            "max_growth_factor": 1.5, "grow_threshold": 0.40,
        }
    elif ram_gb <= 32:
        return {
            "tier": "CPU_M", "vram_gb": 0, "ram_gb": round(ram_gb, 1),
            "min_chunk_frames": 25, "grow_factor": 1.30,
            "max_growth_factor": 2.0, "grow_threshold": 0.50,
        }
    else:
        return {
            "tier": "CPU_L", "vram_gb": 0, "ram_gb": round(ram_gb, 1),
            "min_chunk_frames": 50, "grow_factor": 1.40,
            "max_growth_factor": 2.5, "grow_threshold": 0.55,
        }


# ---------------------------------------------------------------------------
# Memory pressure levels
# ---------------------------------------------------------------------------

class MemoryPressure:
    """Memory pressure levels for adaptive chunk management."""
    NORMAL = "NORMAL"       # < 60% usage — may increase chunk size
    ELEVATED = "ELEVATED"   # 60-80% — keep current chunk size
    WARNING = "WARNING"     # 80-90% — reduce chunk size
    CRITICAL = "CRITICAL"   # > 90% — aggressively reduce
    OOM = "OOM"             # actual OOM occurred

    # Thresholds (fraction of effective memory limit)
    ELEVATED_THRESHOLD = 0.60
    WARNING_THRESHOLD = 0.80
    CRITICAL_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Adaptive chunk manager
# ---------------------------------------------------------------------------

@dataclass
class ChunkMemoryRecord:
    """Record of a single chunk's memory behaviour."""
    chunk_id: int = 0
    chunk_size: int = 0             # frames in this chunk
    peak_vram_bytes: int = 0
    peak_ram_bytes: int = 0
    n_objects: int = 0              # total objects across all prompts
    vram_usage_pct: float = 0.0     # peak_vram / effective_limit
    pressure: str = "NORMAL"
    action: str = "CONTINUE"        # CONTINUE | SHRINK | GROW | RECHUNK
    adjusted_chunk_size: int = 0    # chunk size for NEXT chunk


class AdaptiveChunkManager:
    """Manages dynamic chunk sizing based on observed memory pressure.

    After each chunk completes, call :py:meth:`record_chunk` with peak
    memory observations.  The manager evaluates memory pressure and
    adjusts the chunk size for subsequent chunks.

    When a CUDA OOM occurs during propagation, call :py:meth:`handle_oom`
    to get a safe fallback chunk size and retry instructions.

    Parameters
    ----------
    initial_chunk_size : int
        Starting frames per chunk (from static planner).
    device : str
        ``"cuda"`` or ``"cpu"``.
    vram_limit_bytes : int, optional
        Override GPU memory limit (for simulated testing).
    ram_limit_bytes : int, optional
        Override system RAM limit (for simulated testing).
    min_chunk_frames : int
        Absolute minimum chunk size (to avoid degenerate 1-frame chunks).
    max_growth_factor : float
        Maximum factor by which chunk size can grow between chunks.
    tier : dict, optional
        Output of :func:`get_memory_tier`.  When provided the tier's
        growth-strategy parameters override the class defaults.
    """

    # Region: tuning constants (may be overridden per-instance by tier)
    SHRINK_CRITICAL_FACTOR = 0.50   # halve on critical
    SHRINK_WARNING_FACTOR = 0.75    # reduce 25% on warning
    GROW_FACTOR = 1.25              # default grow 25% (tier may override)
    OOM_SHRINK_FACTOR = 0.40        # aggressive shrink on actual OOM
    MAX_CONSECUTIVE_OOMS = 3        # give up after 3 OOMs on same chunk
    GROW_THRESHOLD = 0.50           # grow only when usage < this (tier may override)

    def __init__(
        self,
        initial_chunk_size: int,
        device: str = "cuda",
        *,
        vram_limit_bytes: Optional[int] = None,
        ram_limit_bytes: Optional[int] = None,
        min_chunk_frames: int = 25,
        max_growth_factor: float = 1.5,
        tier: Optional[dict] = None,
    ):
        self.device = device
        self.initial_chunk_size = initial_chunk_size
        self.current_chunk_size = initial_chunk_size
        self.min_chunk_frames = min_chunk_frames
        self.max_growth_factor = max_growth_factor

        # Resolve memory limits
        if device.startswith("cuda"):
            if vram_limit_bytes is not None:
                self.vram_limit = vram_limit_bytes
            else:
                snap = get_gpu_memory()
                self.vram_limit = snap.total if snap.total > 0 else 80 * 1024**3
        else:
            self.vram_limit = 0

        if ram_limit_bytes is not None:
            self.ram_limit = ram_limit_bytes
        else:
            snap = get_cpu_memory()
            self.ram_limit = snap.total

        # Apply tier-specific growth strategy (explicit params take precedence)
        self._tier = tier or get_memory_tier(self.vram_limit, self.ram_limit)
        self.GROW_FACTOR = self._tier.get("grow_factor", self.GROW_FACTOR)
        self.GROW_THRESHOLD = self._tier.get("grow_threshold", self.GROW_THRESHOLD)
        # Only override min/max if caller used defaults (sentinel-free: tier
        # refines but explicit params always win — callers that pass
        # min_chunk_frames=25 intentionally should subclass or pass tier=None).
        if tier is not None:
            self.min_chunk_frames = tier.get("min_chunk_frames", self.min_chunk_frames)
            self.max_growth_factor = tier.get("max_growth_factor", self.max_growth_factor)

        # Track history
        self.chunk_history: list = []
        self.rechunk_events: list = []
        self._consecutive_ooms = 0

    @property
    def effective_vram_limit(self) -> int:
        """Maximum total GPU allocation before we consider memory critical.

        ``torch.cuda.max_memory_allocated()`` reports *total* allocated
        memory (model weights **+** inference tensors).  So we compare it
        against the full VRAM limit minus a small driver/OS safety reserve
        — we must **not** subtract model weight overhead since it is
        already captured by the peak measurement.
        """
        try:
            from sam3.__globals import GPU_MEMORY_RESERVE_PERCENT
            reserve_pct = GPU_MEMORY_RESERVE_PERCENT
        except ImportError:
            reserve_pct = 0.05
        return int(self.vram_limit * (1 - reserve_pct))

    def evaluate_pressure(self, peak_vram_bytes: int) -> str:
        """Classify memory pressure based on peak VRAM usage."""
        if self.effective_vram_limit <= 0:
            return MemoryPressure.NORMAL

        usage_pct = peak_vram_bytes / self.effective_vram_limit

        if usage_pct >= MemoryPressure.CRITICAL_THRESHOLD:
            return MemoryPressure.CRITICAL
        elif usage_pct >= MemoryPressure.WARNING_THRESHOLD:
            return MemoryPressure.WARNING
        elif usage_pct >= MemoryPressure.ELEVATED_THRESHOLD:
            return MemoryPressure.ELEVATED
        return MemoryPressure.NORMAL

    def record_chunk(
        self,
        chunk_id: int,
        chunk_size: int,
        peak_vram_bytes: int,
        peak_ram_bytes: int = 0,
        n_objects: int = 0,
        soft_warning_seen: bool = False,
    ) -> ChunkMemoryRecord:
        """Record a completed chunk and compute next chunk size.

        Parameters
        ----------
        soft_warning_seen : bool
            If *True*, at least one per-prompt intra-chunk monitor
            triggered a VRAM or RAM soft-limit warning during this
            chunk.  When set, the adaptive manager will never GROW —
            even if the aggregated peak falls under the NORMAL
            threshold — because a prompt was genuinely close to the
            limit at runtime.

        Returns a :class:`ChunkMemoryRecord` with the recommended
        ``adjusted_chunk_size`` for the next chunk.
        """
        self._consecutive_ooms = 0  # reset on success

        pressure = self.evaluate_pressure(peak_vram_bytes)
        usage_pct = (
            peak_vram_bytes / self.effective_vram_limit
            if self.effective_vram_limit > 0
            else 0.0
        )

        # Determine action
        if pressure == MemoryPressure.CRITICAL:
            new_size = max(
                int(chunk_size * self.SHRINK_CRITICAL_FACTOR),
                self.min_chunk_frames,
            )
            action = "SHRINK"
        elif pressure == MemoryPressure.WARNING:
            new_size = max(
                int(chunk_size * self.SHRINK_WARNING_FACTOR),
                self.min_chunk_frames,
            )
            action = "SHRINK"
        elif (
            pressure == MemoryPressure.NORMAL
            and usage_pct < self.GROW_THRESHOLD
            and not soft_warning_seen
        ):
            # Under-utilised: grow but cap at max_growth_factor × initial.
            # Dampen growth when many objects are tracked — SAM3 memory
            # per frame scales roughly with object count, so the naive
            # growth that is safe for 1 object would be dangerous for 29.
            # Formula: growth = 1 + (G-1) / (1 + log2(N))
            #
            # Object-count trend: if objects are increasing between chunks
            # (scene getting busier), further dampen.  If decreasing
            # (objects leaving), boost growth slightly.
            #
            # Growth is suppressed entirely when soft_warning_seen is
            # True — a per-prompt monitor flagged VRAM/RAM pressure
            # even though the post-chunk peak may look tame.
            import math
            n = max(n_objects, 1)
            dampened_growth = 1.0 + (self.GROW_FACTOR - 1.0) / (1.0 + math.log2(n))

            # Trend adjustment: compare current n_objects with previous chunk
            if len(self.chunk_history) >= 1:
                prev_obj = self.chunk_history[-1].n_objects
                if prev_obj > 0 and n_objects > prev_obj * 1.2:
                    # Objects increasing >20% → further dampen
                    dampened_growth = 1.0 + (dampened_growth - 1.0) * 0.5
                elif prev_obj > 0 and n_objects < prev_obj * 0.8:
                    # Objects decreasing >20% → boost (but cap to full factor)
                    dampened_growth = min(
                        dampened_growth * 1.3,
                        1.0 + (self.GROW_FACTOR - 1.0),
                    )

            new_size = min(
                int(chunk_size * dampened_growth),
                int(self.initial_chunk_size * self.max_growth_factor),
            )
            action = "GROW"
        else:
            new_size = chunk_size
            action = "CONTINUE"

        self.current_chunk_size = new_size

        rec = ChunkMemoryRecord(
            chunk_id=chunk_id,
            chunk_size=chunk_size,
            peak_vram_bytes=peak_vram_bytes,
            peak_ram_bytes=peak_ram_bytes,
            n_objects=n_objects,
            vram_usage_pct=round(usage_pct * 100, 1),
            pressure=pressure,
            action=action,
            adjusted_chunk_size=new_size,
        )
        self.chunk_history.append(rec)

        if action in ("SHRINK", "GROW"):
            self.rechunk_events.append({
                "chunk_id": chunk_id,
                "from_size": chunk_size,
                "to_size": new_size,
                "reason": pressure,
                "peak_vram_pct": round(usage_pct * 100, 1),
                "n_objects": n_objects,
            })

        return rec

    def handle_oom(self, chunk_id: int, chunk_size: int) -> int:
        """Called when CUDA OOM occurs during propagation.

        Returns a reduced chunk size for retry.  After
        ``MAX_CONSECUTIVE_OOMS`` failures, raises ``RuntimeError``.
        """
        self._consecutive_ooms += 1
        if self._consecutive_ooms > self.MAX_CONSECUTIVE_OOMS:
            raise RuntimeError(
                f"CUDA OOM persisted after {self.MAX_CONSECUTIVE_OOMS} retries "
                f"at chunk size {chunk_size}. Cannot proceed."
            )

        new_size = max(
            int(chunk_size * self.OOM_SHRINK_FACTOR),
            self.min_chunk_frames,
        )
        self.rechunk_events.append({
            "chunk_id": chunk_id,
            "from_size": chunk_size,
            "to_size": new_size,
            "reason": "OOM",
            "retry": self._consecutive_ooms,
        })
        self.current_chunk_size = new_size
        return new_size

    def replan_remaining(
        self,
        remaining_start: int,
        total_frames: int,
        overlap: int = 1,
    ) -> list:
        """Generate new chunk plan for remaining frames with current chunk size.

        Returns a list of ``{"chunk": i, "start": s, "end": e}`` dicts.
        """
        chunk_size = self.current_chunk_size
        stride = max(chunk_size - overlap, 1)
        chunks = []
        start = remaining_start
        idx = 0
        while start < total_frames:
            end = min(start + chunk_size - 1, total_frames - 1)
            if end > start:
                chunks.append({"chunk": idx, "start": start, "end": end})
            start += stride
            idx += 1
        return chunks

    def to_dict(self) -> dict:
        """Serialise full state for metadata export."""
        return {
            "initial_chunk_size": self.initial_chunk_size,
            "final_chunk_size": self.current_chunk_size,
            "device": self.device,
            "tier": self._tier,
            "grow_factor": self.GROW_FACTOR,
            "grow_threshold": self.GROW_THRESHOLD,
            "vram_limit_bytes": self.vram_limit,
            "ram_limit_bytes": self.ram_limit,
            "effective_vram_limit_bytes": self.effective_vram_limit,
            "min_chunk_frames": self.min_chunk_frames,
            "max_growth_factor": self.max_growth_factor,
            "chunk_history": [
                {
                    "chunk_id": r.chunk_id,
                    "chunk_size": r.chunk_size,
                    "peak_vram_mb": round(r.peak_vram_bytes / (1024**2), 1),
                    "peak_ram_mb": round(r.peak_ram_bytes / (1024**2), 1),
                    "n_objects": r.n_objects,
                    "vram_usage_pct": r.vram_usage_pct,
                    "pressure": r.pressure,
                    "action": r.action,
                    "adjusted_chunk_size": r.adjusted_chunk_size,
                }
                for r in self.chunk_history
            ],
            "rechunk_events": self.rechunk_events,
        }


# ---------------------------------------------------------------------------
# Intra-chunk proactive memory monitoring
# ---------------------------------------------------------------------------

def _linear_regression(xs: List[float], ys: List[float]):
    """Ordinary least-squares fit: y = slope * x + intercept.

    Returns ``(slope, intercept, r_squared)``.  Pure Python — no numpy
    dependency.
    """
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0), 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    ss_yy = sum((y - mean_y) ** 2 for y in ys)

    if ss_xx == 0:
        return 0.0, mean_y, 0.0

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x

    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1 - (ss_res / ss_yy) if ss_yy > 0 else 0.0

    return slope, intercept, max(0.0, r_squared)


@dataclass
class FrameMemorySample:
    """Single memory snapshot taken during frame propagation."""
    iteration: int = 0          # 0-based processing step
    frame_idx: int = 0          # video frame index (may repeat for "both")
    vram_allocated: int = 0     # bytes from torch.cuda.memory_allocated()
    ram_allocated: int = 0      # bytes from psutil.Process().memory_info().rss
    timestamp: float = 0.0      # time.time()


@dataclass
class GrowthCalibration:
    """Linear growth model fitted from calibration samples.

    Predicts VRAM at future iterations via::

        predicted_vram = baseline + growth_rate * iteration
    """
    growth_rate_per_iter: float = 0.0   # bytes per iteration (slope)
    baseline_bytes: int = 0             # intercept (VRAM at iteration 0)
    safe_iterations: int = 0            # max iterations before hard-stop
    confidence: float = 0.0             # 0-1 (fit quality × sample count)
    n_samples: int = 0
    r_squared: float = 0.0             # linear fit quality


@dataclass
class MonitorResult:
    """Summary of intra-chunk monitoring for one propagation run."""
    iterations_planned: int = 0
    iterations_completed: int = 0
    early_stopped: bool = False
    stop_reason: str = "completed"      # completed|hard_limit|predictive_soft_stop|ram_hard_limit|ram_soft_limit|oom_exception
    peak_vram_bytes: int = 0
    peak_ram_bytes: int = 0
    calibration: Optional[GrowthCalibration] = None
    checkpoints_evaluated: int = 0
    samples: List[FrameMemorySample] = field(default_factory=list)


class IntraChunkMonitor:
    """Per-frame VRAM **and** RAM guard for video propagation.

    Checks memory utilisation **after every single frame** — the overhead
    is ~10 µs per call vs ~0.4–1 s per frame of SAM3 inference (<0.01%).

    Dual-resource, dual-threshold design
    ------------------------------------
    Both VRAM and RAM are monitored with soft / hard limits:

    * **Soft limit** (default 80% VRAM, 70% RAM):
      Issues a warning and — once a reliable calibration exists — predicts
      how many frames remain before the hard limit.  If the prediction
      says < ``PREDICTIVE_STOP_FRAMES`` frames, triggers an early stop.

    * **Hard limit** (default 92% VRAM, 85% RAM):
      Signals immediate stop.  The headroom above this prevents
      PyTorch's caching allocator (VRAM) or the Linux OOM-killer (RAM)
      from triggering a catastrophic failure.

    Calibration & recalibration
    --------------------------
    During the first ``CALIBRATION_FRAMES`` iterations, VRAM is sampled
    every frame and a linear growth model (slope + intercept) is fitted.
    The model is **re-fitted every ``RECALIBRATE_INTERVAL`` frames** so
    the slope converges to steady-state behaviour and is not dominated by
    the steep warmup (model init / cache fill) gradient.

    The calibration is used **only** inside the soft-limit zone to
    estimate frames-until-hard-limit.  There is no speculative
    "predicted OOM" check based purely on extrapolation — that approach
    produces false positives from warmup data.

    Usage::

        monitor = IntraChunkMonitor(expected_iterations=1000, device="cuda")
        monitor.start()
        for frame_idx, outputs in stream:
            if not monitor.check(frame_idx):
                break
            collect(outputs)
        result = monitor.finalize()
    """

    # --- Tuning constants ---
    CALIBRATION_FRAMES: int = 5       # first N iterations: always sample
    RECALIBRATE_INTERVAL: int = 50    # re-fit calibration every N frames
    PREDICTIVE_STOP_FRAMES: int = 50  # stop early if predicted < this many frames left
    MIN_CALIBRATION_SAMPLES: int = 3  # minimum for valid prediction

    def __init__(
        self,
        expected_iterations: int,
        device: str = "cuda",
        *,
        vram_limit_bytes: Optional[int] = None,
        ram_limit_bytes: Optional[int] = None,
        soft_limit_pct: Optional[float] = None,
        hard_limit_pct: Optional[float] = None,
        ram_soft_limit_pct: Optional[float] = None,
        ram_hard_limit_pct: Optional[float] = None,
    ):
        self.expected_iterations = expected_iterations
        self.device = device
        self.vram_limit = self._resolve_vram_limit(vram_limit_bytes)

        # ── VRAM thresholds ──
        try:
            from sam3.__globals import VRAM_SOFT_LIMIT_PCT, VRAM_HARD_LIMIT_PCT
            self._soft_pct = soft_limit_pct if soft_limit_pct is not None else VRAM_SOFT_LIMIT_PCT
            self._hard_pct = hard_limit_pct if hard_limit_pct is not None else VRAM_HARD_LIMIT_PCT
        except ImportError:
            self._soft_pct = soft_limit_pct if soft_limit_pct is not None else 0.80
            self._hard_pct = hard_limit_pct if hard_limit_pct is not None else 0.92

        # ── RAM limits and thresholds ──
        self.ram_limit = self._resolve_ram_limit(ram_limit_bytes)
        try:
            from sam3.__globals import RAM_SOFT_LIMIT_PCT, RAM_HARD_LIMIT_PCT
            self._ram_soft_pct = ram_soft_limit_pct if ram_soft_limit_pct is not None else RAM_SOFT_LIMIT_PCT
            self._ram_hard_pct = ram_hard_limit_pct if ram_hard_limit_pct is not None else RAM_HARD_LIMIT_PCT
        except ImportError:
            self._ram_soft_pct = ram_soft_limit_pct if ram_soft_limit_pct is not None else 0.70
            self._ram_hard_pct = ram_hard_limit_pct if ram_hard_limit_pct is not None else 0.85

        # State
        self._samples: List[FrameMemorySample] = []
        self._calibration: Optional[GrowthCalibration] = None
        self._baseline_vram: int = 0
        self._iteration: int = 0
        self._peak_vram: int = 0
        self._peak_ram: int = 0
        self._stop_reason: Optional[str] = None
        self._soft_warning_issued: bool = False
        self._ram_soft_warning_issued: bool = False
        self._frames_at_soft_warning: int = 0

    # ── Setup ────────────────────────────────────────────────────────────

    def _resolve_vram_limit(self, override: Optional[int]) -> int:
        """Determine effective VRAM limit."""
        if override:
            return override
        if self.device.startswith("cuda"):
            snap = get_gpu_memory()
            return snap.total if snap.total > 0 else 0
        return 0

    def _resolve_ram_limit(self, override: Optional[int]) -> int:
        """Determine total system RAM."""
        if override:
            return override
        try:
            return psutil.virtual_memory().total
        except Exception:
            return 0

    @property
    def effective_limit(self) -> int:
        """VRAM budget = total limit minus OS/driver reserve."""
        if self.vram_limit <= 0:
            return 0
        try:
            from sam3.__globals import GPU_MEMORY_RESERVE_PERCENT
            reserve = GPU_MEMORY_RESERVE_PERCENT
        except ImportError:
            reserve = 0.05
        return int(self.vram_limit * (1 - reserve))

    @property
    def effective_ram_limit(self) -> int:
        """RAM budget = total RAM minus OS reserve."""
        if self.ram_limit <= 0:
            return 0
        try:
            from sam3.__globals import CPU_MEMORY_RESERVE_PERCENT
            reserve = CPU_MEMORY_RESERVE_PERCENT
        except ImportError:
            reserve = 0.30
        return int(self.ram_limit * (1 - reserve))

    # ── Runtime ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Record baseline VRAM before propagation begins."""
        if self.device.startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    self._baseline_vram = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    def check(self, frame_idx: int = -1) -> bool:
        """Check memory at current iteration.  Returns *True* to continue.

        Called **after every frame**.  Per-frame overhead is one
        ``torch.cuda.memory_allocated()`` call (~10 µs) plus one
        ``psutil`` RSS query (~5 µs).  Negligible compared to frame time.
        """
        iteration = self._iteration
        self._iteration += 1
        return self._evaluate(iteration, frame_idx)

    # ── Evaluation ───────────────────────────────────────────────────────

    def _evaluate(self, iteration: int, frame_idx: int) -> bool:
        """Full memory evaluation — called every frame."""
        sample = self._take_sample(iteration, frame_idx)
        self._peak_vram = max(self._peak_vram, sample.vram_allocated)
        self._peak_ram = max(self._peak_ram, sample.ram_allocated)

        # Initial calibration after collecting enough samples
        if (len(self._samples) >= self.CALIBRATION_FRAMES
                and self._calibration is None):
            self._calibrate()

        # Periodic recalibration so slope converges to steady-state
        # (initial calibration from warmup frames is unreliable)
        elif (self._calibration is not None
              and iteration > 0
              and iteration % self.RECALIBRATE_INTERVAL == 0
              and len(self._samples) > self._calibration.n_samples):
            self._calibrate()

        # CPU mode with no VRAM limit — still check RAM
        if self.effective_limit <= 0:
            return self._check_ram_limits(sample, iteration)

        return self._check_limits(sample, iteration)

    def _take_sample(self, iteration: int, frame_idx: int) -> FrameMemorySample:
        """Record a VRAM + RAM snapshot."""
        vram = 0
        if self.device.startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    vram = torch.cuda.memory_allocated()
            except Exception:
                pass

        ram = 0
        try:
            ram = psutil.Process().memory_info().rss
        except Exception:
            pass

        sample = FrameMemorySample(
            iteration=iteration,
            frame_idx=frame_idx,
            vram_allocated=vram,
            ram_allocated=ram,
            timestamp=time.time(),
        )
        # Keep all calibration samples; after calibration keep every 10th
        # to bound memory for very long chunks while preserving key points
        if iteration < self.CALIBRATION_FRAMES or iteration % 10 == 0:
            self._samples.append(sample)
        return sample

    def _check_limits(self, sample: FrameMemorySample, iteration: int) -> bool:
        """Evaluate VRAM and RAM hard-stop and soft-stop thresholds."""
        usage_pct = sample.vram_allocated / self.effective_limit

        # ── VRAM HARD LIMIT: current allocation is dangerously high ──
        if usage_pct >= self._hard_pct:
            self._stop_reason = "hard_limit"
            return False

        # ── VRAM SOFT LIMIT: approaching danger zone ──
        if usage_pct >= self._soft_pct:
            if not self._soft_warning_issued:
                self._soft_warning_issued = True
                self._frames_at_soft_warning = iteration
                import sys
                print(f"\033[93m⚠ VRAM warning: {usage_pct:.0%} used "
                      f"(soft limit {self._soft_pct:.0%}) at frame {iteration}\033[0m",
                      file=sys.stderr)

            # Predictive stop: use calibration to estimate frames until hard limit
            if (self._calibration is not None
                    and self._calibration.confidence > 0.3
                    and self._calibration.growth_rate_per_iter > 0):
                headroom_bytes = self.effective_limit * self._hard_pct - sample.vram_allocated
                frames_until_hard = headroom_bytes / self._calibration.growth_rate_per_iter
                if frames_until_hard < self.PREDICTIVE_STOP_FRAMES:
                    self._stop_reason = "predictive_soft_stop"
                    return False

            # Fallback: if no calibration, check if we're very close to hard
            elif usage_pct >= (self._soft_pct + self._hard_pct) / 2:
                # Midpoint between soft and hard — stop without calibration
                self._stop_reason = "soft_limit_no_calibration"
                return False

        # ── RAM checks (same dual-threshold logic) ──
        return self._check_ram_limits(sample, iteration)

    def _check_ram_limits(self, sample: FrameMemorySample, iteration: int) -> bool:
        """Evaluate RAM hard-stop and soft-stop thresholds."""
        if self.effective_ram_limit <= 0:
            return True  # cannot determine RAM limit

        ram_pct = sample.ram_allocated / self.effective_ram_limit

        if ram_pct >= self._ram_hard_pct:
            self._stop_reason = "ram_hard_limit"
            return False

        if ram_pct >= self._ram_soft_pct:
            if not self._ram_soft_warning_issued:
                self._ram_soft_warning_issued = True
                import sys
                print(f"\033[93m⚠ RAM warning: {ram_pct:.0%} used "
                      f"(soft limit {self._ram_soft_pct:.0%}) at frame {iteration}\033[0m",
                      file=sys.stderr)

            # Midpoint heuristic (no calibration model for RAM)
            if ram_pct >= (self._ram_soft_pct + self._ram_hard_pct) / 2:
                self._stop_reason = "ram_soft_limit"
                return False

        return True

    # ── Calibration ──────────────────────────────────────────────────────

    def _calibrate(self) -> None:
        """Fit linear model: vram = intercept + slope × iteration.

        Called once after initial calibration frames, then periodically
        every ``RECALIBRATE_INTERVAL`` frames so the slope converges to
        steady-state and is not dominated by steep warmup growth.
        """
        samples = self._samples
        n = len(samples)
        if n < self.MIN_CALIBRATION_SAMPLES:
            return

        xs = [float(s.iteration) for s in samples]
        ys = [float(s.vram_allocated) for s in samples]

        slope, intercept, r_sq = _linear_regression(xs, ys)

        # Predict safe iterations (how many until hard-stop threshold)
        safe_iters = self.expected_iterations
        if slope > 0 and self.effective_limit > 0:
            headroom = self.effective_limit * self._hard_pct - intercept
            safe_iters = max(0, int(headroom / slope))

        confidence = min(1.0, n / 10) * max(0.0, r_sq)

        self._calibration = GrowthCalibration(
            growth_rate_per_iter=slope,
            baseline_bytes=int(intercept),
            safe_iterations=safe_iters,
            confidence=round(confidence, 4),
            n_samples=n,
            r_squared=round(r_sq, 6),
        )

    def _predict_at(self, target_iteration: int) -> int:
        """Predict VRAM at a future iteration."""
        if not self._calibration:
            return 0
        cal = self._calibration
        return max(0, int(cal.baseline_bytes + cal.growth_rate_per_iter * target_iteration))

    # ── Finalisation ─────────────────────────────────────────────────────

    def finalize(self) -> MonitorResult:
        """Compile monitoring summary.  Call after propagation loop exits."""
        if self.device.startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    self._peak_vram = max(
                        self._peak_vram,
                        torch.cuda.max_memory_allocated(),
                    )
            except Exception:
                pass

        try:
            self._peak_ram = max(self._peak_ram, psutil.Process().memory_info().rss)
        except Exception:
            pass

        return MonitorResult(
            iterations_planned=self.expected_iterations,
            iterations_completed=self._iteration,
            early_stopped=self._stop_reason is not None,
            stop_reason=self._stop_reason or "completed",
            peak_vram_bytes=self._peak_vram,
            peak_ram_bytes=self._peak_ram,
            calibration=self._calibration,
            checkpoints_evaluated=self._iteration,  # every frame evaluated
            samples=list(self._samples),
        )

    def to_dict(self) -> dict:
        """Serialise for JSON metadata export."""
        r = self.finalize()
        cal = r.calibration
        return {
            "iterations_planned": r.iterations_planned,
            "iterations_completed": r.iterations_completed,
            "early_stopped": r.early_stopped,
            "stop_reason": r.stop_reason,
            "peak_vram_mb": round(r.peak_vram_bytes / (1024**2), 1),
            "peak_ram_mb": round(r.peak_ram_bytes / (1024**2), 1),
            "checkpoints_evaluated": r.checkpoints_evaluated,
            "soft_limit_pct": self._soft_pct,
            "hard_limit_pct": self._hard_pct,
            "ram_soft_limit_pct": self._ram_soft_pct,
            "ram_hard_limit_pct": self._ram_hard_pct,
            "soft_warning_issued": self._soft_warning_issued,
            "ram_soft_warning_issued": self._ram_soft_warning_issued,
            "calibration": {
                "growth_rate_mb_per_iter": round(cal.growth_rate_per_iter / (1024**2), 3),
                "baseline_mb": round(cal.baseline_bytes / (1024**2), 1),
                "safe_iterations": cal.safe_iterations,
                "confidence": cal.confidence,
                "r_squared": cal.r_squared,
            } if cal else None,
            "memory_samples": [
                {
                    "iteration": s.iteration,
                    "frame_idx": s.frame_idx,
                    "vram_mb": round(s.vram_allocated / (1024**2), 1),
                    "ram_mb": round(s.ram_allocated / (1024**2), 1),
                }
                for s in r.samples
            ],
        }
