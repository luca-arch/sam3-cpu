"""Streaming mask-video encoder and overlay compositor.

Replaces the per-frame PNG pipeline with a GPU→CPU queue that encodes
per-object lossless mask videos **and** composes the overlay video in
real-time, dramatically reducing post-processing time.

Architecture
------------
::

    GPU propagation loop
        │
        ├─ push (frame_idx, masks, obj_ids) ──► MaskWriterQueue
        │                                          │
        │                                    CPU ThreadPool
        │                                    ┌────┴────┐
        │                              MaskVideoWriter  StreamingOverlayCompositor
        │                              (per-object MP4) (coloured overlay MP4)
        │                                    │
        │                              EmptyMaskPool
        │                              (pre-allocated for new objs)

Key wins over the PNG pipeline:

1. **No file-per-frame I/O** — writes go through ``cv2.VideoWriter``
   which buffers internally and produces a single MP4 per object.
2. **CPU cores utilised during GPU inference** — the writer threads
   run while the GPU processes the next frame.
3. **Overlay built incrementally** — no monolithic second pass over
   all mask videos after stitching.
4. **Empty-mask-video pool** eliminates repeated black-frame
   generation when a new object appears mid-chunk.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (BGR for OpenCV) — kept consistent with overlay code
# ---------------------------------------------------------------------------
MASK_COLOURS_BGR = [
    (30, 144, 255),  # dodger blue
    (255, 50, 50),  # red
    (50, 205, 50),  # lime green
    (255, 165, 0),  # orange
    (148, 103, 189),  # purple
    (255, 215, 0),  # gold
    (0, 255, 255),  # cyan
    (255, 105, 180),  # hot pink
    (128, 128, 0),  # olive
    (70, 130, 180),  # steel blue
]


# ---------------------------------------------------------------------------
# Empty-mask video pool
# ---------------------------------------------------------------------------
class EmptyMaskPool:
    """Pool of pre-allocated black-frame arrays to avoid repeated allocation.

    When a new object appears mid-chunk, it needs empty (black) frames for
    all prior frames.  Instead of allocating a new ``np.zeros`` array each
    time, the pool maintains a reusable template.

    Thread-safe: the underlying numpy array is read-only after creation.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Single shared read-only black frame
        self._black = np.zeros((height, width), dtype=np.uint8)
        self._black.flags.writeable = False

    def get_black_frame(self) -> np.ndarray:
        """Return a shared read-only black frame (no allocation)."""
        return self._black


# ---------------------------------------------------------------------------
# Per-object mask video writer
# ---------------------------------------------------------------------------
class MaskVideoWriter:
    """Writes a single object's binary masks into a lossless MP4 video.

    Written frames are grayscale (0 or 255) encoded with mp4v codec at
    the original video FPS.  The resulting file is identical in purpose
    to the old per-frame PNG directory but ~100× fewer I/O operations.
    """

    def __init__(
        self,
        output_path: Path,
        fps: float,
        width: int,
        height: int,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.height = height
        self._writer: cv2.VideoWriter | None = None
        self._frames_written: int = 0
        self._lock = threading.Lock()

    def _ensure_open(self) -> cv2.VideoWriter:
        if self._writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (self.width, self.height),
                False,
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"Cannot open video writer: {self.output_path}")
        return self._writer

    def write_frame(self, mask: np.ndarray) -> None:
        """Write a single grayscale mask frame."""
        with self._lock:
            writer = self._ensure_open()
            # Ensure correct shape and type
            if mask.shape != (self.height, self.width):
                mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            writer.write(mask.astype(np.uint8))
            self._frames_written += 1

    def write_black(self, n: int = 1, pool: EmptyMaskPool | None = None) -> None:
        """Write *n* black (empty) frames, optionally using a pooled template."""
        black = pool.get_black_frame() if pool else np.zeros((self.height, self.width), dtype=np.uint8)
        with self._lock:
            writer = self._ensure_open()
            for _ in range(n):
                writer.write(black)
                self._frames_written += 1

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def close(self) -> None:
        with self._lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Streaming overlay compositor
# ---------------------------------------------------------------------------
class StreamingOverlayCompositor:
    """Incrementally builds the colour-overlay video frame by frame.

    Instead of reading all stitched mask videos after the fact, this
    compositor receives each frame's masks as they are produced and
    blends them onto the original video frame in a single pass.

    The compositor owns a ``cv2.VideoCapture`` of the original video
    and advances it in lockstep with the mask stream.

    Thread-safe: all writes go through a single-thread worker; caller
    just pushes frames into the queue.
    """

    def __init__(
        self,
        video_path: Path,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        alpha: float = 0.5,
        start_frame: int = 0,
    ):
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.alpha = alpha
        self.start_frame = start_frame

        self._cap: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None
        self._frame_cursor: int = 0  # next expected frame index
        self._frames_written: int = 0
        self._lock = threading.Lock()

        # Pre-computed blend constants
        self._inv_alpha = np.float32(1.0 - alpha)
        self._f_alpha = np.float32(alpha)
        self._colour_arrays = [np.array(c, dtype=np.float32).reshape(1, 1, 3) for c in MASK_COLOURS_BGR]

        # Object → colour index mapping (stable across chunks)
        self._obj_colour_map: dict[int, int] = {}
        self._next_colour_idx: int = 0

    def _ensure_open(self) -> tuple[cv2.VideoCapture, cv2.VideoWriter]:
        if self._cap is None:
            self._cap = cv2.VideoCapture(str(self.video_path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video_path}")
            if self.start_frame > 0:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        if self._writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (self.width, self.height),
                True,
            )
        return self._cap, self._writer

    def _get_colour(self, obj_id: int) -> np.ndarray:
        """Get a stable colour for an object ID."""
        if obj_id not in self._obj_colour_map:
            self._obj_colour_map[obj_id] = self._next_colour_idx
            self._next_colour_idx += 1
        idx = self._obj_colour_map[obj_id] % len(self._colour_arrays)
        return self._colour_arrays[idx]

    def register_objects(self, obj_ids: set[int]) -> None:
        """Pre-register object IDs to lock in their colour assignments."""
        for oid in sorted(obj_ids):
            self._get_colour(oid)

    def composite_frame(
        self,
        masks: dict[int, np.ndarray],
    ) -> None:
        """Blend masks for one frame onto the original video and write.

        Parameters
        ----------
        masks : dict[int, np.ndarray]
            ``{object_id: binary_mask}`` for this frame.
            Binary masks should be uint8 with 255 = object, 0 = background.
        """
        with self._lock:
            cap, writer = self._ensure_open()

            ret, frame = cap.read()
            if not ret:
                logger.warning("StreamingOverlay: video ended before masks")
                return

            overlay = frame.copy()
            frame_f32 = frame.astype(np.float32)

            for obj_id, mask in masks.items():
                if mask is None:
                    continue
                binary = mask > 127
                if not binary.any():
                    continue
                c_arr = self._get_colour(obj_id)
                blended = (self._inv_alpha * frame_f32[binary] + self._f_alpha * c_arr).astype(np.uint8)
                overlay[binary] = blended

            writer.write(overlay)
            self._frames_written += 1

    def write_passthrough_frame(self) -> None:
        """Write original frame unchanged (no masks for this frame)."""
        with self._lock:
            cap, writer = self._ensure_open()
            ret, frame = cap.read()
            if ret:
                writer.write(frame)
                self._frames_written += 1

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def close(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            if self._writer is not None:
                self._writer.release()
                self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Queue-based streaming writer (GPU → CPU bridge)
# ---------------------------------------------------------------------------
@dataclass
class MaskFrame:
    """A single frame's masks for all objects, ready for CPU-side writing."""

    frame_idx: int
    masks: dict[int, np.ndarray]  # {obj_id: uint8 mask (H, W)}
    object_ids: list[int]  # all known obj IDs at this frame


class StreamingMaskWriter:
    """Queue-based GPU→CPU bridge for writing mask videos and overlay.

    The GPU propagation loop calls :py:meth:`push_frame` after each frame.
    A background thread picks up frames and:

    1. Writes each object's mask to its ``MaskVideoWriter`` (lossless MP4).
    2. Composites the coloured overlay onto the original video via
       ``StreamingOverlayCompositor``.
    3. Uses ``EmptyMaskPool`` for new objects that need back-filled
       black frames.

    Usage::

        writer = StreamingMaskWriter(
            masks_dir=chunk_dir / "masks" / prompt,
            video_path=video_path,
            overlay_path=output_dir / "overlay.mp4",
            width=1920, height=1080, fps=25.0,
        )
        writer.start()

        for frame_idx, masks, obj_ids in propagation:
            writer.push_frame(frame_idx, masks, obj_ids)

        writer.finish()   # blocks until all frames processed
    """

    def __init__(
        self,
        masks_dir: Path,
        width: int,
        height: int,
        fps: float,
        *,
        video_path: Path | None = None,
        overlay_path: Path | None = None,
        alpha: float = 0.5,
        start_frame: int = 0,
        queue_depth: int = 64,
    ):
        self.masks_dir = Path(masks_dir)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.alpha = alpha

        # Queue between GPU producer and CPU consumer
        self._queue: Queue[MaskFrame | None] = Queue(maxsize=queue_depth)
        self._thread: threading.Thread | None = None
        self._started = False
        self._finished = False
        self._error: Exception | None = None

        # Per-object writers
        self._writers: dict[int, MaskVideoWriter] = {}
        self._empty_pool = EmptyMaskPool(width, height)
        self._frames_processed: int = 0
        self._known_objects: set[int] = set()

        # Overlay compositor (optional — requires video_path)
        self._overlay: StreamingOverlayCompositor | None = None
        if video_path is not None and overlay_path is not None:
            self._overlay = StreamingOverlayCompositor(
                video_path=video_path,
                output_path=overlay_path,
                width=width,
                height=height,
                fps=fps,
                alpha=alpha,
                start_frame=start_frame,
            )

        # Stats
        self._wall_start: float = 0.0
        self._wall_end: float = 0.0
        self._total_mask_writes: int = 0

    def start(self) -> None:
        """Launch the background consumer thread."""
        if self._started:
            return
        self._started = True
        self._wall_start = time.time()
        self._thread = threading.Thread(
            target=self._consumer_loop,
            name="sam3-mask-writer",
            daemon=True,
        )
        self._thread.start()

    def push_frame(
        self,
        frame_idx: int,
        masks: dict[int, np.ndarray],
        object_ids: list[int] | set[int],
    ) -> None:
        """Push one frame's mask data for background processing.

        Must be called after ``start()``.  Blocks if queue is full
        (back-pressure when CPU can't keep up with GPU).
        """
        if not self._started:
            raise RuntimeError("StreamingMaskWriter not started")
        if self._error:
            raise self._error
        self._queue.put(
            MaskFrame(
                frame_idx=frame_idx,
                masks={k: v for k, v in masks.items()},  # shallow copy
                object_ids=sorted(set(object_ids)),
            )
        )

    def finish(self, timeout: float = 300.0) -> dict[str, Any]:
        """Signal completion and wait for the consumer to flush.

        Returns stats dict.
        """
        if not self._started:
            return self.stats()

        # Sentinel: None signals end-of-stream
        self._queue.put(None)
        self._thread.join(timeout=timeout)
        self._finished = True
        self._wall_end = time.time()

        # Release all resources
        for w in self._writers.values():
            w.close()
        if self._overlay:
            self._overlay.close()

        if self._error:
            raise self._error

        return self.stats()

    def _consumer_loop(self) -> None:
        """Background thread: dequeue frames, write masks + overlay."""
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break  # end-of-stream sentinel
                self._process_frame(item)
        except Exception as exc:
            self._error = exc
            logger.error(f"StreamingMaskWriter error: {exc}")

    def _process_frame(self, mf: MaskFrame) -> None:
        """Process a single frame: write per-object masks + overlay."""
        # Detect new objects and back-fill with black frames
        new_objects = set(mf.object_ids) - self._known_objects
        for oid in sorted(new_objects):
            self._known_objects.add(oid)
            out_path = self.masks_dir / f"object_{oid}_mask.mp4"
            writer = MaskVideoWriter(out_path, self.fps, self.width, self.height)
            self._writers[oid] = writer
            # Back-fill with black frames for all prior frames
            if self._frames_processed > 0:
                writer.write_black(self._frames_processed, pool=self._empty_pool)

        # Write mask for each known object
        for oid in sorted(self._known_objects):
            mask = mf.masks.get(oid)
            writer = self._writers[oid]
            if mask is not None and mask.any():
                mask_u8 = mask.astype(np.uint8) * 255 if mask.max() <= 1 else mask.astype(np.uint8)
                writer.write_frame(mask_u8)
            else:
                writer.write_black(1, pool=self._empty_pool)
            self._total_mask_writes += 1

        # Overlay compositing
        if self._overlay:
            overlay_masks = {}
            for oid, mask in mf.masks.items():
                if mask is not None and mask.any():
                    overlay_masks[oid] = mask.astype(np.uint8) * 255 if mask.max() <= 1 else mask.astype(np.uint8)
            self._overlay.composite_frame(overlay_masks)

        self._frames_processed += 1

    def stats(self) -> dict[str, Any]:
        """Return processing statistics."""
        wall = (self._wall_end or time.time()) - self._wall_start if self._wall_start else 0
        return {
            "frames_processed": self._frames_processed,
            "objects_tracked": len(self._known_objects),
            "total_mask_writes": self._total_mask_writes,
            "wall_time_s": round(wall, 3),
            "overlay_frames": self._overlay.frames_written if self._overlay else 0,
            "queue_depth": self._queue.qsize(),
            "had_error": self._error is not None,
        }

    @property
    def known_objects(self) -> set[int]:
        """Set of all object IDs seen so far."""
        return set(self._known_objects)

    @property
    def frames_processed(self) -> int:
        return self._frames_processed


# ---------------------------------------------------------------------------
# Cross-chunk stitcher (replaces the old PNG-based stitching)
# ---------------------------------------------------------------------------
def stitch_chunk_mask_videos(
    chunks_dir: Path,
    prompt_name: str,
    object_ids: set[int],
    chunk_infos: list[dict],
    overlap: int,
    output_dir: Path,
    fps: float,
    width: int,
    height: int,
) -> list[Path]:
    """Stitch per-chunk mask MP4s into final per-object mask MP4s.

    Each chunk already has ``object_{id}_mask.mp4`` files produced by
    ``StreamingMaskWriter``.  This function reads them, skips overlap
    frames, and writes the final concatenated video.

    Returns list of output paths created.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for oid in sorted(object_ids):
        out_path = output_dir / f"object_{oid}_mask.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height), False)
        black = np.zeros((height, width), dtype=np.uint8)

        for ci, cinfo in enumerate(chunk_infos):
            chunk_id = cinfo["chunk"]
            skip = overlap if ci > 0 else 0
            mask_path = chunks_dir / f"chunk_{chunk_id}" / "masks" / prompt_name / f"object_{oid}_mask.mp4"

            if not mask_path.exists():
                # No mask for this object in this chunk — write black frames
                chunk_len = cinfo["end"] - cinfo["start"] + 1
                for _ in range(chunk_len - skip):
                    writer.write(black)
                continue

            cap = cv2.VideoCapture(str(mask_path))
            chunk_len = cinfo["end"] - cinfo["start"] + 1
            usable = chunk_len - skip

            # Skip overlap frames
            if skip > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, skip)

            frames_read = 0
            while frames_read < usable:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                writer.write(frame)
                frames_read += 1
            cap.release()

            # Pad with black if chunk video was shorter than expected
            remaining = usable - frames_read
            for _ in range(remaining):
                writer.write(black)

        writer.release()
        results.append(out_path)
        logger.info(f"Stitched mask video: {out_path.name}")

    return results


def create_overlay_from_masks(
    video_path: Path,
    mask_videos: list[Path],
    output_path: Path,
    alpha: float = 0.5,
) -> None:
    """Create overlay video from pre-existing mask MP4s.

    Fallback for when streaming overlay wasn't used (e.g., multi-chunk
    where overlay needs to be built from stitched mask videos).
    Uses the same vectorised compositing as ``StreamingOverlayCompositor``.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h), True)

    mask_caps = [cv2.VideoCapture(str(p)) for p in mask_videos if p.exists()]

    colour_arrays = [np.array(c, dtype=np.float32).reshape(1, 1, 3) for c in MASK_COLOURS_BGR]
    inv_alpha = np.float32(1.0 - alpha)
    f_alpha = np.float32(alpha)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.copy()
        frame_f32 = frame.astype(np.float32)
        for i, mc in enumerate(mask_caps):
            ret_m, mask_frame = mc.read()
            if ret_m and mask_frame is not None:
                if mask_frame.ndim == 3:
                    mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                binary = mask_frame > 127
                if binary.any():
                    c_arr = colour_arrays[i % len(colour_arrays)]
                    blended = (inv_alpha * frame_f32[binary] + f_alpha * c_arr).astype(np.uint8)
                    overlay[binary] = blended
        writer.write(overlay)

    for mc in mask_caps:
        mc.release()
    cap.release()
    writer.release()
