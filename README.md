# SAM3-CPU

**CPU-compatible wrapper around Meta's [Segment Anything Model 3 (SAM 3)](https://github.com/facebookresearch/sam3)** for image and video segmentation with intelligent memory management.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: SAM](https://img.shields.io/badge/License-SAM-orange.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/rhubarb-ai/sam3-cpu/pulls)

## Features

- **CPU + GPU** — runs on CPU out of the box; uses CUDA when available
- **Zero GPU footprint** — `--device cpu` hides GPUs completely (0 MiB VRAM used)
- **Configurable CPU utilisation** — `--cpu-utilisation 50..100` controls how many logical cores the model uses (default 100%)
- **Automatic bfloat16 on CPU** — detects AVX2/AVX512 and enables `torch.autocast` for 2–5× speedup on Intel 10th gen+ and AMD Zen 3+ processors
- **Memory-aware chunking** — automatically splits long videos into chunks sized to available RAM / VRAM
- **Cross-chunk continuity** — IoU-based mask remapping keeps object IDs consistent across chunks
- **Streaming MP4 mask pipeline** — per-object mask videos written in real-time via queue-based GPU→CPU bridge (~100× fewer I/O operations vs legacy per-frame PNGs)
- **Simultaneous overlay compositing** — colour overlay composited incrementally during propagation
- **Adaptive memory multiplier** — learns actual per-frame VRAM cost from chunk execution instead of relying on a hardcoded estimate
- **Text, point, box & mask prompts** — unified API for all prompt types
- **Video segment processing** — process a specific frame range or time range instead of the full video
- **Interval-based object tracking** — appearance / disappearance intervals with timestamps and timecodes for every detected object
- **Full cross-chunk IoU matrix** — complete pairwise IoU data for every chunk boundary
- **Built-in profiler** — `@profile()` decorators across the pipeline, enabled via `--profile` (zero overhead when off)
- **CLI tools** — `image_prompter.py` and `video_prompter.py` for quick experiments

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Device Selection](#device-selection)
- [Quick Start](#quick-start)
- [CLI Tools](#cli-tools)
  - [image\_prompter.py](#image_prompterpy)
  - [video\_prompter.py](#video_prompterpy)
- [Python API](#python-api)
- [CPU Performance Optimisations](#cpu-performance-optimisations)
  - [bfloat16 Autocast](#bfloat16-autocast)
  - [Thread Configuration](#thread-configuration)
  - [CPU Utilisation Control](#cpu-utilisation-control)
  - [Benchmark Summary](#benchmark-summary)
- [Video Chunking](#video-chunking)
  - [Memory Management Architecture](#memory-management-architecture)
  - [Streaming Mask Pipeline](#streaming-mask-pipeline)
  - [Adaptive Memory Multiplier](#adaptive-memory-multiplier)
- [Output Structure](#output-structure)
- [Metadata Reference](#metadata-reference)
  - [Schema Version](#schema-version)
  - [Video Metadata](#video-metadata-1)
  - [Image Metadata](#image-metadata)
  - [Object Tracking](#object-tracking)
  - [Cross-Chunk IoU](#cross-chunk-iou)
- [Profiling](#profiling)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)
- [Core Authors](#core-authors)

---

## Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.12+ |
| OS | Linux (Ubuntu 20.04+), macOS 11+ (Intel & Apple Silicon) |
| ffmpeg / ffprobe | Required for video processing |
| HuggingFace account | Model checkpoints are hosted on HuggingFace — you must [request access](https://huggingface.co/facebook/sam3) and authenticate (`huggingface-cli login`) before first use |

---

## Installation

### Automated (recommended)

```bash
git clone https://github.com/rhubarb-ai/sam3-cpu.git
cd sam3-cpu
chmod +x setup.sh && ./setup.sh   # or: make setup
```

### Manual

```bash
# Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e .
```

### System dependencies

**Linux (Debian / Ubuntu)**
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg python3-dev build-essential
```

**macOS**
```bash
brew install ffmpeg
```

### Verify

```bash
uv run python -c "from sam3 import Sam3; print('OK')"
```

---

## Device Selection

By default, SAM3-CPU auto-detects CUDA and uses the GPU when available.
Pass `--device cpu` to any CLI tool to force CPU execution:

```bash
# Image — force CPU
uv run python image_prompter.py --image img.jpg --prompts dog --device cpu

# Video — force CPU
uv run python video_prompter.py --video clip.mp4 --prompts person --device cpu
```

When `--device cpu` is specified **no GPU memory is allocated at all** — the
CUDA runtime context is never initialised, so `nvidia-smi` will show 0 MiB
used by the process.

---

## Quick Start

### Image segmentation (Python)

```python
from sam3 import Sam3

sam3 = Sam3()
result = sam3.process_image_with_prompts(
    image_path="assets/images/truck.jpg",
    prompts=["truck", "wheel"],
    output_dir="results/demo",
)
print(result)
```

### Video segmentation (Python)

```python
from sam3 import Sam3

sam3 = Sam3()
result = sam3.process_video_with_prompts(
    video_path="assets/videos/sample.mp4",
    prompts=["person", "tennis racket"],
    output_dir="results/demo",
)
```

### Makefile shortcuts

```bash
make help                 # show all available targets
make setup                # install uv + dependencies
make test                 # run pytest suite
make run-all              # run every example
make run-example EXAMPLE=a

# CLI tools via make
make image-prompter IMAGES='assets/images/truck.jpg' PROMPTS='truck wheel'
make video-prompter VIDEO='assets/videos/sample.mp4' PROMPTS='person'
make video-prompter VIDEO='clip.mp4' PROMPTS='player' FRAME_RANGE='100 500'
make video-prompter VIDEO='clip.mp4' PROMPTS='player' TIME_RANGE='0:05 0:30'
make video-prompter VIDEO='clip.mp4' PROMPTS='player' CPU_UTIL=75
```

All `make` variables:

| Variable | Used by | Example |
|---|---|---|
| `IMAGES` | `image-prompter` | `'img1.jpg img2.jpg'` |
| `VIDEO` | `video-prompter` | `'clip.mp4'` |
| `PROMPTS` | both | `'person car'` |
| `POINTS` | both | `'320,240 500,300'` |
| `POINT_LABELS` | both | `'1 0'` |
| `BBOX` | `image-prompter` | `'100 50 400 300'` |
| `MASKS` | `video-prompter` | `'mask.png'` |
| `FRAME_RANGE` | `video-prompter` | `'100 500'` |
| `TIME_RANGE` | `video-prompter` | `'0:05 0:30'` |
| `OUTPUT` | both | `'results/demo'` |
| `ALPHA` | both | `0.45` |
| `DEVICE` | both | `cpu` or `cuda` |
| `CPU_UTIL` | both | `75` (use 75% of logical cores) |
| `CHUNK_SPREAD` | `video-prompter` | `even` |
| `KEEP_TEMP` | `video-prompter` | `1` (any non-empty value) |
| `MAX_VRAM_GB` | `video-prompter` | `10` (cap VRAM budget in GB) |
| `MAX_RAM_GB` | `video-prompter` | `16` (cap RAM budget in GB) |
| `VIDEO_RES` | `run-*` examples | `480p`, `720p`, `1080p` |
| `ARGS` | all | extra flags passed through |

---

## CLI Tools

Two standalone scripts provide a quick way to run segmentation from the terminal
without writing Python code.

### image\_prompter.py

Segment one or more images with text prompts, click points, or bounding boxes.

#### Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--images` | `str+` | *(required)* | One or more image file paths |
| `--prompts` | `str+` | `None` | Text prompts (e.g. `person car`) |
| `--points` | `str+` | `None` | Click points as `x,y` pairs (e.g. `320,240 500,300`) |
| `--point-labels` | `int+` | all `1` | Labels for each point (`1` = positive, `0` = negative) |
| `--bbox` | `float+` | `None` | Bounding box(es) as `x y w h` (multiples of 4 for several boxes) |
| `--output` | `str` | `results` | Output directory |
| `--alpha` | `float` | `0.5` | Overlay alpha for mask visualisation (`0.0`–`1.0`) |
| `--device` | `str` | auto | Force `cpu` or `cuda` (auto-detected if omitted) |
| `--cpu-utilisation` | `int` | `100` | Percentage of logical CPU cores to use (50–100) |
| `--profile` | flag | off | Enable the built-in profiler |

At least one of `--prompts`, `--points`, or `--bbox` is required.

#### Examples

```bash
# Text prompt — segment truck and wheels
uv run python image_prompter.py \
    --images assets/images/truck.jpg \
    --prompts "truck" "wheel" \
    --output results/truck_demo

# Bounding-box prompt — segment inside a rectangle
uv run python image_prompter.py \
    --images assets/images/truck.jpg \
    --bbox 100 50 400 300 \
    --output results/truck_bbox

# Batch: multiple images, 75% CPU utilisation
uv run python image_prompter.py \
    --images img1.jpg img2.jpg img3.jpg \
    --prompts "person" \
    --device cpu --cpu-utilisation 75
```

### video\_prompter.py

Segment a video with text prompts, click points, or binary masks.  Supports
automatic memory-aware chunking, segment processing (frame or time ranges),
and generates per-object tracking metadata.

#### Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--video` | `str` | *(required)* | Input video file |
| `--prompts` | `str+` | `None` | Text prompts (e.g. `person ball`) |
| `--points` | `str+` | `None` | Click points as `x,y` pairs |
| `--point-labels` | `int+` | all `1` | Labels for each point (`1` = positive, `0` = negative) |
| `--masks` | `str+` | `None` | Binary mask image(s) for initial object prompts |
| `--output` | `str` | `results` | Output directory |
| `--alpha` | `float` | `0.5` | Overlay alpha (`0.0`–`1.0`) |
| `--device` | `str` | auto | Force `cpu` or `cuda` |
| `--cpu-utilisation` | `int` | `100` | Percentage of logical CPU cores to use (50–100) |
| `--chunk-spread` | `str` | `default` | Chunk size strategy: `default` or `even` |
| `--keep-temp` | flag | off | Preserve intermediate chunk files in output |
| `--frame-range` | `int int` | `None` | Process only frames `START..END` (0-based, inclusive) |
| `--time-range` | `str str` | `None` | Process a time segment (seconds, `MM:SS`, or `HH:MM:SS`) |
| `--max-vram-gb` | `float` | `None` | Cap VRAM budget (GB) |
| `--max-ram-gb` | `float` | `None` | Cap RAM budget (GB) |
| `--profile` | flag | off | Enable the built-in profiler |

At least one of `--prompts`, `--points`, or `--masks` is required.
`--frame-range` and `--time-range` are mutually exclusive.

#### Examples

```bash
# Text prompt — segment all people and tennis rackets
uv run python video_prompter.py \
    --video assets/videos/sample.mp4 \
    --prompts "player" "tennis racket" \
    --output results/tennis_demo

# Frame-range — process only frames 100 to 500
uv run python video_prompter.py \
    --video match.mp4 \
    --prompts "player" \
    --frame-range 100 500

# Time-range with MM:SS notation
uv run python video_prompter.py \
    --video match.mp4 \
    --prompts "player" \
    --time-range 0:05 0:30

# CPU at 75% utilisation
uv run python video_prompter.py \
    --video clip.mp4 \
    --prompts "person" \
    --device cpu --cpu-utilisation 75

# Even chunk spread + preserve temp files
uv run python video_prompter.py \
    --video clip.mp4 \
    --prompts "person" \
    --chunk-spread even --keep-temp
```

---

## Python API

The high-level entry point is `Sam3` (alias for `Sam3API`):

```python
from sam3 import Sam3
sam3 = Sam3()
```

### Image scenarios

| Method | Description |
|---|---|
| `process_image_with_prompts(image_path, prompts, ...)` | Segment by text prompts |
| `process_image_with_boxes(image_path, boxes, ...)` | Segment inside bounding boxes |

### Video scenarios

| Method | Description |
|---|---|
| `process_video_with_prompts(video_path, prompts, ...)` | Segment & track by text prompts |

Video processing supports:

- `propagation_direction` — `"forward"`, `"backward"`, or `"both"` (default)
- Automatic chunking when memory is limited (see below)

Lower-level access is available through `ImageProcessor`, `VideoProcessor`,
`ChunkProcessor`, and the driver classes in `sam3/drivers.py`.

---

## CPU Performance Optimisations

SAM3-CPU includes several optimisations that significantly improve inference
speed on modern x86 CPUs.  The key findings below were measured on an
Intel Xeon 8573C (8 physical / 16 logical cores, 62 GB RAM).

### bfloat16 Autocast

The model runs under `torch.autocast("cpu", dtype=torch.bfloat16)` on any CPU
that reports **AVX2** or **AVX512** capabilities via
`torch.backends.cpu.get_cpu_capability()`.

| Capability | CPU families covered |
|---|---|
| **AVX512** | Intel Xeon Sapphire Rapids+ (AMX-BF16, native 2–5× speedup), Ice Lake+, AMD EPYC Genoa / Ryzen 7000+ (Zen 4/5) |
| **AVX2** | Intel 12th–15th Gen (Alder Lake+), AMD Ryzen 5000+ (Zen 3), Apple M-series via Rosetta 2 |

On AVX512 CPUs with AMX-BF16 (e.g. Sapphire Rapids Xeon), bfloat16 matmuls
are executed natively in the AMX tile accelerator, yielding a **2–5× per-op
speedup**.  On AVX2 CPUs, PyTorch's oneDNN backend performs bf16→fp32
conversion using VNNI or software fallback — still faster than pure fp32 due
to **halved memory bandwidth**.

Very old CPUs (SSE-only, "DEFAULT" capability tier) remain on fp32 automatically.

### Thread Configuration

SAM3 uses **all logical cores** (including hyper-threading siblings) for
`torch.set_num_threads()`, rather than limiting to physical cores.  This is
because SAM3's workload mixes compute-bound operations (matmul, convolution)
with memory-bound operations (attention, elementwise), allowing HT siblings to
hide memory latency from each other.

**Benchmark** (Xeon 8573C, 8 physical / 16 logical, bfloat16 autocast):

| Threads | Mode | Composite latency | Speedup |
|---|---|---|---|
| 8 (physical only) | BF16 | 20.1 ms | baseline |
| 16 (logical, HT) | BF16 | 15.5 ms | **23% faster** |
| 8 (physical only) | FP32 | 23.2 ms | — |
| 16 (logical, HT) | FP32 | 21.2 ms | 9% faster |

The HT benefit is largest in BF16 mode because AMX/VNNI instructions free ALU
cycles that the sibling thread can use.

> **Combined effect of BF16 + HT on propagation:** 8.02 s/frame → 2.69 s/frame
> — a **2.98× speedup** on the per-frame propagation hot loop.

### CPU Utilisation Control

The `--cpu-utilisation` flag (50–100, default 100) scales the number of
logical cores used:

```
threads = max(1, logical_cores × cpu_utilisation / 100)
```

This is useful when:
- Running on a shared server and you want to leave cores for other processes
- Reducing thermal throttling on laptops
- Benchmarking the effect of thread count on throughput

```bash
# Use 75% of logical cores
uv run python video_prompter.py --video clip.mp4 --prompts person --cpu-utilisation 75

# Via Makefile
make video-prompter VIDEO='clip.mp4' PROMPTS='person' CPU_UTIL=75
```

### Benchmark Summary

End-to-end results on Intel Xeon 8573C, 62 GB RAM, CPU-only:

| Optimisation | Before | After | Improvement |
|---|---|---|---|
| bfloat16 autocast (AVX512+AMX) | fp32 baseline | bf16 | ~2.5× per-op |
| Logical cores (HT enabled) | 8 threads | 16 threads | 23% latency reduction |
| Combined (BF16 + HT) | 8.02 s/frame | 2.69 s/frame | **2.98×** |

---

## Video Chunking

When a video is too large to fit in memory the framework automatically splits it
into overlapping chunks, processes each chunk independently, and stitches the
results back together.

**How it works:**

1. `MemoryManager` computes how many frames fit in available RAM (CPU) or VRAM (GPU).
2. The video is split into chunks with configurable overlap (default 1 frame).
3. Each chunk is segmented and tracked independently.
4. At chunk boundaries, masks from the overlap region are matched using IoU and
   object IDs are remapped so they stay consistent across the full video.

**Key parameters** (set in `config.json` or `sam3/__globals.py`):

| Parameter | Default | Meaning |
|---|---|---|
| `ram_usage_percent` | 0.975 | Fraction of free RAM budget for frames |
| `min_frames` | 25 | Minimum frames per chunk |
| `chunk_overlap` | 1 | Overlap frames between chunks |
| `CHUNK_MASK_MATCHING_IOU_THRESHOLD` | 0.75 | IoU threshold for cross-chunk ID matching |

### Memory Management Architecture

SAM3 uses a **two-tier adaptive memory management system** that combines
proactive intra-chunk monitoring with reactive inter-chunk adaptation:

```
┌─────────────────────────────────────────────────────────────────┐
│                    VIDEO PROCESSING PIPELINE                     │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  Tier 1: IntraChunkMonitor (proactive, per-frame)        │  │
│   │                                                          │  │
│   │  Phase 1 — CALIBRATE (first 5 frames)                    │  │
│   │  ├─ Sample VRAM after each frame                         │  │
│   │  ├─ Fit linear growth model (slope, R²)                  │  │
│   │  └─ Predict safe frame count                             │  │
│   │                                                          │  │
│   │  Phase 2 — PROGRESSIVE CHECKPOINTS                       │  │
│   │  ├─ Check at N/2, 3N/4, 7N/8, … iterations              │  │
│   │  ├─ Verify predictions against observed data             │  │
│   │  └─ ~10µs overhead per checkpoint                        │  │
│   │                                                          │  │
│   │  Phase 3 — HARD STOP (≥ 95% VRAM)                       │  │
│   │  ├─ Immediate propagation halt                           │  │
│   │  ├─ Use calibration for smart replan                     │  │
│   │  └─ Resume from stop point in next chunk                 │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  Tier 2: AdaptiveChunkManager (reactive, post-chunk)     │  │
│   │                                                          │  │
│   │  After each chunk completes:                             │  │
│   │  ├─ Classify pressure: NORMAL / ELEVATED / WARNING /     │  │
│   │  │   CRITICAL based on peak VRAM usage                   │  │
│   │  ├─ SHRINK chunk size if WARNING or CRITICAL             │  │
│   │  ├─ GROW chunk size if under-utilised (< 50%)            │  │
│   │  └─ CONTINUE if NORMAL or ELEVATED                       │  │
│   │                                                          │  │
│   │  OOM recovery (safety net):                              │  │
│   │  ├─ 40% aggressive reduction on actual OOM               │  │
│   │  ├─ Max 3 consecutive retries                            │  │
│   │  └─ Falls back if proactive monitoring misses            │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  Async I/O Pipeline                                      │  │
│   │                                                          │  │
│   │  ┌──────────┐  submit()  ┌───────────────┐              │  │
│   │  │ Compute  │──────────>│ AsyncIOWorker  │              │  │
│   │  │ Thread   │           │ (1 thread)     │              │  │
│   │  │          │  overlap  │                │              │  │
│   │  │ Next     │<────────>│ Write masks    │              │  │
│   │  │ prompt   │           │ Write JSON     │              │  │
│   │  └──────────┘  drain()  └───────────────┘              │  │
│   │                before stitching                          │  │
│   └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Key files:**

| Module | Purpose |
|---|---|
| `sam3/memory_optimizer.py` | `IntraChunkMonitor`, `AdaptiveChunkManager`, `AdaptiveMultiplier` |
| `sam3/streaming_masks.py` | `StreamingMaskWriter`, `MaskVideoWriter`, `EmptyMaskPool`, `StreamingOverlayCompositor` |
| `sam3/async_io.py` | `AsyncIOWorker` — background thread for disk writes |
| `sam3/memory_manager.py` | Static chunk planning (`compute_memory_safe_frames`) |
| `sam3/memory_predictor.py` | Background OOM predictor (soft/hard stop callbacks) |

**Memory pressure thresholds:**

| Level | Usage % | Action |
|---|---|---|
| NORMAL | < 60% | May grow chunk size |
| ELEVATED | 60–85% | Keep current size |
| WARNING | 85–95% | Reduce chunk size |
| CRITICAL | ≥ 95% | Aggressively reduce |

### Streaming Mask Pipeline

The **streaming mask pipeline** (`sam3/streaming_masks.py`) replaces the legacy
per-frame PNG approach with real-time MP4 encoding.  For a 1080p video with
1125 frames and 3 objects, this reduces the output from **3,375 PNG files** to
just **3 MP4 files**.

**Architecture:**

```
┌──────────────┐    Queue     ┌────────────────────────────────┐
│  Compute     │─────────────>│  StreamingMaskWriter           │
│  Thread      │  MaskFrame   │  (background consumer thread)  │
│  (propagate) │  dataclass   │                                │
│              │              │  ├─ MaskVideoWriter per object │
│  push_frame()│              │  │  (mp4v, grayscale, lossless) │
│              │              │  │                              │
│              │              │  ├─ EmptyMaskPool               │
│              │              │  │  (shared read-only black     │
│              │              │  │   frame, zero-copy)          │
│              │              │  │                              │
│              │              │  └─ StreamingOverlayCompositor  │
│              │              │     (incremental colour overlay)│
└──────────────┘   finish()   └────────────────────────────────┘
                   ─────>       flush + close all writers
```

**Performance gains:**
- ~100× fewer I/O operations (1 MP4 per object vs. N PNGs per object)
- Compute and I/O work in parallel via the queue bridge
- Zero-copy black frame pool eliminates allocation overhead for new objects
- Overlay compositing happens during propagation, not as a separate pass

### Adaptive Memory Multiplier

The initial chunk size is computed using `MODEL_STATE_MULTIPLIER` (default 4.5).
The **`AdaptiveMultiplier`** (`sam3/memory_optimizer.py`) replaces this static
heuristic after the first chunk by learning from actual execution:

- **Zero overhead on chunk 0** — falls back to static multiplier
- **Self-correcting** — rolling window of last 5 chunks, weighted by R² confidence
- **Reject bad data** — samples with R² < 0.3 or negative growth are silently dropped
- **Exported in metadata** — `adaptive_multiplier` section in `metadata.json`

---

## Output Structure

### image\_prompter.py

```
results/images/
├── pipeline_metadata.json          # pipeline-level timing, memory, thread config
└── <image_name>/
    ├── metadata.json               # enriched per-image metadata (schema v2.0.0)
    ├── <prompt>/
    │   ├── object_0_mask.png
    │   ├── object_0_overlay.png
    │   └── metadata.json
    ├── bbox/
    │   └── ...
    └── points/
        └── ...
```

### video\_prompter.py

```
results/<video_name>/
├── metadata.json                   # enriched run metadata (schema v2.2.0)
├── overlay_<prompt>.mp4            # coloured overlay on original video
├── masks/
│   └── <prompt>/
│       ├── object_0_mask.mp4       # binary mask video per object (lossless MP4)
│       └── object_1_mask.mp4
├── metadata/
│   ├── video_metadata.json         # fps, resolution, frame count, chunk plan
│   ├── memory_info.json            # pre-run RAM / VRAM budget analysis
│   ├── object_tracking.json        # per-object temporal intervals
│   └── cross_chunk_iou.json        # full pairwise IoU matrices (multi-chunk only)
└── temp_files/                     # only when --keep-temp is set
    └── chunks/
        └── chunk_<id>/
            └── masks/<prompt>/object_<id>_mask.mp4
```

---

## Metadata Reference

All metadata files follow a versioned schema so downstream tools can adapt
to format changes.

### Schema Version

| Field | Type | Description |
|---|---|---|
| `schema_version` | `string` | `"2.2.0"` for video, `"2.0.0"` for image |
| `sam3_version` | `string` | SAM3-CPU release version |

### Video Metadata

The top-level `metadata.json` contains the complete run record:

```jsonc
{
  "schema_version": "2.2.0",
  "video": "assets/videos/sample.mp4",
  "resolution": "854x480",
  "total_frames": 200,
  "fps": 25.0,
  "device": "cpu",
  "thread_config": {
    "intra_op_threads": 16,
    "inter_op_threads": 1,
    "cpu_utilisation_pct": 100
  },
  "timing": {
    "total_s": 35.121,
    "model_load_s": 6.944,
    "chunk_processing_s": 26.538,
    "stitching_s": 0.078
  },
  "memory": {
    "peak_rss_bytes": 8008626176,
    "min_rss_bytes": 957571072
  }
}
```

### Image Metadata

Per-image `metadata.json`:

```jsonc
{
  "schema_version": "2.0.0",
  "image_name": "truck",
  "resolution": "1800x1200",
  "device": "cpu",
  "timing": { "image_total_s": 12.34 },
  "objects": {
    "text_prompts": [
      {
        "prompt": "truck",
        "num_objects": 1,
        "objects": [
          {
            "object_id": 0,
            "score": 0.866,
            "mask_area_pct": 29.46,
            "mask_file": "object_0_mask.png",
            "overlay_file": "object_0_overlay.png"
          }
        ]
      }
    ]
  }
}
```

### Object Tracking

Each object's temporal presence is modelled as a list of **contiguous
intervals** — an object that disappears and reappears gets multiple
intervals.  All timestamps account for `--frame-range` / `--time-range`
offsets so they refer to the **original** video timeline.

```json
{
  "person": [
    {
      "object_id": 0,
      "intervals": [
        {
          "start_frame": 12,
          "end_frame": 187,
          "start_timecode": "00:00:00.480",
          "end_timecode": "00:00:07.480",
          "duration_frames": 176,
          "duration_s": 7.04
        }
      ],
      "total_frames_active": 176,
      "first_frame": 12,
      "last_frame": 187
    }
  ]
}
```

This data is also saved separately as `metadata/object_tracking.json`.

### Cross-Chunk IoU

When a video is processed in multiple chunks, the full pairwise IoU matrix
for every chunk boundary is captured in `metadata/cross_chunk_iou.json`:

```json
{
  "chunk_0_to_1": {
    "person": {
      "matrix": { "0": { "0": 0.91, "1": 0.02 } },
      "matched": { "0": 0 },
      "threshold": 0.25
    }
  }
}
```

---

## Profiling

SAM3-CPU ships with a built-in **profiler** that measures wall-clock execution
time and memory consumption (RSS) for any decorated function.

### Instrumented functions

The following functions have `@profile()` decorators (zero cost when profiling
is disabled):

| Module | Functions |
|---|---|
| `model_builder.py` | `build_sam3_video_model` |
| `drivers.py` | `_build_model`, `_get_predictor`, `inference`, `start_session`, `propagate_in_video` |
| `video_processor.py` | `process_with_prompts`, `_create_chunk_plan`, `_process_single_chunk_video`, `_process_multiple_chunks`, `_postprocess_results` |
| `chunk_processor.py` | `process_with_prompts`, `_process_single_prompt` |
| `image_processor.py` | `process_with_prompts`, `_process_single_image_with_prompts`, `_process_single_prompt`, `process_with_boxes` |
| `postprocessor.py` | `process`, `_build_id_mappings`, `_stitch_masks_for_prompt` |
| `memory_manager.py` | `compute_memory_safe_frames`, `chunk_plan_video` |

Additionally, `sam3/model/sam3_video_base.py` logs **per-frame 5-step timing**
at `DEBUG` level:

```
[perf] frame 42: backbone+det=890ms  tracker_prop=320ms  update_plan=15ms  update_exec=1200ms  build_out=45ms  total=2470ms
```

### Enabling the profiler

```bash
# Profile image segmentation
uv run python image_prompter.py \
    --images assets/images/truck.jpg --prompts truck --profile

# Profile video segmentation
uv run python video_prompter.py \
    --video clip.mp4 --prompts person --profile
```

Or toggle programmatically:

```python
import sam3.__globals
sam3.__globals.ENABLE_PROFILING = True
```

### Output files

| File | Format | Content |
|---|---|---|
| `profile_results.json` | JSON array | One object per call: `function_name`, `timestamp`, `execution_time_seconds`, `memory_used_MB`, `total_process_memory_MB` |
| `profile_results.txt` | Plain text | One line per call — human-readable summary |

A standalone demo: `uv run python examples/profiler_example.py --profile`

---

## Configuration

Runtime defaults live in `config.json` and compile-time constants in
`sam3/__globals.py`.  Key settings:

| Setting | Default | Description |
|---|---|---|
| `ram_usage_percent` | 0.975 | Fraction of free RAM used for chunk planning |
| `min_frames` | 25 | Minimum frames per chunk |
| `chunk_overlap` | 1 | Overlap frames between chunks |
| `DEFAULT_CPU_UTILISATION` | 100 | Default CPU utilisation % (overridden by `--cpu-utilisation`) |
| `MODEL_STATE_MULTIPLIER` | 4.5 | Per-frame memory cost multiplier |
| `VRAM_HARD_LIMIT_PCT` | 0.975 | Runtime VRAM hard stop threshold |
| `RAM_HARD_LIMIT_PCT` | 0.975 | Runtime RAM hard stop threshold |

---

## Project Structure

```
sam3-cpu/
├── image_prompter.py          # CLI – image segmentation
├── video_prompter.py          # CLI – video segmentation
├── main.py                    # Simple CLI entry point
├── config.json                # Runtime configuration
├── setup.sh / Makefile        # Build helpers
│
├── sam3/                      # Core package
│   ├── api.py                 # Sam3 high-level API
│   ├── drivers.py             # Sam3ImageDriver / Sam3VideoDriver
│   ├── image_processor.py     # ImageProcessor
│   ├── video_processor.py     # VideoProcessor
│   ├── chunk_processor.py     # ChunkProcessor (cross-chunk logic)
│   ├── postprocessor.py       # VideoPostProcessor
│   ├── memory_manager.py      # MemoryManager (static chunk planning)
│   ├── memory_optimizer.py    # IntraChunkMonitor, AdaptiveChunkManager
│   ├── memory_predictor.py    # Background OOM predictor
│   ├── streaming_masks.py     # StreamingMaskWriter, MaskVideoWriter
│   ├── async_io.py            # AsyncIOWorker (background disk writes)
│   ├── model_builder.py       # Model loading
│   ├── __globals.py           # Constants & defaults
│   ├── utils/                 # Utility modules
│   │   ├── logger.py
│   │   ├── profiler.py
│   │   ├── memory_sampler.py
│   │   ├── system_info.py
│   │   ├── ffmpeglib.py
│   │   └── visualization.py
│   ├── model/                 # SAM 3 model definitions
│   └── sam/                   # SAM core modules
│
├── examples/                  # Runnable example scripts (a–i)
├── notebook/                  # Jupyter notebooks
├── tests/                     # Pytest test suite
├── scripts/                   # Utility scripts
├── assets/                    # Sample images & videos
└── README.md
```

---

## Testing

The test suite lives in `tests/` and uses **pytest**.  Tests run
**without the SAM3 model** — they exercise helper functions, IoU logic,
stitching, and metadata generation using synthetic data.

```bash
# Full suite
uv run python -m pytest tests/ -v

# Fast tests only (skip model-dependent)
uv run python -m pytest tests/ -v -m "not slow"

# Single file
uv run python -m pytest tests/test_video_prompter.py -v
```

Available markers: `@pytest.mark.slow`, `@pytest.mark.gpu`,
`@pytest.mark.image`, `@pytest.mark.video`.

---

## Known Limitations

- **Cross-chunk object ID reassignment** — If an object disappears mid-chunk
  and reappears in a later chunk with no overlapping mask at the boundary, it
  gets a new ID.

- **CPU inference speed** — Running the full SAM 3 model on CPU is
  significantly slower than GPU even with bfloat16 + HT optimisations.
  Use `--frame-range` / `--time-range` to process only the segment you need.

- **macOS / Windows** — Tested primarily on Linux.  macOS works for most
  workflows; Windows support is not yet validated.

---

## Contributing

Contributions are welcome!

1. **Fork** the repository on GitHub.
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/my-improvement
   ```
3. **Make your changes** and add tests where appropriate.
4. **Open a Pull Request** against `main` with a clear description.

### Guidelines

- Follow existing code style and conventions.
- Keep PRs focused — one logical change per PR.
- Ensure `uv run python -m pytest tests/ -v` passes before submitting.
- Update documentation (especially this README) for user-facing changes.

---

## Future Work

- **Docker support** — `Dockerfile` + `docker-compose.yml` for one-command deployment.
- **Full macOS / Windows compatibility** — Validate on Intel Mac, Apple Silicon, and WSL.
- **Model quantisation** — INT8 / INT4 quantisation for further CPU speedup.
- **H.265 mask encoding** — HEVC for smaller mask videos while maintaining lossless quality.
- **Real-time preview** — Leverage the streaming mask pipeline for live preview during processing.
- **Multi-GPU batch processing** — Distribute chunks across GPUs for linear speed-up.

---

## Citation

If you use this project in your research or applications, please cite **both**
the original SAM 3 paper and this repository.

### SAM 3 (Meta)

```bibtex
@misc{carion2025sam3segmentconcepts,
    title   = {SAM 3: Segment Anything with Concepts},
    author  = {Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and
               Shoubhik Debnath and Ronghang Hu and Didac Suris and
               Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr
               and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo
               and Arpit Kalla and Markus Marks and Joseph Greer and
               Meng Wang and Peize Sun and Roman Rädle and
               Triantafyllos Afouras and Effrosyni Mavroudi and
               Katherine Xu and Tsung-Han Wu and Yu Zhou and
               Liliane Momeni and Rishi Hazra and Shuangrui Ding and
               Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li
               and Aishwarya Kamath and Ho Kei Cheng and Piotr Dollár
               and Nikhila Ravi and Kate Saenko and Pengchuan Zhang
               and Christoph Feichtenhofer},
    year    = {2025},
    eprint  = {2511.16719},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV},
    url     = {https://arxiv.org/abs/2511.16719},
}
```

### SAM3-CPU

```bibtex
@misc{aparajeya2026sam3cpu,
    title  = {SAM3-CPU: Segment Anything with Concepts — CPU-compatible
              inference with memory-aware chunking},
    author = {Prashant Aparajeya, Ankuj Arora},
    year   = {2026},
    url    = {https://github.com/rhubarb-ai/sam3-cpu},
}
```

---

## License

This project is released under the **SAM License** — see [LICENSE](LICENSE) for
the full text.  The license covers both the wrapper code and the underlying
SAM 3 model weights.

---

## Core Authors

**Dr Prashant Aparajeya**
Email: p.aparajeya@gmail.com
GitHub: [rhubarb-ai/sam3-cpu](https://github.com/rhubarb-ai/sam3-cpu)
Github Profile: [paparajeya](https://github.com/paparajeya)

**Dr Ankuj Arora**
Email: ankujarora@gmail.com
GitHub: [rhubarb-ai/sam3-cpu](https://github.com/rhubarb-ai/sam3-cpu)
Github Profile: [ankuj](https://github.com/ankuj)
