import os
import sys

import torch
import sam3
from sam3.utils.logger import get_logger, LOG_LEVELS

os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"

LOG_LEVEL = LOG_LEVELS["DEBUG"]
logger = get_logger(__name__, level=LOG_LEVEL)

logger.info(f"Python executable: {sys.executable}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check for --profile flag
if '--profile' in sys.argv:
    ENABLE_PROFILING = True
    logger.info("🔍 Profiling ENABLED\n")
else:
    ENABLE_PROFILING = False
    logger.info("⚡ Profiling DISABLED (use --profile to enable)\n")

SAM3_ROOT = os.path.join(os.path.dirname(sam3.__file__))
BPE_PATH = os.path.join(SAM3_ROOT, "assets/bpe_simple_vocab_16e6.txt.gz")

# Video processing defaults
DEFAULT_MIN_VIDEO_FRAMES = 25
DEFAULT_MIN_CHUNK_OVERLAP = 1

SUPPORTED_VIDEO_FORMATS = ('.mp4', '.avi', '.mov', '.mkv')

# Memory management
IMAGE_INFERENCE_MB = 6760
VIDEO_INFERENCE_MB = 6900
TENSOR_SIZE_BYTES = 1008*1008*3*4 # Approximate size of a 1008x1008 RGB tensor in bytes

# Model state overhead: the SAM3 tracker stores per-frame feature maps, cached
# masks for N tracked objects, positional encodings, and memory-bank tensors.
# The multiplier is applied on top of the raw pixel cost (width × height × 4)
# to estimate the full per-frame GPU memory footprint.
# Calibrated empirically: 480p → ~40 MB/frame, 1080p → ~70 MB/frame.
MODEL_STATE_MULTIPLIER = 4.5

# Memory usage for chunking (percentage of available memory to use)
RAM_USAGE_PERCENT = 0.975   # Use 97.5% of available RAM for CPU video chunking (conservative)
VRAM_USAGE_PERCENT = 0.975  # Use 97.5% of available VRAM for initial GPU chunk planning
                            # (the IntraChunkMonitor enforces tighter per-frame
                            # limits during actual propagation — see below)
CPU_CORES_PERCENT = 0.90   # Use 90% of CPU cores for parallel processing (leave some for OS and other tasks)

MEMORY_SAFETY_MULTIPLIER = 1.5  # Require 1.5x estimated memory for safety (reduced from 3x)
CPU_MEMORY_RESERVE_PERCENT = 0.3  # Reserve 30% for OS
GPU_MEMORY_RESERVE_PERCENT = 0.05  # Reserve 5% for display

# Intra-chunk memory guard thresholds (enforced per-frame during propagation)
# These are the live safety limits.  VRAM_USAGE_PERCENT above is only for
# initial chunk *planning*; the guard below is the runtime enforcement.
VRAM_SOFT_LIMIT_PCT = 0.85   # Warn + predict frames-to-limit
VRAM_HARD_LIMIT_PCT = 0.975   # Immediate stop — 2.5% headroom prevents actual OOM

# RAM guard thresholds (same dual-threshold design as VRAM)
RAM_SOFT_LIMIT_PCT = 0.85    # Warn when process RSS reaches 85% of available RAM
RAM_HARD_LIMIT_PCT = 0.975    # Immediate stop — leave 2.5% headroom for OS/other

# Parallel processing
PARALLEL_CHUNK_THRESHOLD = 0.90  # Start loading next chunk at 90% completion

# Output settings
DEFAULT_PROPAGATION_DIRECTION = "both"
DEFAULT_NUM_WORKERS = 1  # Use all available CPU cores by default
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Post-processing settings
CHUNK_MASK_MATCHING_IOU_THRESHOLD = 0.75  # IoU threshold for matching masks across chunks (75% - expecting high values with lossless PNG storage)

# Directory settings
TEMP_DIR = "/tmp/sam3-cpu" if DEVICE.type == "cpu" else "/tmp/sam3-gpu"
os.makedirs(TEMP_DIR, exist_ok=True)

DEFAULT_OUTPUT_DIR = os.path.join("./results")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

PROFILE_OUTPUT_JSON = "profile_results.json"
PROFILE_OUTPUT_TXT = "profile_results.txt"