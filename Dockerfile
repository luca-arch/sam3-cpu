# ──────────────────────────────────────────────────────────────
# Dockerfile — SAM3 CPU
#
# Stage 1: "builder" — install deps with uv (fast, cached)
# Stage 2: "runtime" — slim image with only what's needed
#
# Build:   docker build -t sam3-cpu .
# Run:     docker run -v ./assets:/app/assets -v ./results:/app/results sam3-cpu
# ──────────────────────────────────────────────────────────────

# ── Builder stage ──
FROM python:3.13-slim AS builder

# System deps needed to compile wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (maximises layer cache hits)
COPY pyproject.toml uv.lock* VERSION ./

# Install dependencies into a virtual env inside /app/.venv
RUN uv venv /app/.venv && \
    uv sync --no-dev --no-install-project

# Copy the rest of the source
COPY . .

# Install the project itself
RUN uv sync --no-dev


# ── Runtime stage ──
FROM python:3.13-slim AS runtime

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual env from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /app/sam3           /app/sam3
COPY --from=builder /app/video_prompter.py  /app/video_prompter.py
COPY --from=builder /app/image_prompter.py  /app/image_prompter.py
COPY --from=builder /app/main.py            /app/main.py
COPY --from=builder /app/config.json        /app/config.json
COPY --from=builder /app/VERSION            /app/VERSION
COPY --from=builder /app/pyproject.toml     /app/pyproject.toml
COPY --from=builder /app/examples           /app/examples

# Put the venv on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Default output directory
RUN mkdir -p /app/results /app/assets/videos /app/assets/images

# Volumes for input assets and output results
VOLUME ["/app/assets", "/app/results"]

# Health check — verify Python + sam3 import
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import sam3; print('ok')" || exit 1

# Default entrypoint — can be overridden
ENTRYPOINT ["python"]
CMD ["--help"]
