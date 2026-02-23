# =============================================================================
# Video-to-3DGS Pipeline — Docker Image
# Includes: MASt3R-SfM, FastMap, COLMAP, Brush trainer, ffmpeg
#
# GPU support:
#   CUDA (MASt3R/PyTorch): Pascal 6.1 → Blackwell 10.0  (all NVIDIA generations)
#   Vulkan (Brush):         NVIDIA / AMD / Intel          (any Vulkan 1.1+ GPU)
#
# Build (from /home/ibrahim — the parent of gaussian-splatting/):
#   docker build -f gaussian-splatting/Dockerfile -t video-to-3dgs .
#
# Run:
#   docker run --gpus all -v /path/to/data:/data video-to-3dgs \
#       --video /data/video.mp4 --output /data/output --sfm-backend mast3r
# =============================================================================


# =============================================================================
# Stage 1 — builder: compile all CUDA extensions
# =============================================================================
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# All NVIDIA GPU generations from Pascal (GTX 10xx) through Blackwell (B200)
# +PTX at the end enables JIT compilation for future architectures
ENV TORCH_CUDA_ARCH_LIST="6.1 7.0 7.5 8.0 8.6 8.9 9.0 10.0+PTX"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"

# ── Build tools ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git cmake build-essential ninja-build \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# ── PyTorch — cu128 supports sm_60+ (Pascal and up) ───────────────────────
RUN pip3 install --no-cache-dir \
        torch==2.7.0 torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cu128

# ── 3DGS CUDA extensions ───────────────────────────────────────────────────
# Copy submodules first (heaviest cached layer)
COPY gaussian-splatting/submodules /app/gaussian-splatting/submodules

WORKDIR /app/gaussian-splatting
RUN pip3 install --no-cache-dir --no-build-isolation \
        submodules/diff-gaussian-rasterization \
        submodules/simple-knn \
        submodules/fused-ssim

# ── FastMap ────────────────────────────────────────────────────────────────
COPY fastmap /app/fastmap
RUN cd /app/fastmap && pip3 install --no-cache-dir --no-build-isolation .


# =============================================================================
# Stage 2 — runtime: lean image with everything needed to run
# =============================================================================
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
# Tells PyTorch to use expandable VRAM segments (avoids OOM on 8GB cards)
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Runtime system packages ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        # Python
        python3.10 python3.10-dev python3-pip \
        # Video / SfM
        ffmpeg colmap \
        # Vulkan (required by Brush — works on NVIDIA/AMD/Intel)
        libvulkan1 \
        mesa-vulkan-drivers \
        # Headless display for COLMAP feature matching
        xvfb \
        # Misc
        git wget curl ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# ── Copy compiled Python packages from builder ─────────────────────────────
COPY --from=builder /usr/local/lib/python3.10/dist-packages \
                    /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin/python* /usr/local/bin/

# ── MASt3R + DUSt3R (cloned with submodules) ──────────────────────────────
RUN git clone --recursive https://github.com/naver/mast3r /app/mast3r

# ── All Python runtime dependencies ───────────────────────────────────────
RUN pip3 install --no-cache-dir \
        # MASt3R deps
        pycolmap \
        kapture \
        kapture-localization \
        roma \
        einops \
        trimesh \
        gradio \
        "huggingface-hub[torch]>=0.22" \
        # Shared pipeline deps
        opencv-python-headless \
        plyfile lpips scipy tqdm \
        joblib numpy loguru dacite \
        pyyaml prettytable psutil \
        matplotlib

# ── Brush 0.3.0 binary (Vulkan/wgpu — GPU-vendor agnostic) ─────────────────
COPY brush-app-x86_64-unknown-linux-gnu/brush_app /usr/local/bin/brush
RUN chmod +x /usr/local/bin/brush

# ── Pipeline source files ───────────────────────────────────────────────────
COPY gaussian-splatting/submodules     /app/gaussian-splatting/submodules
COPY gaussian-splatting/scene          /app/gaussian-splatting/scene
COPY gaussian-splatting/utils          /app/gaussian-splatting/utils
COPY gaussian-splatting/gaussian_renderer /app/gaussian-splatting/gaussian_renderer
COPY gaussian-splatting/arguments      /app/gaussian-splatting/arguments
COPY gaussian-splatting/lpipsPyTorch   /app/gaussian-splatting/lpipsPyTorch

COPY gaussian-splatting/train.py \
     gaussian-splatting/render.py \
     gaussian-splatting/convert.py \
     gaussian-splatting/full_eval.py \
     gaussian-splatting/metrics.py \
     gaussian-splatting/video_to_3dgs.py \
     gaussian-splatting/mast3r_sfm.py \
     gaussian-splatting/run_pipeline.sh \
     /app/gaussian-splatting/

COPY fastmap /app/fastmap

# ── Fix hardcoded host paths ───────────────────────────────────────────────
RUN sed -i \
    's|FASTMAP_DIR = Path("/home/ibrahim/fastmap")|FASTMAP_DIR = Path("/app/fastmap")|' \
    /app/gaussian-splatting/video_to_3dgs.py

# MAST3R_DIR is already correct: Path(__file__).parent.parent / 'mast3r'
# = /app/gaussian-splatting/../../mast3r ... no wait, let's be explicit:
RUN sed -i \
    "s|MAST3R_DIR = Path(__file__).parent.parent / 'mast3r'|MAST3R_DIR = Path('/app/mast3r')|" \
    /app/gaussian-splatting/mast3r_sfm.py

# ── Data directory ─────────────────────────────────────────────────────────
RUN mkdir -p /data
VOLUME ["/data"]

# ── Entrypoint ─────────────────────────────────────────────────────────────
WORKDIR /app/gaussian-splatting

ENTRYPOINT ["python3", "video_to_3dgs.py"]
CMD ["--help"]
