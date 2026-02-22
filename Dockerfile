# =============================================================================
# Video-to-3DGS Pipeline Docker Image
# Includes: 3D Gaussian Splatting, FastMap, COLMAP, ffmpeg
# =============================================================================
# Build (from the parent directory of gaussian-splatting/):
#   docker build -f gaussian-splatting/Dockerfile -t video-to-3dgs .
#
# Run:
#   docker run --gpus all -v /path/to/data:/data video-to-3dgs \
#       --video /data/video.mp4 --output /data/output
# =============================================================================

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
# Ampere (30xx) + Ada Lovelace (40xx) + Hopper + Blackwell (50xx / RTX PRO 6000)
ENV TORCH_CUDA_ARCH_LIST="8.6 8.9 9.0 12.0+PTX"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"

# ---- System dependencies ----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git cmake build-essential ninja-build \
        ffmpeg colmap \
        libglew-dev libglfw3-dev \
        xvfb \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# ---- PyTorch (cu128) --------------------------------------------------------
RUN pip3 install --no-cache-dir \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128

# ---- 3D Gaussian Splatting --------------------------------------------------
COPY gaussian-splatting/submodules /app/gaussian-splatting/submodules

# Build CUDA extensions first (heaviest layer, cached unless submodules change)
WORKDIR /app/gaussian-splatting
RUN pip3 install --no-cache-dir --no-build-isolation \
        submodules/diff-gaussian-rasterization \
        submodules/simple-knn \
        submodules/fused-ssim

# Copy the rest of gaussian-splatting source
COPY gaussian-splatting/train.py \
     gaussian-splatting/render.py \
     gaussian-splatting/convert.py \
     gaussian-splatting/full_eval.py \
     gaussian-splatting/metrics.py \
     gaussian-splatting/video_to_3dgs.py \
     gaussian-splatting/run_pipeline.sh \
     /app/gaussian-splatting/

COPY gaussian-splatting/scene      /app/gaussian-splatting/scene
COPY gaussian-splatting/utils       /app/gaussian-splatting/utils
COPY gaussian-splatting/gaussian_renderer /app/gaussian-splatting/gaussian_renderer
COPY gaussian-splatting/arguments   /app/gaussian-splatting/arguments
COPY gaussian-splatting/lpipsPyTorch /app/gaussian-splatting/lpipsPyTorch

# ---- FastMap -----------------------------------------------------------------
COPY fastmap /app/fastmap
RUN cd /app/fastmap && pip3 install --no-cache-dir --no-build-isolation .

# ---- Remaining Python dependencies ------------------------------------------
RUN pip3 install --no-cache-dir \
        opencv-python-headless plyfile lpips scipy tqdm joblib numpy \
        loguru dacite pyyaml einops prettytable psutil

# ---- Patch hardcoded paths for container -------------------------------------
RUN sed -i 's|FASTMAP_DIR = Path("/home/ibrahim/fastmap")|FASTMAP_DIR = Path("/app/fastmap")|' \
        /app/gaussian-splatting/video_to_3dgs.py

# ---- Runtime setup -----------------------------------------------------------
WORKDIR /app/gaussian-splatting
RUN mkdir -p /data

CMD ["sleep", "infinity"]
