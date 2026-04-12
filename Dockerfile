# =============================================================================
# Video-to-3DGS Pipeline — RunPod Serverless Docker Image
# Includes: MASt3R-SfM, GLOMAP, COLMAP, Brush trainer, ffmpeg
#
# GPU support:
#   CUDA (MASt3R/PyTorch): Pascal 6.1 → Blackwell 10.0  (all NVIDIA generations)
#   Vulkan (Brush):         NVIDIA / AMD / Intel          (any Vulkan 1.1+ GPU)
#
# Build (from /home/ibrahim — the parent of gaussian-splatting/):
#   docker build -f gaussian-splatting/Dockerfile -t hjeij2000/video-to-3dgs:serverless .
#
# RunPod serverless env vars required:
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET, AWS_S3_REGION
# =============================================================================


# =============================================================================
# Stage 1 — builder: compile all CUDA extensions
# =============================================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# All NVIDIA GPU generations from Pascal (GTX 10xx) through Blackwell (B200)
# +PTX at the end enables JIT compilation for future architectures
ENV TORCH_CUDA_ARCH_LIST="6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX"
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
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124

# ── 3DGS CUDA extensions ───────────────────────────────────────────────────
# Copy submodules first (heaviest cached layer)
COPY gaussian-splatting/submodules /app/gaussian-splatting/submodules

WORKDIR /app/gaussian-splatting
RUN pip3 install --no-cache-dir --no-build-isolation \
        submodules/diff-gaussian-rasterization \
        submodules/simple-knn \
        submodules/fused-ssim

# ── 2DGS CUDA extension (diff-surfel-rasterization) ───────────────────────
# simple-knn is identical to the 3DGS one already installed above
COPY 2d-gaussian-splatting/submodules/diff-surfel-rasterization \
     /app/2d-gaussian-splatting/submodules/diff-surfel-rasterization
RUN pip3 install --no-cache-dir --no-build-isolation \
        /app/2d-gaussian-splatting/submodules/diff-surfel-rasterization

# ── FastMap ────────────────────────────────────────────────────────────────
COPY fastmap /app/fastmap
RUN cd /app/fastmap && pip3 install --no-cache-dir --no-build-isolation .

# ── gsplat (3DGUT CUDA extension) ─────────────────────────────────────────
# Copy only the package source + examples (skip .git, docs, assets)
COPY Luminance-GS/gsplat/gsplat      /app/gsplat/gsplat
COPY Luminance-GS/gsplat/examples    /app/gsplat/examples
COPY Luminance-GS/gsplat/setup.py    /app/gsplat/setup.py
COPY Luminance-GS/gsplat/README.md   /app/gsplat/README.md
# labeled_partition (used by 3DGUT kernels) requires sm_70+; drop Pascal (sm_61) for gsplat only
RUN cd /app/gsplat && \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX" \
    pip3 install --no-cache-dir --no-build-isolation -e .

# ── 3DGUT runtime CUDA extensions (need nvcc — must live in builder stage) ──
RUN pip3 install --no-cache-dir --no-build-isolation \
        "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5" \
        "git+https://github.com/harry7557558/fused-bilagrid@90f9788e57d3545e3a033c1038bb9986549632fe"


# =============================================================================
# Stage 2 — runtime: lean image with everything needed to run
# =============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
# Tells PyTorch to use expandable VRAM segments (avoids OOM on 8GB cards)
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Enable all NVIDIA driver capabilities (compute + graphics/Vulkan for Brush)
ENV NVIDIA_DRIVER_CAPABILITIES=all

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
        # SSH (required by RunPod for interactive pod access)
        openssh-server \
        # Boost + OpenBLAS — required by the glomap binary (mast3r backend)
        libboost-program-options1.74.0 \
        libboost-filesystem1.74.0 \
        libboost-graph1.74.0 \
        libopenblas0-pthread \
        # Misc
        git wget curl ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    # Configure SSH: allow root login, no password auth
    && mkdir -p /run/sshd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config \
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
        matplotlib \
        # Serverless handler deps
        runpod \
        gdown \
        boto3 \
        # 3DGUT / gsplat trainer deps
        "imageio[ffmpeg]" \
        viser \
        "tyro>=0.8.8" \
        scikit-learn \
        "torchmetrics[image]" \
        tensorboard \
        tensorly \
        splines

RUN pip3 install --no-cache-dir \
        "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787"

# ── Copy compiled gsplat package from builder ──────────────────────────────
COPY --from=builder /app/gsplat /app/gsplat

# ── Brush 0.3.0 binary (Vulkan/wgpu — GPU-vendor agnostic) ─────────────────
COPY brush-app-x86_64-unknown-linux-gnu/brush_app /usr/local/bin/brush
RUN chmod +x /usr/local/bin/brush

# ── NVIDIA Vulkan ICD (needed for Brush/wgpu to use the hardware GPU) ───────
# NVIDIA_DRIVER_CAPABILITIES=all tells the container runtime to mount the
# NVIDIA graphics libs; this ICD JSON tells Vulkan where to find them.
RUN mkdir -p /usr/share/vulkan/icd.d && \
    printf '{\n    "file_format_version": "1.0.0",\n    "ICD": {\n        "library_path": "libGLX_nvidia.so.0",\n        "api_version": "1.3.0"\n    }\n}\n' \
    > /usr/share/vulkan/icd.d/nvidia_icd.json

# ── GLOMAP binary (compiled from source on host) ────────────────────────────
COPY local/bin/glomap /usr/local/bin/glomap
RUN chmod +x /usr/local/bin/glomap

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
     gaussian-splatting/hloc_sfm.py \
     gaussian-splatting/pycusfm_sfm.py \
     gaussian-splatting/handler.py \
     gaussian-splatting/run_pipeline.sh \
     gaussian-splatting/start.sh \
     /app/gaussian-splatting/

RUN chmod +x /app/gaussian-splatting/start.sh

COPY fastmap /app/fastmap

# ── 2DGS source (train.py + scene/gaussian_renderer/utils/arguments) ──────
COPY 2d-gaussian-splatting/train.py          /app/2d-gaussian-splatting/
COPY 2d-gaussian-splatting/scene             /app/2d-gaussian-splatting/scene
COPY 2d-gaussian-splatting/gaussian_renderer /app/2d-gaussian-splatting/gaussian_renderer
COPY 2d-gaussian-splatting/utils             /app/2d-gaussian-splatting/utils
COPY 2d-gaussian-splatting/arguments        /app/2d-gaussian-splatting/arguments
COPY 2d-gaussian-splatting/lpipsPyTorch     /app/2d-gaussian-splatting/lpipsPyTorch

# ── hloc (SuperPoint + LightGlue) ─────────────────────────────────────────
COPY hloc /app/hloc
RUN pip3 install --no-cache-dir \
        h5py \
        kornia>=0.6.11 \
        "lightglue @ git+https://github.com/cvg/LightGlue"

# ── Fix hardcoded host paths ───────────────────────────────────────────────
RUN sed -i \
    's|FASTMAP_DIR = Path("/home/ibrahim/fastmap")|FASTMAP_DIR = Path("/app/fastmap")|' \
    /app/gaussian-splatting/video_to_3dgs.py

RUN sed -i \
    "s|'--glomap-bin', '/home/ibrahim/local/bin/glomap'|'--glomap-bin', '/usr/local/bin/glomap'|" \
    /app/gaussian-splatting/video_to_3dgs.py

RUN sed -i \
    "s|MAST3R_DIR = Path(__file__).parent.parent / 'mast3r'|MAST3R_DIR = Path('/app/mast3r')|" \
    /app/gaussian-splatting/mast3r_sfm.py

RUN sed -i \
    "s|default='/home/ibrahim/local/bin/glomap'|default='/usr/local/bin/glomap'|g" \
    /app/gaussian-splatting/mast3r_sfm.py

RUN sed -i \
    "s|HLOC_DIR = Path(\"/home/ibrahim/hloc\")|HLOC_DIR = Path(\"/app/hloc\")|" \
    /app/gaussian-splatting/hloc_sfm.py

RUN sed -i \
    's|GSPLAT_DIR = Path("/home/ibrahim/Luminance-GS/gsplat")|GSPLAT_DIR = Path("/app/gsplat")|' \
    /app/gaussian-splatting/video_to_3dgs.py

RUN sed -i \
    's|TWODGS_DIR = Path("/home/ibrahim/2d-gaussian-splatting")|TWODGS_DIR = Path("/app/2d-gaussian-splatting")|' \
    /app/gaussian-splatting/video_to_3dgs.py

# ── Pre-download MASt3R weights (baked into image to avoid cold-start delay) ─
RUN PYTHONPATH=/app/mast3r:/app/mast3r/dust3r:/app/mast3r/dust3r/dust3r_visloc \
    python3 -c "\
from mast3r.model import AsymmetricMASt3R; \
AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'); \
print('[build] MASt3R weights cached.')"

# ── Workspace (RunPod mounts persistent storage here) ──────────────────────
RUN mkdir -p /workspace
VOLUME ["/workspace"]

# ── Expose SSH port ────────────────────────────────────────────────────────
EXPOSE 22

# ── Entrypoint ─────────────────────────────────────────────────────────────
WORKDIR /app/gaussian-splatting

ENTRYPOINT ["python3", "/app/gaussian-splatting/handler.py"]
