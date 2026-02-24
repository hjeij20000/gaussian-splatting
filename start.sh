#!/bin/bash
# =============================================================================
# RunPod container startup script
# - Sets up SSH with the injected PUBLIC_KEY
# - Auto-runs the pipeline if VIDEO env var is set
# - Otherwise keeps the container alive for interactive SSH use
# =============================================================================
set -e

# ── SSH setup ────────────────────────────────────────────────────────────────
mkdir -p /root/.ssh
chmod 700 /root/.ssh

if [ -n "${PUBLIC_KEY}" ]; then
    echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    echo "[start] SSH public key installed."
fi

service ssh start
echo "[start] SSH server started on port 22."

# ── Optional auto-run ────────────────────────────────────────────────────────
# Set VIDEO=/workspace/my_video.mp4 to run the pipeline automatically on start.
# All other pipeline args can be set via env vars too.
if [ -n "${VIDEO}" ]; then
    OUTPUT="${OUTPUT:-/workspace/output}"
    ITERATIONS="${ITERATIONS:-30000}"
    SFM_BACKEND="${SFM_BACKEND:-mast3r}"
    FPS="${FPS:-2}"

    echo "[start] Auto-running pipeline:"
    echo "  VIDEO=${VIDEO}"
    echo "  OUTPUT=${OUTPUT}"
    echo "  SFM_BACKEND=${SFM_BACKEND}"
    echo "  ITERATIONS=${ITERATIONS}"

    cd /app/gaussian-splatting
    python3 video_to_3dgs.py \
        --video "${VIDEO}" \
        --output "${OUTPUT}" \
        --sfm-backend "${SFM_BACKEND}" \
        --iterations "${ITERATIONS}" \
        --fps "${FPS}"
else
    echo ""
    echo "[start] Container ready. SSH in and run the pipeline manually:"
    echo "  cd /app/gaussian-splatting"
    echo "  python3 video_to_3dgs.py --video /workspace/video.mp4 \\"
    echo "      --output /workspace/output --sfm-backend mast3r"
    echo ""
fi

# Keep container alive so RunPod can SSH in / inspect logs
sleep infinity
