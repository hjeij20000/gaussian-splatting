#!/bin/bash
# Install MASt3R-SfM dependencies into the gaussian-splatting venv.
# Run once before using --sfm-backend mast3r.

set -e

VENV="/home/ibrahim/gaussian-splatting/venv"
MAST3R_DIR="/home/ibrahim/mast3r"

echo "============================================================"
echo "Installing MASt3R-SfM into: $VENV"
echo "============================================================"

source "$VENV/bin/activate"

# Core deps not already in the venv
pip install \
    pycolmap \
    kapture \
    kapture-localization \
    roma \
    einops \
    trimesh \
    gradio \
    opencv-python \
    tqdm \
    matplotlib \
    "huggingface-hub[torch]>=0.22"

# MASt3R and DUSt3R have no setup.py — they are imported via sys.path
# in mast3r_sfm.py (MAST3R_DIR = /home/ibrahim/mast3r)

echo "============================================================"
echo "Done! Test with:"
echo "  python mast3r_sfm.py --images /path/to/images --output /tmp/test_recon"
echo "============================================================"
