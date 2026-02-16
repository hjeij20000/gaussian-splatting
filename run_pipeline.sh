#!/bin/bash
# Video to 3DGS Pipeline Runner
# Activates the virtual environment and runs the pipeline

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Run the pipeline with all arguments
python video_to_3dgs.py "$@"
