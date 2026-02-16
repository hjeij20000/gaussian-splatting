#!/usr/bin/env python3
"""
Video to 3D Gaussian Splatting Pipeline
Converts a video file to a trained 3DGS model.

Usage:
    python video_to_3dgs.py --video /path/to/video.mp4 --output /path/to/output

Pipeline steps:
1. Extract frames from video (ffmpeg)
2. Run COLMAP for camera pose estimation
3. Train 3D Gaussian Splatting model
"""

import os
import sys
import argparse
import subprocess
import shutil
import time
from pathlib import Path


def run_command(cmd, description, cwd=None):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"[STEP] {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()
    result = subprocess.run(cmd, cwd=cwd)
    elapsed_time = time.time() - start_time

    if result.returncode != 0:
        print(f"[ERROR] {description} failed with code {result.returncode}")
        sys.exit(1)

    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"[OK] {description} completed successfully")
    print(f"[TIME] Elapsed: {minutes}m {seconds:.2f}s ({elapsed_time:.2f}s total)")

    return elapsed_time


def extract_frames(video_path: Path, output_dir: Path, fps: int = 2):
    """Extract frames from video using ffmpeg."""
    frames_dir = output_dir / "input"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "1",
        str(frames_dir / "frame_%04d.jpg")
    ]

    elapsed = run_command(cmd, f"Extracting frames at {fps} FPS")

    # Count extracted frames
    frame_count = len(list(frames_dir.glob("*.jpg")))
    print(f"[INFO] Extracted {frame_count} frames")

    if frame_count < 3:
        print("[ERROR] Need at least 3 frames for reconstruction")
        sys.exit(1)

    return frames_dir, elapsed


def run_colmap(project_dir: Path, use_gpu: bool = True):
    """Run COLMAP to estimate camera poses."""

    input_dir = project_dir / "input"
    distorted_dir = project_dir / "distorted"
    sparse_dir = distorted_dir / "sparse"
    database_path = distorted_dir / "database.db"

    distorted_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    gpu_flag = "1" if use_gpu else "0"

    timings = {}

    # Feature extraction
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(input_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "OPENCV",
        "--SiftExtraction.use_gpu", gpu_flag,
    ]
    timings['feature_extraction'] = run_command(cmd, "COLMAP feature extraction")

    # Feature matching
    cmd = [
        "colmap", "exhaustive_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", gpu_flag,
    ]
    timings['feature_matching'] = run_command(cmd, "COLMAP feature matching")

    # Sparse reconstruction (Structure from Motion)
    cmd = [
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(input_dir),
        "--output_path", str(sparse_dir),
    ]
    timings['sparse_reconstruction'] = run_command(cmd, "COLMAP sparse reconstruction")

    # Check if reconstruction succeeded
    model_dirs = list(sparse_dir.glob("*"))
    if not model_dirs:
        print("[ERROR] COLMAP failed to reconstruct any model")
        sys.exit(1)

    # Use the largest model (most images registered)
    model_path = sparse_dir / "0"
    if not model_path.exists():
        model_path = model_dirs[0]

    print(f"[INFO] Using COLMAP model: {model_path}")

    return model_path, timings


def undistort_images(project_dir: Path):
    """Undistort images using COLMAP."""

    distorted_dir = project_dir / "distorted"
    sparse_model = distorted_dir / "sparse" / "0"

    if not sparse_model.exists():
        # Try to find any model
        sparse_models = list((distorted_dir / "sparse").glob("*"))
        if sparse_models:
            sparse_model = sparse_models[0]
        else:
            print("[ERROR] No COLMAP model found")
            sys.exit(1)

    cmd = [
        "colmap", "image_undistorter",
        "--image_path", str(project_dir / "input"),
        "--input_path", str(sparse_model),
        "--output_path", str(project_dir),
        "--output_type", "COLMAP",
    ]
    elapsed = run_command(cmd, "COLMAP image undistortion")

    # Fix: Move sparse model to sparse/0/ subdirectory for 3DGS training
    sparse_dir = project_dir / "sparse"
    sparse_0_dir = sparse_dir / "0"
    if sparse_dir.exists() and not sparse_0_dir.exists():
        sparse_0_dir.mkdir(parents=True, exist_ok=True)
        for file in sparse_dir.glob("*.bin"):
            shutil.move(str(file), str(sparse_0_dir / file.name))
        print(f"[INFO] Moved sparse model to {sparse_0_dir}")

    return elapsed


def train_gaussian_splatting(project_dir: Path, output_dir: Path, iterations: int = 30000):
    """Train the 3D Gaussian Splatting model."""

    script_dir = Path(__file__).parent
    train_script = script_dir / "train.py"

    cmd = [
        sys.executable, str(train_script),
        "-s", str(project_dir),
        "-m", str(output_dir),
        "--iterations", str(iterations),
    ]
    elapsed = run_command(cmd, f"Training 3DGS model ({iterations} iterations)", cwd=script_dir)
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Convert video to 3D Gaussian Splatting model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python video_to_3dgs.py --video my_video.mp4 --output ./my_3dgs

    # Extract more frames (higher quality, slower)
    python video_to_3dgs.py --video my_video.mp4 --output ./my_3dgs --fps 5

    # Quick training (lower quality, faster)
    python video_to_3dgs.py --video my_video.mp4 --output ./my_3dgs --iterations 7000
        """
    )

    parser.add_argument("--video", "-v", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Path to output directory")
    parser.add_argument("--fps", type=int, default=2,
                        help="Frames per second to extract (default: 2)")
    parser.add_argument("--iterations", "-i", type=int, default=30000,
                        help="Training iterations (default: 30000, use 7000 for quick test)")
    parser.add_argument("--skip-frames", action="store_true",
                        help="Skip frame extraction (use existing frames)")
    parser.add_argument("--skip-colmap", action="store_true",
                        help="Skip COLMAP (use existing camera poses)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU for COLMAP")

    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    output_dir = Path(args.output).resolve()

    if not video_path.exists() and not args.skip_frames:
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Working directory for COLMAP data
    work_dir = output_dir / "colmap_workspace"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Model output directory
    model_dir = output_dir / "model"

    print(f"""
{'='*60}
Video to 3D Gaussian Splatting Pipeline
{'='*60}
Video:      {video_path}
Output:     {output_dir}
FPS:        {args.fps}
Iterations: {args.iterations}
{'='*60}
    """)

    # Track timing for each stage
    timings = {}
    pipeline_start = time.time()

    # Step 1: Extract frames
    if not args.skip_frames:
        _, timings['frame_extraction'] = extract_frames(video_path, work_dir, args.fps)
    else:
        print("[SKIP] Frame extraction (using existing frames)")
        timings['frame_extraction'] = 0

    # Step 2: Run COLMAP
    if not args.skip_colmap:
        _, colmap_timings = run_colmap(work_dir, use_gpu=not args.no_gpu)
        timings['undistortion'] = undistort_images(work_dir)
        timings.update(colmap_timings)
    else:
        print("[SKIP] COLMAP (using existing camera poses)")
        timings['feature_extraction'] = 0
        timings['feature_matching'] = 0
        timings['sparse_reconstruction'] = 0
        timings['undistortion'] = 0

    # Step 3: Train 3DGS
    timings['training'] = train_gaussian_splatting(work_dir, model_dir, args.iterations)

    total_time = time.time() - pipeline_start

    # Print timing summary
    print(f"""
{'='*60}
Pipeline Complete!
{'='*60}
Model saved to: {model_dir}

Output files:
  - {model_dir}/point_cloud/iteration_{args.iterations}/point_cloud.ply
  - {model_dir}/cameras.json

{'='*60}
TIMING SUMMARY
{'='*60}
Frame Extraction:        {timings['frame_extraction']:8.2f}s ({timings['frame_extraction']/60:6.2f}m)
COLMAP Feature Extract:  {timings['feature_extraction']:8.2f}s ({timings['feature_extraction']/60:6.2f}m)
COLMAP Feature Matching: {timings['feature_matching']:8.2f}s ({timings['feature_matching']/60:6.2f}m)
COLMAP Reconstruction:   {timings['sparse_reconstruction']:8.2f}s ({timings['sparse_reconstruction']/60:6.2f}m)
COLMAP Undistortion:     {timings['undistortion']:8.2f}s ({timings['undistortion']/60:6.2f}m)
3DGS Training:           {timings['training']:8.2f}s ({timings['training']/60:6.2f}m)
{'='*60}
TOTAL TIME:              {total_time:8.2f}s ({total_time/60:6.2f}m)
{'='*60}

To view the model, use the SIBR viewer:
  ./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m {model_dir}

To render images:
  python render.py -m {model_dir}
{'='*60}
    """)


if __name__ == "__main__":
    main()
