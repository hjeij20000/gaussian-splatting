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

import cv2
import numpy as np


def run_command(cmd, description, cwd=None):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"[STEP] {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    # Clean env to prevent cv2's Qt plugin path from breaking COLMAP's Qt
    env = os.environ.copy()
    env.pop("QT_PLUGIN_PATH", None)
    env["QT_QPA_PLATFORM"] = "offscreen"

    start_time = time.time()
    result = subprocess.run(cmd, cwd=cwd, env=env)
    elapsed_time = time.time() - start_time

    if result.returncode != 0:
        print(f"[ERROR] {description} failed with code {result.returncode}")
        sys.exit(1)

    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"[OK] {description} completed successfully")
    print(f"[TIME] Elapsed: {minutes}m {seconds:.2f}s ({elapsed_time:.2f}s total)")

    return elapsed_time


def _blur_score(path: Path) -> float:
    """Compute sharpness score (Laplacian variance). Higher = sharper."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(img, cv2.CV_64F).var()


def select_sharpest_frames(frames_dir: Path, oversample: int = 3):
    """Smart frame selection: group oversampled frames and keep the sharpest per window.

    Instead of extracting at target FPS and deleting blurry ones (which creates
    coverage gaps), we extract at oversample * target FPS, then pick the sharpest
    frame from each window of `oversample` consecutive frames. This guarantees
    uniform temporal coverage with no gaps.
    """
    start_time = time.time()

    frames = sorted(frames_dir.glob("*.jpg"))
    total = len(frames)

    # Score all frames
    scores = [(f, _blur_score(f)) for f in frames]

    # Group into windows and pick the sharpest from each
    kept = []
    for i in range(0, total, oversample):
        window = scores[i:i + oversample]
        best = max(window, key=lambda x: x[1])
        kept.append(best[0])
        # Delete the rest
        for f, _ in window:
            if f != best[0]:
                f.unlink()

    # Rename kept frames to be sequential (COLMAP expects this)
    for i, f in enumerate(kept, 1):
        new_name = frames_dir / f"frame_{i:04d}.jpg"
        if f != new_name:
            f.rename(new_name)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"\n{'='*60}")
    print(f"[STEP] Smart frame selection (oversample={oversample}x)")
    print(f"{'='*60}")
    print(f"[INFO] Scored {total} frames, kept {len(kept)} sharpest (1 per {oversample})")
    print(f"[TIME] Elapsed: {minutes}m {seconds:.2f}s ({elapsed:.2f}s total)")

    if len(kept) < 3:
        print("[ERROR] Too few frames for reconstruction")
        sys.exit(1)

    return elapsed


def extract_frames(video_path: Path, output_dir: Path, fps: int = 2,
                   oversample: int = 3):
    """Extract frames from video, then pick the sharpest from each time window.

    Extracts at fps * oversample rate, then keeps the sharpest frame per window
    of `oversample` frames. This ensures uniform coverage with sharp frames.
    """
    frames_dir = output_dir / "input"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    extract_fps = fps * oversample

    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-c:v", "hevc_cuvid",
        "-i", str(video_path),
        "-vf", f"fps={extract_fps}",
        "-q:v", "1",
        str(frames_dir / "frame_%04d.jpg")
    ]

    elapsed = run_command(cmd, f"Extracting frames at {extract_fps} FPS (oversample {oversample}x)")

    # Count extracted frames
    frame_count = len(list(frames_dir.glob("*.jpg")))
    print(f"[INFO] Extracted {frame_count} candidate frames")

    if frame_count < 3:
        print("[ERROR] Need at least 3 frames for reconstruction")
        sys.exit(1)

    # Smart selection: pick sharpest from each window
    select_elapsed = select_sharpest_frames(frames_dir, oversample)
    elapsed += select_elapsed

    final_count = len(list(frames_dir.glob("*.jpg")))
    print(f"[INFO] Final frame count: {final_count}")

    return frames_dir, elapsed


FASTMAP_DIR = Path("/home/ibrahim/fastmap")


def _needs_xvfb():
    """Check if xvfb-run is needed (no DISPLAY set and xvfb-run available)."""
    if os.environ.get("DISPLAY"):
        return False
    return shutil.which("xvfb-run") is not None


def _wrap_colmap_cmd(cmd):
    """Wrap a COLMAP command with xvfb-run if needed for headless GPU mode."""
    if _needs_xvfb():
        return ["xvfb-run", "-a"] + cmd
    return cmd


def run_colmap(project_dir: Path, use_gpu: bool = True, use_colmap_mapper: bool = False,
               match_overlap: int = 5, sfm_backend: str = 'fastmap', window_size: int = 10):
    """Run SfM to produce a COLMAP-format sparse reconstruction.

    sfm_backend choices:
      'fastmap'  – COLMAP SIFT features + FastMap mapper (default)
      'colmap'   – COLMAP SIFT features + COLMAP incremental mapper
      'mast3r'   – MASt3R deep features + pycolmap incremental mapper (best quality)
    """

    input_dir = project_dir / "input"
    distorted_dir = project_dir / "distorted"
    sparse_dir = distorted_dir / "sparse"
    database_path = distorted_dir / "database.db"

    distorted_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    gpu_flag = "1" if use_gpu else "0"

    timings = {}

    if sfm_backend == 'mast3r':
        # ── MASt3R: deep feature matching + pycolmap mapper ─────────────────
        # Skips COLMAP SIFT entirely. Outputs COLMAP binary format directly.
        mast3r_script = Path(__file__).parent / 'mast3r_sfm.py'
        sparse_0 = sparse_dir / '0'
        sparse_0.mkdir(parents=True, exist_ok=True)
        cache_dir = distorted_dir / 'mast3r_cache'
        cmd = [
            sys.executable, str(mast3r_script),
            '--images', str(input_dir),
            '--output', str(sparse_0),
            '--cache-dir', str(cache_dir),
            '--mapper', 'glomap',
            '--glomap-bin', '/home/ibrahim/local/bin/glomap',
            '--window-size', str(window_size),
        ]
        timings['feature_extraction'] = 0
        timings['feature_matching'] = 0
        timings['sfm_reconstruction'] = run_command(cmd, "MASt3R-SfM (deep matching + GLOMAP)")
        print(f"[INFO] MASt3R reconstruction saved to {sparse_0}")

    else:
        # ── COLMAP SIFT feature extraction ──────────────────────────────────
        # FastMap only supports SIMPLE_RADIAL; COLMAP mapper handles OPENCV fine.
        camera_model = "OPENCV" if use_colmap_mapper else "SIMPLE_RADIAL"

        cmd = _wrap_colmap_cmd([
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(input_dir),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", camera_model,
            "--SiftExtraction.use_gpu", gpu_flag,
        ])
        timings['feature_extraction'] = run_command(cmd, "COLMAP feature extraction")

        # Feature matching
        cmd = _wrap_colmap_cmd([
            "colmap", "sequential_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.use_gpu", gpu_flag,
            "--SequentialMatching.overlap", str(match_overlap),
        ])
        timings['feature_matching'] = run_command(cmd, "COLMAP sequential feature matching")

        # Sparse reconstruction (Structure from Motion)
        if sfm_backend == 'colmap' or use_colmap_mapper:
            cmd = _wrap_colmap_cmd([
                "colmap", "mapper",
                "--database_path", str(database_path),
                "--image_path", str(input_dir),
                "--output_path", str(sparse_dir),
            ])
            timings['sfm_reconstruction'] = run_command(cmd, "COLMAP sparse reconstruction (mapper)")
        else:
            # FastMap: refuses to write to existing dir, use temp then move
            fastmap_out = distorted_dir / "fastmap_out"
            if fastmap_out.exists():
                shutil.rmtree(fastmap_out)
            cmd = [
                sys.executable, str(FASTMAP_DIR / "run.py"),
                "--database", str(database_path),
                "--image_dir", str(input_dir),
                "--output_dir", str(fastmap_out),
                "--headless",
            ]
            timings['sfm_reconstruction'] = run_command(cmd, "FastMap sparse reconstruction")
            src_sparse = fastmap_out / "sparse" / "0"
            dst_sparse = sparse_dir / "0"
            if dst_sparse.exists():
                shutil.rmtree(dst_sparse)
            shutil.copytree(str(src_sparse), str(dst_sparse))
            shutil.rmtree(fastmap_out)
            print(f"[INFO] Moved FastMap output to {dst_sparse}")

    # Check reconstruction succeeded
    model_dirs = list(sparse_dir.glob("*"))
    if not model_dirs:
        print("[ERROR] SfM failed to reconstruct any model")
        sys.exit(1)

    model_path = sparse_dir / "0"
    if not model_path.exists():
        model_path = model_dirs[0]

    print(f"[INFO] Using {sfm_backend} model: {model_path}")
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


def train_gaussian_splatting(project_dir: Path, output_dir: Path, iterations: int = 30000,
                             max_resolution: int = 1920):
    """Train the 3D Gaussian Splatting model using Brush."""

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "brush", str(project_dir),
        "--total-steps", str(iterations),
        "--max-resolution", str(max_resolution),
        "--export-path", str(output_dir),
        "--export-name", "export_{iter}.ply",
        "--export-every", str(iterations),
    ]
    elapsed = run_command(cmd, f"Training 3DGS model with Brush ({iterations} steps)")
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
    parser.add_argument("--use-colmap", action="store_true",
                        help="Use COLMAP mapper instead of FastMap for SfM (ignored if --sfm-backend is set)")
    parser.add_argument("--sfm-backend", type=str, default='fastmap',
                        choices=['fastmap', 'colmap', 'mast3r'],
                        help="SfM backend: fastmap (default), colmap, or mast3r (best quality)")
    parser.add_argument("--max-resolution", type=int, default=1920,
                        help="Max image resolution for Brush training in pixels (default: 1920)")
    parser.add_argument("--oversample", type=int, default=3,
                        help="Extract N x FPS frames, keep sharpest per window (default: 3)")
    parser.add_argument("--match-overlap", type=int, default=5,
                        help="COLMAP sequential matching overlap (default: 5, lower=faster)")
    parser.add_argument("--window-size", type=int, default=10,
                        help="MASt3R sliding window size for pair creation (default: 10)")

    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    output_dir = Path(args.output).resolve()

    sfm_backend = args.sfm_backend
    if args.use_colmap and sfm_backend == 'fastmap':
        sfm_backend = 'colmap'

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
SfM backend:{sfm_backend}
{'='*60}
    """)

    # Track timing for each stage
    timings = {}
    pipeline_start = time.time()

    # Step 1: Extract frames
    if not args.skip_frames:
        _, timings['frame_extraction'] = extract_frames(video_path, work_dir, args.fps,
                                                        oversample=args.oversample)
    else:
        print("[SKIP] Frame extraction (using existing frames)")
        timings['frame_extraction'] = 0

    # Step 2: Run SfM
    if not args.skip_colmap:
        _, colmap_timings = run_colmap(work_dir, use_gpu=not args.no_gpu,
                                       use_colmap_mapper=(sfm_backend == 'colmap'),
                                       match_overlap=args.match_overlap,
                                       sfm_backend=sfm_backend,
                                       window_size=args.window_size)
        timings['undistortion'] = undistort_images(work_dir)
        timings.update(colmap_timings)
    else:
        print("[SKIP] COLMAP (using existing camera poses)")
        timings['feature_extraction'] = 0
        timings['feature_matching'] = 0
        timings['sfm_reconstruction'] = 0
        timings['undistortion'] = 0

    # Step 3: Train 3DGS
    timings['training'] = train_gaussian_splatting(work_dir, model_dir, args.iterations,
                                                   max_resolution=args.max_resolution)

    total_time = time.time() - pipeline_start

    # Print timing summary
    print(f"""
{'='*60}
Pipeline Complete!
{'='*60}
Model saved to: {model_dir}

Output files:
  - {model_dir}/export_{args.iterations}.ply

{'='*60}
TIMING SUMMARY
{'='*60}
Frame Extraction:        {timings['frame_extraction']:8.2f}s ({timings['frame_extraction']/60:6.2f}m)
Feature Extraction:      {timings['feature_extraction']:8.2f}s ({timings['feature_extraction']/60:6.2f}m)
Feature Matching:        {timings['feature_matching']:8.2f}s ({timings['feature_matching']/60:6.2f}m)
SfM Reconstruction:      {timings['sfm_reconstruction']:8.2f}s ({timings['sfm_reconstruction']/60:6.2f}m)
Image Undistortion:      {timings['undistortion']:8.2f}s ({timings['undistortion']/60:6.2f}m)
3DGS Training:           {timings['training']:8.2f}s ({timings['training']/60:6.2f}m)
{'='*60}
TOTAL TIME:              {total_time:8.2f}s ({total_time/60:6.2f}m)
{'='*60}

To view the model:
  brush {model_dir}/export_{args.iterations}.ply --with-viewer
{'='*60}
    """)


if __name__ == "__main__":
    main()
