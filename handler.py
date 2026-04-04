#!/usr/bin/env python3
"""
RunPod Serverless Handler — Video to 3D Gaussian Splatting
==========================================================

Required environment variables (set in RunPod endpoint config):
    AWS_ACCESS_KEY_ID       AWS credentials for S3 output upload
    AWS_SECRET_ACCESS_KEY
    AWS_S3_BUCKET           Bucket to upload the output PLY to
    AWS_S3_REGION           (optional, default: us-east-1)

Input JSON:
    {
        "video_url":     "https://... or https://drive.google.com/...",  # required
        "fps":            2,         # frames/sec to extract            (default: 2)
        "iterations":     7000,      # Brush training steps             (default: 7000)
        "sfm_backend":    "mast3r",  # mast3r | fastmap | colmap | hloc | pycusfm (default: mast3r)
        "trainer":        "brush",   # brush (default) | 3dgut

        # Common optional args (all backends):
        "max_resolution": 1920,      # max image dimension for Brush    (default: 1920, ignored for 3dgut)
        "oversample":     3,         # oversample factor for blur filter (default: 3)

        # Backend-specific args — only used when the matching backend is selected:
        #
        #   sfm_backend = "mast3r"
        "window_size":    10,        # MASt3R sliding-window pair size  (default: 10)
        #
        #   sfm_backend = "fastmap" | "colmap" | "hloc"
        "match_overlap":  5,         # COLMAP sequential matching overlap (default: 5)
    }

Output JSON (success):
    {
        "ply_url":     "https://s3.amazonaws.com/...presigned...",
        "ply_size_mb": 42.3,
        "job_id":      "...",
        "gpu":         "NVIDIA GeForce RTX 4090, 24564 MiB",
        "timings": {
            "frame_extraction":   12.5,
            "feature_extraction":  8.3,
            "feature_matching":    5.1,
            "sfm_reconstruction": 45.2,
            "undistortion":        3.8,
            "training":          120.4,
            "total":             195.3
        }
    }

Output JSON (failure):
    {"error": "...message...", "gpu": "..."}
"""

import os
import re
import sys
import time
import shutil
import tempfile
import subprocess
from pathlib import Path

import boto3
from botocore.config import Config
import gdown
import requests
import runpod

PIPELINE_SCRIPT = Path(__file__).parent / "video_to_3dgs.py"

# Which extra args each backend accepts, their CLI flags, and defaults
_BACKEND_ARGS = {
    "mast3r":  {"window_size":   ("--window-size",   10)},
    "fastmap": {"match_overlap": ("--match-overlap",  5)},
    "colmap":  {"match_overlap": ("--match-overlap",  5)},
    "hloc":    {"match_overlap": ("--match-overlap",  5)},
    "pycusfm": {"match_overlap": ("--match-overlap",  5)},
}

VALID_BACKENDS = list(_BACKEND_ARGS)

# Maps [STEP] text fragments → (step_number/total, short label)
_STEP_MAP = [
    ("Extracting frames",           "1/6", "Extracting frames"),
    ("Smart frame selection",       "1/6", "Selecting sharpest frames"),
    ("COLMAP feature extraction",   "2/6", "Extracting features (COLMAP)"),
    ("feature extraction",          "2/6", "Extracting features"),
    ("COLMAP feature matching",     "3/6", "Matching features (COLMAP)"),
    ("feature matching",            "3/6", "Matching features"),
    ("FastMap sparse",              "3/6", "FastMap SfM reconstruction"),
    ("COLMAP mapper",               "3/6", "SfM reconstruction (COLMAP)"),
    ("MASt3R",                      "3/6", "MASt3R SfM reconstruction"),
    ("hloc",                        "3/6", "HLoc SfM reconstruction"),
    ("PyCuSfM",                     "3/6", "PyCuSfM reconstruction"),
    ("COLMAP image undistortion",   "4/6", "Undistorting images"),
    ("image undistortion",          "4/6", "Undistorting images"),
    ("Training 3DGS",               "5/6", "Training Gaussians (Brush)"),
    ("Training 3DGS with 3DGUT",    "5/6", "Training Gaussians (3DGUT)"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_gpu_info() -> str:
    """Query nvidia-smi for GPU name and total VRAM."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            return r.stdout.strip().split("\n")[0].strip()
    except Exception:
        pass
    return "Unknown GPU"


def download_video(url: str, dest: str):
    """Download video from a regular URL or a Google Drive share link."""
    if "drive.google.com" in url:
        gdown.download(url, dest, fuzzy=True, quiet=False)
    else:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        raise RuntimeError(f"Video download failed or empty: {url}")


def upload_to_s3(local_path: str, s3_key: str) -> str:
    """Upload a file to S3 and return a 24-hour presigned URL using the regional endpoint."""
    bucket = os.environ["AWS_S3_BUCKET"]
    region = os.environ.get("AWS_S3_REGION", "us-east-1")
    endpoint_url = f"https://s3.{region}.amazonaws.com"
    s3 = boto3.client("s3", region_name=region, endpoint_url=endpoint_url,
                      config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))
    s3.upload_file(local_path, bucket, s3_key)
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=86400,
    )
    return url


def parse_sfm_result(stdout: str) -> dict:
    """Parse [SFM_RESULT] line: registered=N total=M points3d=P"""
    m = re.search(r"\[SFM_RESULT\] registered=(\d+) total=(\d+) points3d=(\d+)", stdout)
    if m:
        return {"registered": int(m.group(1)), "total": int(m.group(2)), "points3d": int(m.group(3))}
    return {}


def count_ply_splats(ply_path: str) -> int:
    """Read PLY header to get vertex count (= number of Gaussian splats)."""
    try:
        with open(ply_path, "rb") as f:
            for line in f:
                line = line.decode("ascii", errors="ignore").strip()
                if line.startswith("element vertex"):
                    return int(line.split()[-1])
                if line == "end_header":
                    break
    except Exception:
        pass
    return 0


def parse_timings(stdout: str) -> dict:
    """Parse the timing summary block printed by video_to_3dgs.py."""
    patterns = {
        "frame_extraction":   r"Frame Extraction:\s+([\d.]+)s",
        "feature_extraction": r"Feature Extraction:\s+([\d.]+)s",
        "feature_matching":   r"Feature Matching:\s+([\d.]+)s",
        "sfm_reconstruction": r"SfM Reconstruction:\s+([\d.]+)s",
        "undistortion":       r"Image Undistortion:\s+([\d.]+)s",
        "training":           r"3DGS Training:\s+([\d.]+)s",
        "total":              r"TOTAL TIME:\s+([\d.]+)s",
    }
    timings = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, stdout)
        if m:
            timings[key] = round(float(m.group(1)), 2)
    return timings


def format_timings_summary(timings: dict) -> str:
    """Return a human-readable timing table for handler logs."""
    if not timings:
        return ""
    rows = [
        ("frame_extraction",   "Frame Extraction"),
        ("feature_extraction", "Feature Extraction"),
        ("feature_matching",   "Feature Matching"),
        ("sfm_reconstruction", "SfM Reconstruction"),
        ("undistortion",       "Image Undistortion"),
        ("training",           "3DGS Training"),
        ("total",              "TOTAL TIME"),
    ]
    lines = ["\n" + "=" * 50, "TIMING SUMMARY", "=" * 50]
    for key, label in rows:
        if key in timings:
            secs = timings[key]
            lines.append(f"  {label:<22} {secs:8.2f}s  ({secs / 60:6.2f}m)")
    lines.append("=" * 50)
    return "\n".join(lines)


def run_pipeline_with_progress(job, cmd, gpu_info: str, job_start: float, work_dir: str):
    """
    Run the pipeline subprocess, streaming stdout line-by-line.
    Sends a progress_update to RunPod whenever a [STEP] marker is detected.
    Returns (returncode, stdout_str, stderr_str).

    stderr is redirected to a file to avoid the classic pipe deadlock:
    if the subprocess fills the 64 KB stderr OS pipe buffer, it blocks
    and can no longer write to stdout either — causing the handler to hang
    forever waiting for stdout, leaving the RunPod worker stuck in WORKING.
    """
    stderr_path = os.path.join(work_dir, "stderr.txt")
    with open(stderr_path, "w") as stderr_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            bufsize=1,
        )

        stdout_lines = []
        for line in proc.stdout:
            stdout_lines.append(line)
            stripped = line.strip()
            if stripped.startswith("[STEP]"):
                step_text = stripped[6:].strip()
                elapsed = time.time() - job_start
                for fragment, step_num, label in _STEP_MAP:
                    if fragment.lower() in step_text.lower():
                        msg = f"[{step_num}] {label} — {elapsed:.0f}s elapsed | {gpu_info}"
                        runpod.serverless.progress_update(job, msg)
                        print(f"[handler] Progress: {msg}")
                        break

        proc.wait()

    stdout = "".join(stdout_lines)
    with open(stderr_path) as f:
        stderr = f.read()
    return proc.returncode, stdout, stderr


# ── Handler ───────────────────────────────────────────────────────────────────

def handler(job):
    job_start = time.time()
    inp = job["input"]

    video_url      = inp.get("video_url")
    fps            = int(inp.get("fps", 2))
    iterations     = int(inp.get("iterations", 7000))
    sfm_backend    = inp.get("sfm_backend", "mast3r")
    trainer        = inp.get("trainer", "brush")
    max_resolution = int(inp.get("max_resolution", 1920))
    oversample     = int(inp.get("oversample", 3))

    if trainer not in ("brush", "3dgut"):
        return {"error": f"Unknown trainer '{trainer}'. Valid choices: brush, 3dgut"}

    if not video_url:
        return {"error": "Missing required field: video_url"}

    if sfm_backend not in VALID_BACKENDS:
        return {"error": f"Unknown sfm_backend '{sfm_backend}'. "
                         f"Valid choices: {VALID_BACKENDS}"}

    # ── GPU info ─────────────────────────────────────────────────────────────
    gpu_info = get_gpu_info()
    print(f"[handler] GPU: {gpu_info}")

    work_dir = tempfile.mkdtemp(prefix="runpod_3dgs_")

    try:
        # ── 1. Download video ────────────────────────────────────────────────
        runpod.serverless.progress_update(job, f"[0/6] Downloading video... | {gpu_info}")
        video_ext = Path(video_url.split("?")[0]).suffix or ".mp4"
        video_path = os.path.join(work_dir, f"input{video_ext}")
        download_video(video_url, video_path)
        video_mb = os.path.getsize(video_path) / 1e6
        elapsed = time.time() - job_start
        print(f"[handler] Video downloaded: {video_mb:.1f} MB in {elapsed:.1f}s")

        # ── 2. Build pipeline command ────────────────────────────────────────
        output_dir = os.path.join(work_dir, "output")
        cmd = [
            sys.executable, str(PIPELINE_SCRIPT),
            "--video",          video_path,
            "--output",         output_dir,
            "--sfm-backend",    sfm_backend,
            "--trainer",        trainer,
            "--fps",            str(fps),
            "--iterations",     str(iterations),
            "--max-resolution", str(max_resolution),
            "--oversample",     str(oversample),
        ]

        for arg_name, (flag, default) in _BACKEND_ARGS[sfm_backend].items():
            value = inp.get(arg_name, default)
            cmd += [flag, str(value)]

        backend_args_used = {
            arg: inp.get(arg, default)
            for arg, (_, default) in _BACKEND_ARGS[sfm_backend].items()
        }
        print(f"[handler] Backend: {sfm_backend}  |  args: {backend_args_used}")

        # ── 3. Run pipeline (with live progress) ─────────────────────────────
        runpod.serverless.progress_update(
            job, f"[1/6] Starting pipeline ({sfm_backend}, {fps}fps, "
                 f"{iterations} iters, {max_resolution}px) | {gpu_info}"
        )
        returncode, stdout, stderr = run_pipeline_with_progress(
            job, cmd, gpu_info, job_start, work_dir
        )
        print(stdout[-3000:])

        if returncode != 0:
            error_msg = (
                f"STDERR: {stderr[-2000:]}\n\nSTDOUT (last 1500): {stdout[-1500:]}"
                if stderr.strip() else stdout[-3000:]
            )
            return {"error": error_msg, "gpu": gpu_info}

        # ── 4. Locate PLY ────────────────────────────────────────────────────
        ply_path = os.path.join(output_dir, "model", f"export_{iterations}.ply")
        if not os.path.exists(ply_path):
            return {"error": f"PLY not found at {ply_path}.", "gpu": gpu_info}

        ply_size_mb = os.path.getsize(ply_path) / 1e6
        splat_count = count_ply_splats(ply_path)
        print(f"[handler] PLY size: {ply_size_mb:.1f} MB | splats: {splat_count:,}")

        # ── 5. Parse & log timing summary ────────────────────────────────────
        timings = parse_timings(stdout)
        sfm_result = parse_sfm_result(stdout)
        print(format_timings_summary(timings))

        # ── 6. Write timings.txt ──────────────────────────────────────────────
        from datetime import datetime
        job_id = job["id"]
        rows = [
            ("frame_extraction",   "Frame Extraction"),
            ("feature_extraction", "Feature Extraction"),
            ("feature_matching",   "Feature Matching"),
            ("sfm_reconstruction", "SfM Reconstruction"),
            ("undistortion",       "Image Undistortion"),
            ("training",           "3DGS Training"),
        ]
        timing_lines = [
            "=== 3DGS Pipeline Timings (Serverless) ===",
            f"Date:        {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"GPU:         {gpu_info}",
            f"Job ID:      {job_id}",
            f"Backend:     {sfm_backend}",
            f"Settings:    fps={fps}, iterations={iterations}, max_resolution={max_resolution}",
            "",
            "Step Timings:",
        ]
        for key, label in rows:
            s = timings.get(key, 0)
            timing_lines.append(f"  {label:<22} {s:8.2f}s  ({s/60:6.2f}m)")
        total_s = timings.get("total", 0)
        timing_lines += [
            "  " + "─" * 42,
            f"  {'TOTAL':<22} {total_s:8.2f}s  ({total_s/60:6.2f}m)",
        ]
        timings_txt = os.path.join(work_dir, "timings.txt")
        with open(timings_txt, "w") as f:
            f.write("\n".join(timing_lines) + "\n")

        # ── 7. Upload PLY + timings.txt to S3 ────────────────────────────────
        runpod.serverless.progress_update(job, f"[6/6] Uploading to S3... | {gpu_info}")
        s3_key = f"3dgs-outputs/{job_id}/export_{iterations}.ply"
        ply_url = upload_to_s3(ply_path, s3_key)
        timings_url = upload_to_s3(timings_txt, f"3dgs-outputs/{job_id}/timings.txt")

        return {
            "ply_url":      ply_url,
            "timings_url":  timings_url,
            "ply_size_mb":  round(ply_size_mb, 1),
            "splat_count":  splat_count,
            "sfm_result":   sfm_result,
            "job_id":       job_id,
            "gpu":          gpu_info,
            "timings":      timings,
        }

    except Exception as e:
        return {"error": str(e), "gpu": gpu_info}

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
