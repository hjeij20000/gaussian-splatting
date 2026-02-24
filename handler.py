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
        "video_url":   "https://... or https://drive.google.com/...",  # required
        "fps":          2,        # frames/sec to extract   (default: 2)
        "window_size":  2,        # MASt3R pair window size (default: 2)
        "iterations":   7000,     # Brush training steps    (default: 7000)
        "sfm_backend":  "mast3r"  # mast3r | fastmap | colmap (default: mast3r)
    }

Output JSON:
    {"ply_url": "https://s3.amazonaws.com/...presigned..."}   on success
    {"error":  "...message..."}                               on failure
"""

import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path

import boto3
import gdown
import requests
import runpod

PIPELINE_SCRIPT = Path(__file__).parent / "video_to_3dgs.py"


# ── Helpers ──────────────────────────────────────────────────────────────────

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
    """Upload a file to S3 and return a 24-hour presigned URL."""
    bucket = os.environ["AWS_S3_BUCKET"]
    region = os.environ.get("AWS_S3_REGION", "us-east-1")
    s3 = boto3.client("s3", region_name=region)
    s3.upload_file(local_path, bucket, s3_key)
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=86400,
    )
    return url


# ── Handler ───────────────────────────────────────────────────────────────────

def handler(job):
    inp = job["input"]

    video_url   = inp.get("video_url")
    fps         = int(inp.get("fps", 2))
    window_size = int(inp.get("window_size", 2))
    iterations  = int(inp.get("iterations", 7000))
    sfm_backend = inp.get("sfm_backend", "mast3r")

    if not video_url:
        return {"error": "Missing required field: video_url"}

    work_dir = tempfile.mkdtemp(prefix="runpod_3dgs_")

    try:
        # ── 1. Download video ────────────────────────────────────────────────
        runpod.serverless.progress_update(job, "Downloading video...")
        video_ext = Path(video_url.split("?")[0]).suffix or ".mp4"
        video_path = os.path.join(work_dir, f"input{video_ext}")
        download_video(video_url, video_path)
        print(f"[handler] Video downloaded: {os.path.getsize(video_path) / 1e6:.1f} MB")

        # ── 2. Run pipeline ──────────────────────────────────────────────────
        runpod.serverless.progress_update(job, "Running 3DGS pipeline...")
        output_dir = os.path.join(work_dir, "output")
        cmd = [
            sys.executable, str(PIPELINE_SCRIPT),
            "--video",       video_path,
            "--output",      output_dir,
            "--sfm-backend", sfm_backend,
            "--fps",         str(fps),
            "--window-size", str(window_size),
            "--iterations",  str(iterations),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout[-3000:])

        if result.returncode != 0:
            return {"error": result.stderr[-3000:] or result.stdout[-3000:]}

        # ── 3. Locate PLY ────────────────────────────────────────────────────
        ply_path = os.path.join(output_dir, "model", f"export_{iterations}.ply")
        if not os.path.exists(ply_path):
            return {"error": f"PLY not found at {ply_path}. Pipeline may have failed."}

        ply_size_mb = os.path.getsize(ply_path) / 1e6
        print(f"[handler] PLY size: {ply_size_mb:.1f} MB")

        # ── 4. Upload to S3 ──────────────────────────────────────────────────
        runpod.serverless.progress_update(job, "Uploading result to S3...")
        job_id = job["id"]
        s3_key = f"3dgs-outputs/{job_id}/export_{iterations}.ply"
        ply_url = upload_to_s3(ply_path, s3_key)

        return {
            "ply_url":    ply_url,
            "ply_size_mb": round(ply_size_mb, 1),
            "job_id":     job_id,
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
