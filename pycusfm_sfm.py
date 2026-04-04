#!/usr/bin/env python3
"""
PyCuSfM: GPU-accelerated SIFT feature extraction + matching + COLMAP reconstruction.

Faster than CPU-based COLMAP SIFT on large frame sets thanks to CUDA-accelerated
feature extraction and matching. Uses pycolmap incremental mapper for reconstruction,
so output format is identical to the fastmap/colmap/hloc backends.

Usage:
    python pycusfm_sfm.py --images /path/to/images --output /path/to/sparse/0
"""

import argparse
import sys
from pathlib import Path


def run_pycusfm_sfm(image_dir: Path, output_dir: Path, match_overlap: int = 5):
    """
    Run PyCuSfM pipeline: GPU-SIFT extraction + GPU matching + pycolmap reconstruction.

    Args:
        image_dir:      Directory of input images.
        output_dir:     Where to write sparse/0/ (cameras.bin, images.bin, points3D.bin).
        match_overlap:  Number of sequential neighbours to match per frame.
    """
    import pycolmap
    import pycusfm

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    work_dir = output_dir.parent / 'pycusfm_workspace'
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix in exts])
    if len(image_files) < 3:
        raise RuntimeError(f"Need at least 3 images, found {len(image_files)}")
    print(f"[PyCuSfM] {len(image_files)} images in {image_dir}")

    db_path = work_dir / 'database.db'
    if db_path.exists():
        db_path.unlink()

    # ── 1. GPU-accelerated SIFT feature extraction ──────────────────────────
    print("[PyCuSfM] Extracting GPU-SIFT features...")
    pycusfm.extract_features(
        image_dir=str(image_dir),
        database_path=str(db_path),
    )

    # ── 2. GPU-accelerated sequential matching ───────────────────────────────
    print(f"[PyCuSfM] Sequential matching (overlap={match_overlap})...")
    pycusfm.match_sequential(
        database_path=str(db_path),
        overlap=match_overlap,
    )

    # ── 3. Geometric verification ────────────────────────────────────────────
    print("[PyCuSfM] Running geometric verification...")
    pycolmap.verify_matches(
        str(db_path),
        pycolmap.TwoViewGeometryOptions(),
    )

    # ── 4. Incremental reconstruction with pycolmap ──────────────────────────
    print("[PyCuSfM] Running pycolmap incremental reconstruction...")
    recon_path = work_dir / 'reconstruction'
    recon_path.mkdir(parents=True, exist_ok=True)

    reconstructions = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(image_dir),
        output_path=str(recon_path),
    )

    if not reconstructions:
        raise RuntimeError("[PyCuSfM] Reconstruction failed — no models produced.")

    # Pick the largest model
    best = max(reconstructions.values(), key=lambda r: r.num_reg_images())

    n_images = len(image_files)
    print(f"[PyCuSfM] Registered {best.num_reg_images()} / {n_images} images")
    print(f"[PyCuSfM] {len(best.points3D)} 3D points")
    print(f"[SFM_RESULT] registered={best.num_reg_images()} total={n_images} points3d={len(best.points3D)}")

    # ── 5. Write COLMAP binary format to output_dir ─────────────────────────
    best.write_binary(str(output_dir))
    print(f"[PyCuSfM] Sparse model written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="PyCuSfM pipeline")
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--match-overlap', type=int, default=5)
    args = parser.parse_args()

    run_pycusfm_sfm(
        image_dir=Path(args.images),
        output_dir=Path(args.output),
        match_overlap=args.match_overlap,
    )


if __name__ == '__main__':
    main()
