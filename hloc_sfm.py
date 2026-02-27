#!/usr/bin/env python3
"""
hloc SfM: SuperPoint feature extraction + LightGlue matching + COLMAP reconstruction.

A middle ground between FastMap (fast/SIFT) and MASt3R (slow/deep):
  - SuperPoint learned keypoints handle low-texture and metallic surfaces better than SIFT
  - LightGlue matching is fast and robust
  - COLMAP incremental mapper for reconstruction

Usage:
    python hloc_sfm.py --images /path/to/images --output /path/to/sparse/0
"""

import argparse
import sys
from pathlib import Path

HLOC_DIR = Path("/home/ibrahim/hloc")
sys.path.insert(0, str(HLOC_DIR))


def run_hloc_sfm(image_dir: Path, output_dir: Path,
                 match_overlap: int = 10,
                 feature_conf: str = 'superpoint_aachen',
                 matcher_conf: str = 'superpoint+lightglue'):
    """
    Run hloc pipeline: SuperPoint + LightGlue + COLMAP reconstruction.

    Args:
        image_dir:      Directory of input images.
        output_dir:     Where to write sparse/0/ (cameras/images/points3D.bin).
        match_overlap:  Number of sequential neighbors to match per frame.
        feature_conf:   hloc feature extractor config name.
        matcher_conf:   hloc matcher config name.
    """
    import torch
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    from hloc import extract_features, match_features, reconstruction
    import pycolmap

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    work_dir = output_dir.parent / 'hloc_workspace'
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Extract SuperPoint features ─────────────────────────────────────
    print("[hloc] Extracting SuperPoint features...")
    feat_conf = extract_features.confs[feature_conf]
    features_name = feat_conf['output']  # e.g. 'feats-superpoint-n4096-r1024'
    features_path = extract_features.main(
        conf=feat_conf,
        image_dir=image_dir,
        export_dir=work_dir,
    )

    # Use h5 keys as ground truth for image names (handles jpg/png/etc uniformly)
    from hloc.utils.io import list_h5_names
    image_names = sorted(list_h5_names(features_path))

    # ── 2. Create sequential pairs ──────────────────────────────────────────
    print(f"[hloc] Creating sequential pairs (overlap={match_overlap})...")
    pairs_path = work_dir / 'pairs.txt'

    with open(pairs_path, 'w') as f:
        for i, name in enumerate(image_names):
            for j in range(i + 1, min(i + 1 + match_overlap, len(image_names))):
                f.write(f"{name} {image_names[j]}\n")

    print(f"[hloc] {len(image_names)} images → {sum(1 for _ in open(pairs_path))} pairs")

    # ── 3. Match with LightGlue ─────────────────────────────────────────────
    # Pass features as a name string so hloc auto-constructs both h5 paths.
    print("[hloc] Matching features with LightGlue...")
    matches_path = match_features.main(
        conf=match_features.confs[matcher_conf],
        pairs=pairs_path,
        features=features_name,
        export_dir=work_dir,
    )

    # ── 4. COLMAP reconstruction ────────────────────────────────────────────
    print("[hloc] Running COLMAP incremental reconstruction...")
    recon = reconstruction.main(
        sfm_dir=output_dir,
        image_dir=image_dir,
        pairs=pairs_path,
        features=features_path,
        matches=matches_path,
        camera_mode=pycolmap.CameraMode.AUTO,
        image_list=image_names,
    )

    if recon is None or len(recon.images) == 0:
        raise RuntimeError("[hloc] Reconstruction failed — no images registered.")

    print(f"[hloc] Registered {len(recon.images)} / {len(image_names)} images")
    print(f"[hloc] {len(recon.points3D)} 3D points")

    # ── 5. Write COLMAP binary format to output_dir ─────────────────────────
    recon.write_binary(str(output_dir))
    print(f"[hloc] Sparse model written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="hloc SfM pipeline")
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--match-overlap', type=int, default=10)
    parser.add_argument('--feature-conf', default='superpoint_aachen',
                        choices=['superpoint_aachen', 'superpoint_max', 'disk', 'aliked-n16'])
    parser.add_argument('--matcher-conf', default='superpoint+lightglue',
                        choices=['superpoint+lightglue', 'disk+lightglue',
                                 'aliked+lightglue', 'superglue'])
    args = parser.parse_args()

    run_hloc_sfm(
        image_dir=Path(args.images),
        output_dir=Path(args.output),
        match_overlap=args.match_overlap,
        feature_conf=args.feature_conf,
        matcher_conf=args.matcher_conf,
    )


if __name__ == '__main__':
    main()
