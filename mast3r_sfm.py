#!/usr/bin/env python3
"""
MASt3R-SfM: Replace COLMAP feature extraction + matching + FastMap/COLMAP mapper.

Pipeline:
  images/ → MASt3R deep matching → GLOMAP global mapping → sparse/0/ (COLMAP binary)

The output (cameras.bin, images.bin, points3D.bin) is compatible with
'colmap image_undistorter' and Brush training.

Usage:
    python mast3r_sfm.py --images /path/to/images --output /path/to/sparse/0
    python mast3r_sfm.py --images /path/to/images --output /path/to/sparse/0 --mapper pycolmap
"""

import os
import sys
import shutil
import argparse
import tempfile
from pathlib import Path

# Add MASt3R, DUSt3R, and dust3r_visloc to path
MAST3R_DIR = Path(__file__).parent.parent / 'mast3r'
sys.path.insert(0, str(MAST3R_DIR))
sys.path.insert(0, str(MAST3R_DIR / 'dust3r'))
sys.path.insert(0, str(MAST3R_DIR / 'dust3r' / 'dust3r_visloc'))


def run_sfm(image_dir: Path, output_dir: Path, model_name: str, device: str,
            image_size: int, window_size: int, shared_intrinsics: bool,
            conf_thr: float, pixel_tol: int, cache_dir: Path = None,
            mapper: str = 'glomap', glomap_bin: str = '/home/ibrahim/local/bin/glomap'):

    import torch
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    import pycolmap
    from kapture.converter.colmap.database_extra import kapture_to_colmap
    from kapture.converter.colmap.database import COLMAPDatabase

    from mast3r.model import AsymmetricMASt3R
    from mast3r.image_pairs import make_pairs
    from mast3r.colmap.mapping import (kapture_import_image_folder_or_list,
                                       run_mast3r_matching, pycolmap_run_mapper,
                                       glomap_run_mapper)
    from dust3r.utils.image import load_images

    torch.backends.cuda.matmul.allow_tf32 = True  # Ampere+ speedup

    # ── 1. Discover images ──────────────────────────────────────────────────
    exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix in exts])
    if len(image_files) < 3:
        raise RuntimeError(f"Need at least 3 images, found {len(image_files)}")
    print(f"[MASt3R] {len(image_files)} images in {image_dir}")

    # ── 2. Load model ───────────────────────────────────────────────────────
    print(f"[MASt3R] Loading model: {model_name}")
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    model.eval()

    # ── 3. Create image pairs ───────────────────────────────────────────────
    filelist = [str(f) for f in image_files]
    filelist_relpath = [f.name for f in image_files]
    root_path = str(image_dir)

    print(f"[MASt3R] Loading images at size={image_size} for pairing...")
    imgs = load_images(filelist, size=image_size, verbose=True)

    # swin = sliding window (cyclic=True for orbital captures / 360 video)
    scene_graph = f'swin-{window_size}'
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    print(f"[MASt3R] {len(pairs)} pairs created (scene_graph={scene_graph})")

    image_pairs = [
        (filelist_relpath[img1['idx']], filelist_relpath[img2['idx']])
        for img1, img2 in pairs
    ]

    # ── 4. Build kapture structure + COLMAP database ────────────────────────
    kdata = kapture_import_image_folder_or_list(
        (root_path, filelist_relpath), use_single_camera=shared_intrinsics
    )

    work_dir = cache_dir if cache_dir else Path(tempfile.mkdtemp(prefix='mast3r_sfm_'))
    work_dir.mkdir(parents=True, exist_ok=True)

    colmap_db_path = str(work_dir / 'colmap.db')
    if os.path.isfile(colmap_db_path):
        os.remove(colmap_db_path)
    os.makedirs(os.path.dirname(colmap_db_path), exist_ok=True)

    colmap_db = COLMAPDatabase.connect(colmap_db_path)
    try:
        kapture_to_colmap(kdata, root_path, tar_handler=None, database=colmap_db,
                          keypoints_type=None, descriptors_type=None,
                          export_two_view_geometry=False)

        # ── 5. MASt3R inference + matching ──────────────────────────────────
        print(f"[MASt3R] Running inference + matching on {len(image_pairs)} pairs...")
        colmap_image_pairs = run_mast3r_matching(
            model, image_size, 16, device,
            kdata, root_path, image_pairs, colmap_db,
            dense_matching=False, pixel_tol=pixel_tol, conf_thr=conf_thr,
            skip_geometric_verification=False, min_len_track=5,
        )
        colmap_db.close()
    except Exception as e:
        colmap_db.close()
        raise RuntimeError(f"MASt3R matching failed: {e}") from e

    if not colmap_image_pairs:
        raise RuntimeError(
            "No matched pairs survived — try lowering --conf-thr (current: {conf_thr})"
        )
    print(f"[MASt3R] {len(colmap_image_pairs)} valid pairs after filtering")

    # ── 6. Geometric verification ───────────────────────────────────────────
    pairs_txt = str(work_dir / 'pairs.txt')
    with open(pairs_txt, 'w') as f:
        for p1, p2 in colmap_image_pairs:
            f.write(f"{p1} {p2}\n")
    pycolmap.verify_matches(colmap_db_path, pairs_txt)

    # ── 7. Mapping ───────────────────────────────────────────────────────────
    recon_path = str(work_dir / 'reconstruction')
    if os.path.isdir(recon_path):
        shutil.rmtree(recon_path)
    os.makedirs(recon_path)

    if mapper == 'glomap':
        print(f"[MASt3R] Running GLOMAP global mapping (binary: {glomap_bin})...")
        glomap_run_mapper(glomap_bin, colmap_db_path, recon_path, root_path)
        # GLOMAP writes directly to recon_path/0
        recon_model_path = Path(recon_path) / '0'
        if not recon_model_path.exists():
            raise RuntimeError("GLOMAP produced no reconstruction — check matches")
        recon = pycolmap.Reconstruction(str(recon_model_path))
    else:
        print("[MASt3R] Running pycolmap incremental mapping...")
        pycolmap_run_mapper(colmap_db_path, recon_path, root_path)
        model_dirs = [d for d in Path(recon_path).iterdir() if d.is_dir()]
        if not model_dirs:
            raise RuntimeError("pycolmap mapper produced no reconstruction — check matches")
        recon_model_path = max(model_dirs, key=lambda d: pycolmap.Reconstruction(str(d)).num_reg_images())
        recon = pycolmap.Reconstruction(str(recon_model_path))

    print(f"[MASt3R] {recon.summary()}")
    print(f"[SFM_RESULT] registered={len(recon.images)} total={len(image_files)} points3d={len(recon.points3D)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for f in recon_model_path.iterdir():
        shutil.copy(f, output_dir / f.name)

    print(f"[MASt3R] Reconstruction saved to {output_dir}")
    return recon


def main():
    parser = argparse.ArgumentParser(
        description="MASt3R-SfM: deep feature matching + pycolmap reconstruction"
    )
    parser.add_argument('--images', required=True, help='Input image directory')
    parser.add_argument('--output', required=True,
                        help='Output directory for COLMAP binary reconstruction (cameras.bin etc.)')
    parser.add_argument('--model', default='naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
                        help='MASt3R model name (HuggingFace) or local checkpoint path')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--image-size', type=int, default=512,
                        help='Max image dimension fed to MASt3R (default: 512)')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Sliding window size for pair creation (default: 10, cyclic)')
    parser.add_argument('--no-shared-intrinsics', action='store_true',
                        help='Use per-image intrinsics instead of shared camera (not recommended for video)')
    parser.add_argument('--conf-thr', type=float, default=1.001,
                        help='Match confidence threshold — lower = more pairs kept (default: 1.001)')
    parser.add_argument('--pixel-tol', type=int, default=5,
                        help='Pixel tolerance for match filtering (default: 5)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory for intermediate files (default: system temp)')
    parser.add_argument('--mapper', type=str, default='glomap', choices=['glomap', 'pycolmap'],
                        help='SfM mapper backend: glomap (default, faster) or pycolmap (incremental)')
    parser.add_argument('--glomap-bin', type=str, default='/home/ibrahim/local/bin/glomap',
                        help='Path to glomap binary (default: /home/ibrahim/local/bin/glomap)')

    args = parser.parse_args()

    cache = Path(args.cache_dir) if args.cache_dir else None
    run_sfm(
        image_dir=Path(args.images),
        output_dir=Path(args.output),
        model_name=args.model,
        device=args.device,
        image_size=args.image_size,
        window_size=args.window_size,
        shared_intrinsics=not args.no_shared_intrinsics,
        conf_thr=args.conf_thr,
        pixel_tol=args.pixel_tol,
        cache_dir=cache,
        mapper=args.mapper,
        glomap_bin=args.glomap_bin,
    )


if __name__ == '__main__':
    main()
