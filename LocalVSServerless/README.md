# Local vs Serverless — 3DGS Backend Comparison

Comparing three SfM backends, each tested on local GPU and RunPod serverless,
using the same video and settings.

## Test Video
- **File:** `~/Downloads/IMG_2188.MOV`
- **Settings:** `fps=1`, `iterations=1000`, `max_resolution=1280`

## Folder Structure

```
LocalVSServerless/
├── fastmap/         SIFT features + FastMap mapper (fast, good default)
│   ├── local/       → PLY from local RTX 3070 8GB
│   └── serverless/  → PLY downloaded from RunPod (RTX 4090)
├── mast3r/          Deep learning features + GLOMAP (best quality, slowest)
│   ├── local/
│   └── serverless/
└── hloc/            SuperPoint + LightGlue + COLMAP (best for low-texture scenes)
    ├── local/
    └── serverless/
```

## What Goes in Each Folder

Each `local/` and `serverless/` folder will contain:
- `export_1000.ply` — the trained 3DGS point cloud
- `timings.txt` — pipeline step timings

## Run Commands

### Local (run from repo root)

```bash
# fastmap
python3 video_to_3dgs.py \
  --video ~/Downloads/IMG_2188.MOV \
  --output LocalVSServerless/fastmap/local \
  --sfm-backend fastmap --fps 1 --iterations 1000 --max-resolution 1280

# mast3r
python3 video_to_3dgs.py \
  --video ~/Downloads/IMG_2188.MOV \
  --output LocalVSServerless/mast3r/local \
  --sfm-backend mast3r --fps 1 --iterations 1000 --max-resolution 1280

# hloc
python3 video_to_3dgs.py \
  --video ~/Downloads/IMG_2188.MOV \
  --output LocalVSServerless/hloc/local \
  --sfm-backend hloc --fps 1 --iterations 1000 --max-resolution 1280
```

### Serverless (RunPod)

```bash
export $(grep -v '^#' .env | xargs)

# Upload video once
aws s3 cp ~/Downloads/IMG_2188.MOV \
  s3://$AWS_S3_BUCKET/test-inputs/IMG_2188.MOV --region $AWS_S3_REGION

# Get pre-signed URL (valid 24h)
aws s3 presign s3://$AWS_S3_BUCKET/test-inputs/IMG_2188.MOV \
  --region $AWS_S3_REGION --expires-in 86400

# Submit job (change sfm_backend to mast3r / hloc as needed)
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"video_url": "<PRESIGNED_URL>", "sfm_backend": "fastmap",
       "fps": 1, "iterations": 1000, "max_resolution": 1280}}'

# Poll status
curl "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/<JOB_ID>" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"

# Download PLY into the right folder
aws s3 cp s3://$AWS_S3_BUCKET/3dgs-outputs/<JOB_ID>/export_1000.ply \
  LocalVSServerless/fastmap/serverless/export_1000.ply --region $AWS_S3_REGION
```

## Viewing Results

```bash
python3 -c "
import open3d as o3d
pcd = o3d.io.read_point_cloud('LocalVSServerless/fastmap/local/model/export_1000.ply')
print('Points:', len(pcd.points))
o3d.visualization.draw_geometries([pcd])
"
```
