# Local vs Serverless — 3DGS Backend Comparison

Comparing three SfM backends, each tested on local GPU and RunPod serverless
**in parallel** using the same video and settings. Only the SfM method changes.

## Test Video
- **File:** `~/Downloads/IMG_2188.MOV`
- **Settings:** `fps=2`, `iterations=7000`, `max_resolution=1280`

## Folder Structure

```
LocalVSServerless/
├── fastmap/         SIFT features + FastMap mapper (fast, good default)
│   ├── local/       → PLY + timings.txt from local RTX 3070 8GB
│   └── serverless/  → PLY + timings.txt downloaded from RunPod (RTX 4090)
├── mast3r/          Deep learning features + GLOMAP (best quality, slowest)
│   ├── local/
│   └── serverless/
└── hloc/            SuperPoint + LightGlue + COLMAP (best for low-texture scenes)
    ├── local/
    └── serverless/
```

Each folder contains after the run:
- `export_7000.ply`  — trained 3DGS point cloud
- `timings.txt`      — GPU used, per-step timings, total time

---

## Parallel Execution Strategy

Start all 6 runs at the same time:
1. Launch the 3 local runs in separate terminals (or background them)
2. Submit the 3 RunPod jobs via curl (they queue/run on the cluster)
3. Poll RunPod until all 3 jobs complete, then download PLY + timings.txt

---

## Step 1 — Upload video to S3 (one-time)

```bash
cd ~/gaussian-splatting
export $(grep -v '^#' .env | xargs)

aws s3 cp ~/Downloads/IMG_2188.MOV \
  s3://$AWS_S3_BUCKET/test-inputs/IMG_2188.MOV --region $AWS_S3_REGION

# Generate a pre-signed URL valid for 24h (paste into all 3 job submissions)
aws s3 presign s3://$AWS_S3_BUCKET/test-inputs/IMG_2188.MOV \
  --region $AWS_S3_REGION --expires-in 86400
```

---

## Step 2 — Launch local runs (3 terminals in parallel)

```bash
cd ~/gaussian-splatting

# Terminal 1 — fastmap
python3 video_to_3dgs.py \
  --video ~/Downloads/IMG_2188.MOV \
  --output LocalVSServerless/fastmap/local \
  --sfm-backend fastmap --fps 2 --iterations 7000 --max-resolution 1280

# Terminal 2 — mast3r
python3 video_to_3dgs.py \
  --video ~/Downloads/IMG_2188.MOV \
  --output LocalVSServerless/mast3r/local \
  --sfm-backend mast3r --fps 2 --iterations 7000 --max-resolution 1280

# Terminal 3 — hloc
python3 video_to_3dgs.py \
  --video ~/Downloads/IMG_2188.MOV \
  --output LocalVSServerless/hloc/local \
  --sfm-backend hloc --fps 2 --iterations 7000 --max-resolution 1280
```

Each run writes its PLY to `<backend>/local/model/export_7000.ply`
and its timings to `<backend>/local/timings.txt`.

---

## Step 3 — Submit RunPod jobs (replace PRESIGNED_URL from Step 1)

```bash
export $(grep -v '^#' .env | xargs)
VIDEO_URL="<PASTE_PRESIGNED_URL_HERE>"

# fastmap job
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d "{\"input\": {\"video_url\": \"$VIDEO_URL\", \"sfm_backend\": \"fastmap\",
       \"fps\": 2, \"iterations\": 7000, \"max_resolution\": 1280}}"

# mast3r job
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d "{\"input\": {\"video_url\": \"$VIDEO_URL\", \"sfm_backend\": \"mast3r\",
       \"fps\": 2, \"iterations\": 7000, \"max_resolution\": 1280}}"

# hloc job
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d "{\"input\": {\"video_url\": \"$VIDEO_URL\", \"sfm_backend\": \"hloc\",
       \"fps\": 2, \"iterations\": 7000, \"max_resolution\": 1280}}"
```

Note the job IDs from each response.

---

## Step 4 — Poll status

```bash
export $(grep -v '^#' .env | xargs)
curl "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/<JOB_ID>" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

When status is `COMPLETED`, the output JSON contains `ply_url`, `timings_url`, `gpu`, `timings`.

---

## Step 5 — Download PLY + timings.txt for each serverless job

```bash
export $(grep -v '^#' .env | xargs)

# fastmap
aws s3 cp s3://$AWS_S3_BUCKET/3dgs-outputs/<FASTMAP_JOB_ID>/export_7000.ply \
  LocalVSServerless/fastmap/serverless/export_7000.ply --region $AWS_S3_REGION
aws s3 cp s3://$AWS_S3_BUCKET/3dgs-outputs/<FASTMAP_JOB_ID>/timings.txt \
  LocalVSServerless/fastmap/serverless/timings.txt --region $AWS_S3_REGION

# mast3r
aws s3 cp s3://$AWS_S3_BUCKET/3dgs-outputs/<MAST3R_JOB_ID>/export_7000.ply \
  LocalVSServerless/mast3r/serverless/export_7000.ply --region $AWS_S3_REGION
aws s3 cp s3://$AWS_S3_BUCKET/3dgs-outputs/<MAST3R_JOB_ID>/timings.txt \
  LocalVSServerless/mast3r/serverless/timings.txt --region $AWS_S3_REGION

# hloc
aws s3 cp s3://$AWS_S3_BUCKET/3dgs-outputs/<HLOC_JOB_ID>/export_7000.ply \
  LocalVSServerless/hloc/serverless/export_7000.ply --region $AWS_S3_REGION
aws s3 cp s3://$AWS_S3_BUCKET/3dgs-outputs/<HLOC_JOB_ID>/timings.txt \
  LocalVSServerless/hloc/serverless/timings.txt --region $AWS_S3_REGION
```

---

## Step 6 — View results

```bash
# Swap path to compare any of the 6 outputs
python3 -c "
import open3d as o3d
pcd = o3d.io.read_point_cloud('LocalVSServerless/fastmap/local/model/export_7000.ply')
print('Points:', len(pcd.points))
o3d.visualization.draw_geometries([pcd])
"

# Read timings
cat LocalVSServerless/fastmap/local/timings.txt
cat LocalVSServerless/fastmap/serverless/timings.txt
```
