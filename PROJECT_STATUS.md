# Video-to-3DGS RunPod Serverless — Project Status

## What This Project Does
Converts a video URL into a 3D Gaussian Splatting `.ply` model, running as a
RunPod serverless endpoint. The pipeline is: download video → extract frames →
SfM reconstruction → 3DGS training (Brush) → upload PLY to S3.

---

## Key Files
| File | Purpose |
|------|---------|
| `gaussian-splatting/handler.py` | RunPod serverless handler (entry point) |
| `gaussian-splatting/video_to_3dgs.py` | Full pipeline script |
| `gaussian-splatting/Dockerfile` | Docker image definition |
| `gaussian-splatting/example_inputs.json` | Example job inputs for all 4 backends |

---

## Docker Image
- **Docker Hub:** `hjeij2000/video-to-3dgs:serverless`
- **Latest versioned tag:** `:v4` (2026-02-26, added libboost for mast3r)
- **Last built:** 2026-02-26 (~13:17–14:30, image size: 25.6GB)
- **Build command (RELIABLE — use this):**
  ```bash
  # Step 1: Build with default docker builder (has network access, no DNS issues)
  docker build \
    --file gaussian-splatting/Dockerfile \
    --tag hjeij2000/video-to-3dgs:serverless \
    .
  # Step 2: Push (chunked upload, more reliable than buildx --push)
  docker push hjeij2000/video-to-3dgs:serverless
  # Step 3: Create versioned tag server-side
  docker buildx imagetools create hjeij2000/video-to-3dgs:serverless --tag hjeij2000/video-to-3dgs:vN
  ```
  **NOTE:** `docker buildx build --push` and `docker buildx build --load` both fail intermittently
  due to DNS resolution failures inside the buildx container. Use plain `docker build` instead.
  - No layer cache between `docker build` and buildx — full rebuild each time (~75 min)
  - After build + push, remove local image: `docker image rm hjeij2000/video-to-3dgs:serverless`

---

## RunPod Setup
- **API domain:** `https://api.runpod.ai` *(NOT `.io` — that returns 404)*
- **Template ID:** `mrgxwb470f` (name: `video-to-3dgs`, currently on `:v4`)
- **Current Endpoint ID:** `6knqxtvmxxsbur` (name: `video-to-3dgs -fb`)
- **GPUs:** RTX 4090 (primary) + RTX 3090 (fallback)
- **Workers:** 0 min / 3 max, FlashBoot enabled, idle timeout 5 min
- **Template env vars set:**
  - `AWS_ACCESS_KEY_ID` = `<YOUR_AWS_ACCESS_KEY_ID>`
  - `AWS_SECRET_ACCESS_KEY` = `<YOUR_AWS_SECRET_ACCESS_KEY>`
  - `AWS_S3_BUCKET` = `splats-bucket`
  - `AWS_S3_REGION` = `me-south-1`

### Useful API calls
```bash
# Submit job
curl -X POST "https://api.runpod.ai/v2/6knqxtvmxxsbur/run" \
  -H "Authorization: Bearer <YOUR_RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"input": {"video_url": "...", "sfm_backend": "fastmap", "fps": 2, "iterations": 1000}}'

# Check status
curl "https://api.runpod.ai/v2/6knqxtvmxxsbur/status/{JOB_ID}" \
  -H "Authorization: Bearer <YOUR_RUNPOD_API_KEY>"

# List endpoints (REST)
curl "https://rest.runpod.io/v1/endpoints" \
  -H "Authorization: Bearer <YOUR_RUNPOD_API_KEY>"

# Download PLY from S3 (use AWS CLI — pre-signed URLs have region endpoint issues)
AWS_ACCESS_KEY_ID=<YOUR_AWS_ACCESS_KEY_ID> AWS_SECRET_ACCESS_KEY='<YOUR_AWS_SECRET_ACCESS_KEY>' \
  aws s3 cp s3://splats-bucket/3dgs-outputs/{JOB_ID}/export_1000.ply ./export_1000.ply --region me-south-1
```

---

## Session Log

### Session 4 (2026-02-26) — LocalVSServerless benchmark testing
- **Goal:** Compare fastmap / mast3r / hloc serverless outputs for same video (IMG_2188.MOV)
- **Settings:** fps=2 (fastmap/hloc at fps=1 due to OOM), iterations=7000, max_resolution=1280 (hloc: 960)
- **Docker rebuild (`:v4`):** Added `libboost-program-options1.74.0`, `libboost-filesystem1.74.0`,
  `libboost-graph1.74.0` to fix mast3r/GLOMAP crash (`libboost_program_options.so.1.74.0: not found`)
- **Build issues:**
  - buildx builder DNS failed (transient network issue in container) → switched to plain `docker build`
  - Full rebuild took ~75 min (no shared cache between buildx and default builder)
- **Results so far:**
  - **fastmap** ✅ COMPLETED — `LocalVSServerless/fastmap/serverless/` (121.3MB PLY, 160.64s total, RTX 4090)
    - fps=1, 58 frames, `8f73a5dd-...`
  - **hloc** ✅ COMPLETED — `LocalVSServerless/hloc/serverless/` (80MB PLY, 100.48s total, RTX 4090)
    - fps=2, max_res=960, `20e317a2-...` (feat extract/matching timings = 0 — hloc handles these inside SfM step)
  - **mast3r** ⏳ IN_QUEUE — `41d4e61b-86fc-42db-a1d5-f5f140972353-e2`, fps=1, iterations=7000, max_res=1280

### Session 3 (2026-02-26) — IMG_2188 real-world test
- **Discovered Bug #4** (pre-signed URL region issue) — confirmed broken on download
- **Discovered Bug #5** (stale image cache on worker `wk5qtlitw4nwlw`) — old image
  had no Vulkan/XDG fix, causing immediate Brush crash on any job hitting that worker
- **Fix:** Created `:v2` tag on Docker Hub server-side (no local download) via
  `docker buildx imagetools create`, updated RunPod template from `:serverless` → `:v2`
  to force all workers to pull fresh image
- **Discovered Bug #6** (Brush OOM on large video) — `IMG_2188.MOV` is 2816x1584,
  57.9s at 24fps. At fps=2 → 115 frames at max_res=1920, Brush crashes with SIGSEGV
  (-11) due to GPU memory exhaustion
- **Fix:** Reduced to fps=1 (58 frames) + max_resolution=1280 → fits in GPU VRAM
- **Test job result:** `71b86608-673e-4caf-bb44-6b741b4c334e-e1` — **COMPLETED ✅**
  - Video: IMG_2188.MOV (2816x1584, 57.9s, iPhone HEVC)
  - Settings: fastmap, fps=1, iterations=1000, max_resolution=1280
  - PLY: 5.5MB uploaded to S3, downloaded and viewed in Open3D
  - Timings: frame 13.86s | feat extract 16.06s | matching 44.73s |
    SfM 53.71s | training 12s | undistortion 0.97s | **total 141.32s**

### Session 1 (pre-2026-02-25) — Initial pipeline + bug fixes
- Built initial Docker image and RunPod serverless handler
- Fixed ffmpeg CUDA decode crash (see Bug #1)
- Fixed XDG_RUNTIME_DIR error (see Bug #2)
- Identified Brush SIGSEGV (see Bug #3), applied Dockerfile fix, triggered rebuild

### Session 3 (2026-02-26) — IMG_2188 real-world test
- **Discovered Bug #4** (pre-signed URL region issue) — confirmed broken on download
- **Discovered Bug #5** (stale image cache on worker `wk5qtlitw4nwlw`) — old image
  had no Vulkan/XDG fix, causing immediate Brush crash on any job hitting that worker
- **Fix:** Created `:v2` tag on Docker Hub server-side (no local download) via
  `docker buildx imagetools create`, updated RunPod template from `:serverless` → `:v2`
  to force all workers to pull fresh image
- **Discovered Bug #6** (Brush OOM on large video) — `IMG_2188.MOV` is 2816x1584,
  57.9s at 24fps. At fps=2 → 115 frames at max_res=1920, Brush crashes with SIGSEGV
  (-11) due to GPU memory exhaustion
- **Fix:** Reduced to fps=1 (58 frames) + max_resolution=1280 → fits in GPU VRAM
- **Test job result:** `71b86608-673e-4caf-bb44-6b741b4c334e-e1` — **COMPLETED ✅**
  - Video: IMG_2188.MOV (2816x1584, 57.9s, iPhone HEVC)
  - Settings: fastmap, fps=1, iterations=1000, max_resolution=1280
  - PLY: 5.5MB uploaded to S3, downloaded and viewed in Open3D
  - Timings: frame 13.86s | feat extract 16.06s | matching 44.73s |
    SfM 53.71s | training 12s | undistortion 0.97s | **total 141.32s**

### Session 2 (2026-02-25)
- **Confirmed build completed** at 15:35 (25.6GB image, already pushed to Docker Hub)
- **Adopted registry-cache build strategy:** cache stored on Docker Hub (`:buildcache`),
  no local disk bloat; `--push` flag eliminates separate push step
- **Ran post-build cleanup:**
  - `docker image rm hjeij2000/video-to-3dgs:serverless` → freed 25.6GB
  - `docker builder prune -f` → freed 39.37GB
- **Submitted test job** `9330028c-452a-4209-94c5-0adc25917f37-e1`
  - Backend: `fastmap`, fps: 2, iterations: 1000, video: Big Buck Bunny 360p 10s
  - **Result: COMPLETED ✅** — Brush trained successfully, no SIGSEGV
  - PLY (1.1MB) uploaded to S3
  - Timings: frame 0.51s | feat extract 4.7s | matching 7.78s | SfM 41.07s | training 9.27s | total 63.47s
- **Downloaded and viewed PLY** locally using Open3D (`python3 -c "import open3d..."`)
  - Pre-signed URL from job output failed with `IllegalLocationConstraintException`
    (S3 regional bucket `me-south-1` incompatible with global endpoint in signed URL)
  - Workaround: download via AWS CLI with `--region me-south-1` ← **known bug to fix in handler**
- **Next test planned:** run pipeline on `~/Downloads/IMG_2188.mov` (real-world video)

---

## Bugs Fixed

### 1. ffmpeg CUDA decode crash ✅
- **Error:** `CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected`
- **Cause:** ffmpeg used `-hwaccel cuda -c:v hevc_cuvid`, fails on RunPod workers
- **Fix (`video_to_3dgs.py`):** Removed hardware decode flags; uses software decode

### 2. XDG_RUNTIME_DIR not set ✅
- **Error:** `error: XDG_RUNTIME_DIR not set in the environment.` (4× from wgpu)
- **Cause:** Brush/wgpu looks for the runtime dir before starting
- **Fix (`video_to_3dgs.py` `run_command()`):** Added `env.setdefault("XDG_RUNTIME_DIR", "/tmp")`

### 3. Brush SIGSEGV (exit code -11) ✅
- **Error:** `Training 3DGS model with Brush (1000 steps) failed with code -11`
- **Cause:** Brush uses wgpu/Vulkan. Container only gets CUDA `compute` capability by
  default — no Vulkan/graphics driver is mounted. wgpu crashes trying to init Vulkan.
- **Fix (Dockerfile):**
  1. `ENV NVIDIA_DRIVER_CAPABILITIES=all` → mounts graphics/Vulkan libs at runtime
  2. `/usr/share/vulkan/icd.d/nvidia_icd.json` pointing to `libGLX_nvidia.so.0`
     → tells Vulkan where to find the NVIDIA hardware ICD
- **Verified fixed** by test job `9330028c-...` (2026-02-25) — Brush trained in 9.27s

---

## Known Issues / To Fix

### 7. GLOMAP missing libboost ✅ FIXED in :v4
- **Error:** `libboost_program_options.so.1.74.0: No such file or directory` (mast3r backend only)
- **Cause:** GLOMAP binary linked against boost libs not present in runtime image
- **Fix (Dockerfile):** Added to runtime apt-get: `libboost-program-options1.74.0`,
  `libboost-filesystem1.74.0`, `libboost-graph1.74.0`
- **Rebuilt as `:v4`** (2026-02-26)

### 6. buildx DNS failure inside builder container ⚠️
- **Error:** `Temporary failure in name resolution` during `pip install torch` inside buildx container
- **Cause:** Docker buildx builder runs in its own network namespace which sometimes loses DNS
- **Workaround:** Use plain `docker build` (runs in host network) then `docker push`
- Full rebuild takes ~75 min with default builder (no cache sharing with buildx)

### 5. Brush OOM on high-res/long videos ⚠️ (workaround known)
- **Error:** Brush crashes with SIGSEGV (-11) during training
- **Cause:** Too many frames or too high resolution exhausts GPU VRAM
- **Affected:** Videos > ~60 frames or > 1920px max resolution
- **Workaround:** Use fps=1 and max_resolution=1280 for long/4K videos
- **Proper fix:** Auto-detect frame count and resolution in handler, auto-scale fps/max_res

### 4. Pre-signed S3 URL region mismatch ⚠️
- **Error:** `IllegalLocationConstraintException` when downloading via pre-signed URL
- **Cause:** Handler generates pre-signed URL using global S3 endpoint, but bucket is
  in `me-south-1` which requires the regional endpoint (`s3.me-south-1.amazonaws.com`)
- **Workaround:** Download via AWS CLI with `--region me-south-1` (see Useful API calls above)
- **Fix needed in `handler.py`:** Pass `endpoint_url='https://s3.me-south-1.amazonaws.com'`
  when creating the boto3 client for pre-signed URL generation

---

## Handler Features (handler.py)
- **Backend-specific args:** Each SfM backend only accepts its own args:
  - `mast3r` → `window_size` (default 10)
  - `fastmap` / `colmap` / `hloc` → `match_overlap` (default 5)
- **Timing summary:** Pipeline timings parsed from stdout, included in job output JSON
- **Error reporting:** On failure returns both stderr and last 1500 chars of stdout
- **Common args:** `video_url`, `fps`, `iterations`, `sfm_backend`, `max_resolution`, `oversample`

---

## SfM Backends
| Backend | Description | Unique arg |
|---------|-------------|------------|
| `mast3r` | Best quality, slowest. Deep feature matching + GLOMAP | `window_size` |
| `fastmap` | Fast SIFT + FastMap mapper. Good default | `match_overlap` |
| `colmap` | SIFT + COLMAP incremental mapper. More reliable on tricky scenes | `match_overlap` |
| `hloc` | SuperPoint + LightGlue + COLMAP. Best for low-texture/metallic | `match_overlap` |

---

## Credentials
- Stored locally in `gaussian-splatting/.env` (gitignored — never commit this file)
- `.env` contains: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_BUCKET`,
  `AWS_S3_REGION`, `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`
- Load in shell: `export $(grep -v '^#' .env | xargs)`

---

## Testing Milestones

### MILESTONE 1 — Local pipeline test ⬜ TODO
Goal: confirm `video_to_3dgs.py` runs end-to-end on local GPU (RTX 3070 8GB)

```bash
cd ~/gaussian-splatting
python3 video_to_3dgs.py \
  --video ~/Downloads/IMG_2188.MOV \
  --output /tmp/3dgs_local_test \
  --sfm-backend fastmap \
  --fps 1 \
  --iterations 1000 \
  --max-resolution 1280
```

Expected output: `/tmp/3dgs_local_test/model/export_1000.ply`

View result:
```bash
python3 -c "
import open3d as o3d
pcd = o3d.io.read_point_cloud('/tmp/3dgs_local_test/model/export_1000.ply')
print('Points:', len(pcd.points))
o3d.visualization.draw_geometries([pcd])
"
```

Record: frame count, timings, PLY size, visual quality notes.

---

### MILESTONE 2 — RunPod pipeline test ⬜ TODO
Goal: confirm the same video produces a good result on the serverless endpoint

Step 1 — Upload video to S3 so RunPod can fetch it:
```bash
export $(grep -v '^#' .env | xargs)
aws s3 cp ~/Downloads/IMG_2188.MOV \
  s3://$AWS_S3_BUCKET/test-inputs/IMG_2188.MOV \
  --region $AWS_S3_REGION
```

Step 2 — Generate a pre-signed download URL (valid 24h):
```bash
aws s3 presign s3://$AWS_S3_BUCKET/test-inputs/IMG_2188.MOV \
  --region $AWS_S3_REGION \
  --expires-in 86400
# Copy the URL output — use it as video_url below
```

Step 3 — Submit job:
```bash
export $(grep -v '^#' .env | xargs)
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "video_url": "<PASTE_PRESIGNED_URL>",
      "sfm_backend": "fastmap",
      "fps": 1,
      "iterations": 1000,
      "max_resolution": 1280
    }
  }'
# Note the job ID from the response
```

Step 4 — Poll status:
```bash
export $(grep -v '^#' .env | xargs)
curl "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/<JOB_ID>" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

Step 5 — Download PLY from S3:
```bash
export $(grep -v '^#' .env | xargs)
aws s3 cp s3://$AWS_S3_BUCKET/3dgs-outputs/<JOB_ID>/export_1000.ply \
  /tmp/3dgs_runpod_test.ply --region $AWS_S3_REGION
```

Step 6 — View and compare with local result:
```bash
python3 -c "
import open3d as o3d
pcd = o3d.io.read_point_cloud('/tmp/3dgs_runpod_test.ply')
print('Points:', len(pcd.points))
o3d.visualization.draw_geometries([pcd])
"
```

---

### MILESTONE 3 — Quality improvement ⬜ TODO (after both tests pass)
- Increase iterations: 1000 → 7000 (better density/quality, ~10× longer training)
- Test mast3r backend (best quality SfM, slower)
- Test hloc backend (best for low-texture scenes)
- Auto-scale fps/max_resolution in handler (fix Bug #5)

---

## Next Steps
1. ~~Fix pre-signed URL region issue (Bug #4)~~ ✅ Fixed in handler.py (session 3)
2. ~~MILESTONE 2 serverless testing~~ ✅ fastmap + hloc COMPLETED (session 4)
3. ⏳ **Poll mast3r job** `41d4e61b-86fc-42db-a1d5-f5f140972353-e2` until complete → download to `LocalVSServerless/mast3r/serverless/`
4. Compare all 3 PLYs in Open3D: fastmap (121MB) vs hloc (80MB) vs mast3r (TBD)
5. Clean up docker image to free ~25.6GB: `docker image rm hjeij2000/video-to-3dgs:serverless`

---

## Storage Management
- Disk: 192GB total, **currently 160GB used, 22GB free (89%)**
- Local docker image `hjeij2000/video-to-3dgs:serverless` = 25.6GB (safely removable — already on Docker Hub)
- **Run to free ~25.6GB:** `docker image rm hjeij2000/video-to-3dgs:serverless`
- **LocalVSServerless folder:** 195MB (fastmap 121MB + hloc 80MB + mast3r TBD)
- **Post-build cleanup routine (run after every build):**
  1. `docker image rm hjeij2000/video-to-3dgs:serverless` — remove local image (already on Hub)
  2. `docker builder prune -f` — clear buildx builder cache (~7GB currently)
- `docker system prune -a -f` — nuclear option, only if disk critically low
