# Video-to-3DGS RunPod Serverless — Project Status

## What This Project Does
Converts a video into a 3D Gaussian Splatting `.ply` model via a RunPod serverless
GPU endpoint. User-facing frontend is a **Telegram bot** (hosted on Railway) that guides
users through settings and delivers a download link when done.

Full pipeline: download video → extract frames → SfM reconstruction → 3DGS training
(Brush) → upload PLY to S3 → send presigned URL back to user.

---

## Key Files
| File | Purpose |
|------|---------|
| `gaussian-splatting/handler.py` | RunPod serverless handler (entry point) |
| `gaussian-splatting/video_to_3dgs.py` | Full pipeline script |
| `gaussian-splatting/Dockerfile` | Docker image definition |
| `gaussian-splatting/example_inputs.json` | Example job inputs for all 4 backends |
| `gaussian-splatting/telegram_bot/bot.py` | Telegram bot frontend |
| `gaussian-splatting/telegram_bot/Procfile` | Railway process definition (`worker: python bot.py`) |
| `gaussian-splatting/telegram_bot/requirements.txt` | Bot deps (python-telegram-bot, boto3, aiohttp) |
| `gaussian-splatting/telegram_bot/.env.example` | Env var template for bot |

---

## Docker Image (RunPod serverless)
- **Docker Hub:** `hjeij2000/video-to-3dgs:serverless`
- **Latest versioned tag:** `:v9` ✅ (2026-02-27)
  - SfM registration stats + splat count in bot output
  - RUST_BACKTRACE=1 for all commands (stack trace on Brush -11)
  - Explicit `--seed 42` on Brush command
- **Previous:** `:v8` ✅ (2026-02-27)
  - CUDA 12.4.1 base (was 12.8.1 — caused container start failure on some RTX 4090 nodes with driver <570)
  - torch 2.6.0+cu124 (was 2.7.0+cu128)
  - stderr deadlock fix in handler.py (fixes stuck WORKING worker)
  - torch seeds added to hloc + mast3r for reproducibility
  - digest: `sha256:565f9e6b90d7b9f0234d2b7d0e8403e8395822cdec9aa1acf7594647f253e103`
- **Previous:** `:v7` ✅ (2026-02-27, handler.py S3 botocore Config fix)
- **Last built:** 2026-02-27 (image size: 21.3GB, ~90 min build)
- **RunPod template:** Updated to `:v9` ✅ (template `mrgxwb470f`)
- **Local image status:** ✅ Removed (disk freed, 35GB build cache also pruned)
- **Build command (RELIABLE — use this):**
  ```bash
  # Step 1: Build with default docker builder (has network access, no DNS issues)
  docker build \
    --file gaussian-splatting/Dockerfile \
    --tag hjeij2000/video-to-3dgs:serverless \
    .
  # Step 2: Push (chunked upload, more reliable than buildx --push)
  docker push hjeij2000/video-to-3dgs:serverless
  # Step 3: Create versioned tag server-side (increment vN)
  docker buildx imagetools create hjeij2000/video-to-3dgs:serverless --tag hjeij2000/video-to-3dgs:vN
  # Step 4: Clean up local image
  docker image rm hjeij2000/video-to-3dgs:serverless
  ```
  **NOTE:** `docker buildx build --push` and `docker buildx build --load` both fail intermittently
  due to DNS resolution failures inside the buildx container. Use plain `docker build` instead.
  - No layer cache between `docker build` and buildx — full rebuild each time (~75 min)

---

## Telegram Bot
A 5-step wizard that lets non-technical users submit jobs and receive results.

### Flow
1. User sends a video file (≤50 MB), Google Drive share link, or direct URL
2. Wizard prompts: **backend → fps → resolution → backend arg → iterations**
3. Bot uploads video to S3 (`telegram-inputs/<uuid>/<filename>`) and generates a presigned URL
4. Submits job to RunPod, polls every 20s, edits the status message live
5. On completion: sends timings summary, then delivers `.ply` directly as a Telegram file attachment (≤45 MB); falls back to presigned S3 link for larger files

### Supported input types
- Video files via Telegram (mp4, mov, avi, mkv) — capped at 50 MB (Telegram bot limit)
- Google Drive share links (auto-converted to direct download URL)
- Any direct HTTP/HTTPS video URL

### Files
- **`telegram_bot/bot.py`** — full bot code (single file)
- **`telegram_bot/Procfile`** — `worker: python bot.py` (Railway process)
- **`telegram_bot/requirements.txt`** — `python-telegram-bot[webhooks]==21.9`, `boto3`, `aiohttp`
- **`telegram_bot/.env.example`** — template for required env vars

### Env vars required
```
TELEGRAM_BOT_TOKEN=...     # from BotFather
RUNPOD_API_KEY=...
RUNPOD_ENDPOINT_ID=uupefx2whvkg13
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_BUCKET=splats-bucket
AWS_S3_REGION=me-south-1
```

### Bugs fixed during development
| Commit | Fix |
|--------|-----|
| `e57f560` | aiohttp brotli decode error on RunPod API responses — added `Accept-Encoding: gzip, deflate` header |
| `e6f200b` | Telegram Markdown parse error — underscores in arg names (e.g. `match_overlap`) break formatting; escaped properly |
| `0f41248` | Silent crash on iterations step — added try/except + error handler |
| `0f3fa93` | Full 5-step wizard implemented (backend → fps → resolution → arg → iterations) |
| `6f8af43` | Show all timing steps in bot result message |
| `b221ea1` | Bot tasks dropped silently — switched to `ctx.application.create_task()` to keep tasks alive |
| `cbb88c0` | S3 presigned URL region fix in bot — added `botocore.Config(signature_version=s3v4, addressing_style=virtual)` |
| `f9e71b7` | PLY delivered as Telegram file attachment instead of bare link (non-technical users couldn't download from presigned URLs) |

---

## Railway Deployment (Telegram Bot)
The Telegram bot runs as a Railway **worker** (not a web service — no HTTP port needed).

### Deploy steps
1. Connect Railway project to the `gaussian-splatting/telegram_bot/` directory (or root with path override)
2. Railway detects `Procfile` → runs `python bot.py`
3. Set all env vars from the list above in Railway dashboard
4. Bot uses long-polling (`run_polling`) — no webhook setup required

### Notes
- Bot state (user sessions) is **in-memory only** — sessions are lost on restart
- If bot restarts mid-wizard, user must send the video again (session expired message shown)
- S3 uploads from bot go to `telegram-inputs/` prefix (separate from pipeline outputs at `3dgs-outputs/`)

---

## RunPod Setup
- **API domain:** `https://api.runpod.ai` *(NOT `.io` — that returns 404)*
- **Template ID:** `mrgxwb470f` (name: `video-to-3dgs`, currently on `:v7`)
- **Current Endpoint ID:** `uupefx2whvkg13` (name: `video-to-3dgs`)
- **GPUs:** ADA_24 + AMPERE_24
- **Workers:** 1 min / 2 max, FlashBoot OFF, idle timeout 10 min
- **Template env vars set:**
  - `AWS_ACCESS_KEY_ID` = `<YOUR_AWS_ACCESS_KEY_ID>`
  - `AWS_SECRET_ACCESS_KEY` = `<YOUR_AWS_SECRET_ACCESS_KEY>`
  - `AWS_S3_BUCKET` = `splats-bucket`
  - `AWS_S3_REGION` = `me-south-1`

### Useful API calls
```bash
# Submit job
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"video_url": "...", "sfm_backend": "mast3r", "fps": 1, "iterations": 7000, "max_resolution": 960}}'

# Check status
curl "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/{JOB_ID}" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"

# List endpoints (REST)
curl "https://rest.runpod.io/v1/endpoints" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"

# Download PLY from S3 (use AWS CLI — pre-signed URLs have region endpoint issues)
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY='...' \
  aws s3 cp s3://splats-bucket/3dgs-outputs/{JOB_ID}/export_1000.ply ./export_1000.ply --region me-south-1
```

---

## Session Log

### Session 10 (2026-02-27) — SfM stats, splat count, Brush diagnostics 🔄 IN PROGRESS
- **Auto-retry on Brush -11:** bot.py retries job once silently before reporting failure
- **SfM stats in bot output:** registered/total images + 3D point count (all backends)
- **Splat count in bot output:** reads PLY header vertex count
- **RUST_BACKTRACE=1:** added to run_command env — next -11 will give a real stack trace
- **Brush --seed 42:** explicit seed for reproducibility
- **:v9 build in progress**

### Session 9 (2026-02-27) — Bug hunt + :v8 build 🔄 IN PROGRESS
- **Workers force-stopped:** set workersMin=0, workersMax=0 via API (stuck worker + CUDA error)
- **Bug: container start failure** — `cuda>=12.8` requirement not met on some RTX 4090 nodes (driver <570). Fix: downgrade base to `cuda:12.4.1` (requires driver ≥550, much more compatible)
- **Bug: stderr pipe deadlock** — handler.py used `stderr=PIPE` but never drained it while reading stdout. COLMAP stderr fills 64KB OS buffer → subprocess blocks → handler hangs → worker stuck WORKING. Fix: redirect stderr to file in work_dir.
- **Bug: result inconsistency** — added `torch.manual_seed(42)` to hloc_sfm.py and mast3r_sfm.py to reduce non-determinism in deep learning feature matching
- **Bug: broken presigned URL** — Telegram Markdown V1 ate underscores in URL (`export_3000`→`export3000`, `aws4_request`→`aws4request`). Fix: removed `parse_mode="Markdown"` from URL-containing messages in bot.py ✅ (deployed, Railway redeploy needed)
- **:v8 deployed:** CUDA 12.4.1, torch 2.6.0+cu124, all fixes above ✅
- Workers restored: workersMin=1, workersMax=2

### Session 8 (2026-02-27) — PLY delivery fix + stale worker resolved ✅ COMPLETE
- **PLY delivery (UX fix):** Bot now sends `.ply` as a Telegram file attachment (`bot.send_document()`) instead of a presigned S3 link — non-technical users couldn't handle the bare URL. Files ≤45 MB sent directly; fallback to link for larger.
- **Commits:** `cbb88c0` (S3 botocore Config fix in bot) + `f9e71b7` (PLY as attachment) — pushed to `myfork/main`
- **Stale worker issue (Brush -11):** Intermittent SIGSEGV on RunPod despite same RTX 4090 and same settings. Root cause: warm workers running pre-`:v7` image. All workers confirmed updated to `:v7` via RunPod dashboard — issue resolved.
- **Railway redeploy needed:** Bot code (PLY attachment) is pushed to GitHub; Railway must redeploy to pick up the change.

### Session 7 (2026-02-27) — Local hloc test ✅ COMPLETE
- **Video:** Drive `1OvptV4p43OGIpKTpT8dQFdopDF6-7ejO` (172MB, 58s)
- **Settings:** hloc, fps=2, 960px, match_overlap=10, 3000 iters
- **Result:** `/tmp/3dgs_hloc_local/model/export_3000.ply` — **20MB, 86,214 points**
- **Timings:** frames 30.9s | SfM 89.2s | undistort 6.1s | Brush 36.3s | **total 162.5s**
- **Local deps issue fixed:** hloc requires `h5py`, `pycolmap`, `lightglue` — all present in `gaussian-splatting/venv` but NOT in system Python. Always run with `/home/ibrahim/gaussian-splatting/venv/bin/python3`

### Session 6 (2026-02-27) — Build + deploy :v7 ✅ COMPLETE
- **Goal:** Build new image with handler.py S3 fix and deploy to RunPod
- **Change:** `handler.py` — added `botocore.Config(signature_version=s3v4, addressing_style=virtual)` to fix S3 presigned URLs on `me-south-1` regional bucket (same fix already in bot.py)
- **Also committed:** `bot.py` — refactored `_run_job` to use `bot.send_message` instead of `query.edit_message_text` for result delivery
- **Build:** ~90 min, 25.6GB image, pruned 22GB build cache beforehand (disk: 68GB free → 16GB during export → 68GB free after cleanup)
- **Push:** `docker push` + `buildx imagetools create :v7` ✅
- **RunPod template updated:** `mrgxwb470f` → `:v7` ✅ (via PATCH REST API)
- **Disk after cleanup:** 68GB free (build cache pruned post-push)

### Session 5 (2026-02-26/27) — Telegram Bot ✅ COMPLETE
- **Goal:** Build a non-technical user interface for the pipeline
- **Built:** Full Telegram bot with 5-step wizard (7 commits)
- **Deployed on Railway** as a worker process
- **Key bugs fixed:** brotli decode, markdown escaping, task dropping, error handling
- **Input sources:** Telegram video upload, Google Drive links, direct URLs
- **Output:** timings summary + PLY presigned URL (24h) + supersplat.xyz link

### Session 4 (2026-02-26) — LocalVSServerless benchmark testing ✅ COMPLETE
- **Goal:** Compare fastmap / mast3r / hloc serverless outputs for same video (IMG_2188.MOV)
- **Settings:** fps=1 (mast3r/fastmap), fps=2 (hloc), iterations=7000, max_resolution=1280 (hloc+mast3r: 960)
- **Docker fixes (`:v4`→`:v5`→`:v6`):**
  - `:v4` — Added libboost for GLOMAP (but `docker build` used cached layers + `docker commit` overwrote ENTRYPOINT)
  - `:v5` — Fixed ENTRYPOINT via `docker commit --change='ENTRYPOINT [...]'` (was set to apt-get cmd!)
  - `:v6` — Added `libopenblas0-pthread` for GLOMAP (`libopenblas.so.0: not found`)
- **Root cause of all "workers idle, jobs IN_QUEUE" issues:** The `:v4` image had its ENTRYPOINT
  overwritten by `docker commit` to `bash -c "apt-get install boost..."`. Every worker ran apt-get
  then exited — handler.py never started, so workers appeared idle but never processed jobs.
- **Results:**
  - **fastmap** ✅ COMPLETED — `LocalVSServerless/fastmap/serverless/` (121.3MB PLY, 160.64s total, RTX 4090)
    - fps=1, `8f73a5dd-...`
  - **hloc** ✅ COMPLETED — `LocalVSServerless/hloc/serverless/` (80MB PLY, 100.48s total, RTX 4090)
    - fps=2, max_res=960, `20e317a2-...`
  - **mast3r** ✅ COMPLETED — `LocalVSServerless/mast3r/serverless/` (130MB PLY, 387.34s total, RTX 4090)
    - fps=1, max_res=960, `107dcd45-...` (SfM=308s dominant, 58 images, 54572 tracks)

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
  - Workaround: download via AWS CLI with `--region me-south-1` ← **known bug, fixed in Session 3**

### Session 1 (pre-2026-02-25) — Initial pipeline + bug fixes
- Built initial Docker image and RunPod serverless handler
- Fixed ffmpeg CUDA decode crash (see Bug #1)
- Fixed XDG_RUNTIME_DIR error (see Bug #2)
- Identified Brush SIGSEGV (see Bug #3), applied Dockerfile fix, triggered rebuild

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

### 4. Pre-signed S3 URL region mismatch ✅ FIXED in handler.py (Session 3)
- **Error:** `IllegalLocationConstraintException` when downloading via pre-signed URL
- **Cause:** Handler generated pre-signed URL using global S3 endpoint, but bucket is
  in `me-south-1` which requires the regional endpoint
- **Fix:** Pass `endpoint_url=f"https://s3.{region}.amazonaws.com"` when creating boto3 client
- **Also fixed in bot.py** — uses same regional endpoint pattern

### 5. Brush OOM on high-res/long videos ⚠️ (workaround known)
- **Error:** Brush crashes with SIGSEGV (-11) during training
- **Cause:** Too many frames or too high resolution exhausts GPU VRAM
- **Affected:** Videos > ~60 frames or > 1920px max resolution
- **Workaround:** Use fps=1 and max_resolution=1280 for long/4K videos
- **Proper fix:** Auto-detect frame count and resolution in handler, auto-scale fps/max_res

### 6. Stale worker image cache ✅ (fixed by re-tagging)
- **Error:** Worker uses old image, Vulkan not configured → immediate Brush SIGSEGV
- **Fix:** Create new versioned tag on Docker Hub and update RunPod template to force pull

### 7. GLOMAP missing libboost ✅ FIXED in :v4
- **Error:** `libboost_program_options.so.1.74.0: No such file or directory` (mast3r backend only)
- **Cause:** GLOMAP binary linked against boost libs not present in runtime image
- **Fix (Dockerfile):** Added `libboost-program-options1.74.0`, `libboost-filesystem1.74.0`, `libboost-graph1.74.0`

### 8. docker commit overwrites ENTRYPOINT ✅ FIXED in :v5
- **Error:** Workers appear "idle" but jobs stay IN_QUEUE forever
- **Cause:** `docker commit` without `--change='ENTRYPOINT ...'` copies the running container's CMD
  as the new image CMD, replacing the original ENTRYPOINT. Workers ran apt-get then exited.
- **Fix:** Always use `docker commit --change='ENTRYPOINT [...]' --change='CMD []'` when patching

### 9. GLOMAP missing libopenblas ✅ FIXED in :v6
- **Error:** `/usr/local/bin/glomap: error while loading shared libraries: libopenblas.so.0`
- **Fix (Dockerfile):** Added `libopenblas0-pthread` to runtime apt-get

### 10. buildx DNS failure inside builder container ⚠️
- **Error:** `Temporary failure in name resolution` during `pip install torch` inside buildx container
- **Cause:** Docker buildx builder runs in its own network namespace which sometimes loses DNS
- **Workaround:** Use plain `docker build` (runs in host network) then `docker push`

### 11. aiohttp brotli decode error (bot) ✅ FIXED
- **Error:** Bot crashes when reading RunPod API responses (brotli content-encoding not installed)
- **Fix (`bot.py`):** Added `NO_BR = {"Accept-Encoding": "gzip, deflate"}` header on all aiohttp calls

### 12. Telegram Markdown parse error (bot) ✅ FIXED
- **Error:** Markdown V1 parse error when arg name contains underscores (e.g. `match_overlap`)
- **Fix (`bot.py`):** Replace underscores with spaces before embedding in Markdown strings

### 14. Container start failure on RTX 4090 nodes with old drivers ✅ FIXED in :v8
- **Error:** `cuda>=12.8, please update your driver` — some RTX 4090 datacenter nodes have driver <570
- **Fix (Dockerfile):** Downgrade base from `cuda:12.8.1` → `cuda:12.4.1` (requires driver ≥550)
- **Also:** torch 2.7.0+cu128 → 2.6.0+cu124, dropped `10.0+PTX` from TORCH_CUDA_ARCH_LIST

### 15. Handler stderr pipe deadlock (stuck worker) ✅ FIXED in :v8
- **Error:** Worker stays in WORKING state indefinitely even with no active jobs
- **Cause:** `subprocess.Popen(stderr=PIPE)` — handler reads stdout line-by-line but never drains stderr. COLMAP/Brush fill the 64KB OS pipe buffer → subprocess blocks → handler deadlocks in `for line in proc.stdout` → never returns → RunPod worker stuck WORKING forever
- **Fix (handler.py):** Redirect stderr to a temp file in work_dir instead of PIPE

### 16. Result inconsistency across runs ⚠️ PARTIALLY FIXED in :v8
- **Symptom:** Same input + same settings → different PLY size and quality
- **Causes:** (1) PyTorch non-determinism in hloc LightGlue + MASt3R inference; (2) different GPU hardware across workers; (3) COLMAP RANSAC (already seeded at 0)
- **Fix:** Added `torch.manual_seed(42)` + `torch.cuda.manual_seed_all(42)` to hloc_sfm.py and mast3r_sfm.py
- **Remaining variation:** Cross-GPU hardware float differences (unfixable in software)

### 13. Bot tasks silently dropped ✅ FIXED
- **Error:** Job polling coroutine dropped — bot showed "submitted" but never sent result
- **Cause:** `asyncio.ensure_future()` doesn't keep tasks alive; they get garbage collected
- **Fix:** Use `ctx.application.create_task()` which the Application lifecycle keeps alive

---

## Known Issues / To Fix

### Brush OOM auto-scaling (Bug #5)
- Auto-detect frame count + resolution in handler, auto-reduce fps/max_res to fit VRAM
- Currently requires user to manually pick safe settings

### PLY delivery for large files (>45 MB)
- Bot sends file directly via Telegram for PLY ≤45 MB (most jobs)
- Falls back to presigned S3 link for larger PLYs — link still tricky for non-technical users
- Potential fix: upload to a public/permanent URL, or warn user upfront about file size

### Bot sessions lost on restart
- Sessions are in-memory dict — Railway restart loses all in-progress wizard sessions
- Users get "session expired" and must restart

---

## Handler Features (handler.py)
- **Backend-specific args:** Each SfM backend only accepts its own args:
  - `mast3r` → `window_size` (default 10)
  - `fastmap` / `colmap` / `hloc` → `match_overlap` (default 5)
- **Progress updates:** Live `runpod.serverless.progress_update()` calls on each pipeline step
- **Timing summary:** Pipeline timings parsed from stdout, included in job output JSON
- **timings.txt:** Written per job and uploaded to S3 (`3dgs-outputs/<job_id>/timings.txt`)
- **Error reporting:** On failure returns both stderr and last 1500 chars of stdout

---

## SfM Backends
| Backend | Description | Unique arg | Default FPS |
|---------|-------------|------------|-------------|
| `mast3r` | Best quality, slowest. Deep feature matching + GLOMAP | `window_size` | 1 |
| `fastmap` | Fast SIFT + FastMap mapper. Good default | `match_overlap` | 2 |
| `colmap` | SIFT + COLMAP incremental mapper. More reliable on tricky scenes | `match_overlap` | 2 |
| `hloc` | SuperPoint + LightGlue + COLMAP. Best for low-texture/metallic | `match_overlap` | 2 |

---

## Benchmark Results (LocalVSServerless — IMG_2188.MOV, RTX 4090)
| Backend | PLY Size | Total Time | SfM Time | Notes |
|---------|----------|------------|----------|-------|
| fastmap | 121.3 MB | 160.64s | ~60s | fps=1, 960px |
| hloc | 80.0 MB | 100.48s | ~40s | fps=2, 960px |
| mast3r | 130.0 MB | 387.34s | 308s | fps=1, 960px, 58 imgs, 54572 tracks |

---

## Credentials
- Stored locally in `gaussian-splatting/.env` (gitignored — never commit this file)
- `.env` contains: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_BUCKET`,
  `AWS_S3_REGION`, `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`
- Load in shell: `export $(grep -v '^#' .env | xargs)`
- Railway bot env vars set separately in Railway dashboard

---

## Storage Management
- Disk: 192GB total
- Local docker image removed ✅ (25.6GB freed)
- **LocalVSServerless folder:** ~330MB (fastmap 121MB + hloc 80MB + mast3r 130MB)
- **Post-build cleanup routine (run after every build):**
  1. `docker image rm hjeij2000/video-to-3dgs:serverless` — remove local image (already on Hub)
  2. `docker builder prune -f` — clear buildx builder cache
- `docker system prune -a -f` — nuclear option, only if disk critically low

---

## Next Steps
1. ✅ Fix pre-signed URL region issue (Bug #4)
2. ✅ Serverless benchmark testing (fastmap + hloc + mast3r)
3. ✅ Telegram bot with 5-step wizard (deployed on Railway)
4. ✅ Build + push `:v7` with handler.py S3 fix
5. ✅ Update RunPod template to `:v7`
6. ⏳ **Redeploy Railway bot** to pick up PLY-as-attachment fix (`f9e71b7`)
7. ⏳ **End-to-end test** Telegram bot → RunPod → PLY file delivered directly in chat
8. ⏳ Auto-scale fps/max_resolution in handler based on video length/resolution (fix Bug #5)
9. ⏳ MILESTONE 1: Local pipeline test (`video_to_3dgs.py` on local RTX 3070)
