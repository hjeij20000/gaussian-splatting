# Video-to-3DGS RunPod Serverless — Project Status

## What This Project Does
Converts a video into a 3D Gaussian Splatting `.ply` model via a RunPod serverless
GPU endpoint. User-facing frontend is a **Telegram bot** (hosted on Railway) that guides
users through settings and delivers a download link when done.

Full pipeline: download video → extract frames → SfM reconstruction → [undistort] → 3DGS training → upload PLY to S3 → send to user.

---

## Key Files
| File | Purpose |
|------|---------|
| `gaussian-splatting/handler.py` | RunPod serverless handler (entry point) |
| `gaussian-splatting/video_to_3dgs.py` | Full pipeline script |
| `gaussian-splatting/Dockerfile` | Docker image definition |
| `gaussian-splatting/example_inputs.json` | Example job inputs for all backends |
| `gaussian-splatting/mast3r_sfm.py` | MASt3R SfM backend |
| `gaussian-splatting/hloc_sfm.py` | hloc SfM backend |
| `gaussian-splatting/pycusfm_sfm.py` | pyCuSFM backend (placeholder — see note) |
| `gaussian-splatting/telegram_bot/bot.py` | Telegram bot frontend |
| `gaussian-splatting/telegram_bot/Procfile` | Railway process definition (`worker: python bot.py`) |
| `gaussian-splatting/telegram_bot/requirements.txt` | Bot deps (python-telegram-bot, boto3, aiohttp) |
| `gaussian-splatting/telegram_bot/.env.example` | Env var template for bot |
| `Luminance-GS/gsplat/` | gsplat repo (main branch, v1.5.3, patched for modern pycolmap) |

---

## Docker Image (RunPod serverless)
- **Docker Hub:** `hjeij2000/video-to-3dgs:serverless`
- **Latest deployed tag:** `:v11` ✅ (2026-03-01)
- **Next planned tag:** `:v12` ⏳ — adds 3DGUT trainer + pycusfm_sfm.py (Dockerfile ready, not built yet)

### :v12 changes (pending build)
- 3DGUT training backend (gsplat MCMC + `--with_ut --with_eval3d`)
- pycusfm_sfm.py added (placeholder for when NVIDIA adds unposed monocular support)
- Telegram bot upgraded to 6-step wizard (adds trainer selection: Brush vs 3DGUT)
- `trainer` input field in handler.py (`brush` default, `3dgut`)
- gsplat built from `Luminance-GS/gsplat` main in builder stage
- New deps: imageio[ffmpeg], viser, nerfview, tyro, scikit-learn, torchmetrics, tensorboard, tensorly, splines, fused-ssim (rahul-goel fork), fused-bilagrid

### :v11 ✅ (2026-03-01)
- **FIXED:** Removed `RUST_BACKTRACE=1` — caused all Brush jobs to SIGSEGV (-11)
- Brush retry: up to 3 attempts on -11, sleep only on retries (10s/20s)
- Always include AWS env vars in saveTemplate calls (`env: []` wipes them)

### :v10 ⚠️ (2026-02-28) — still broken, RUST_BACKTRACE=1 present
### :v9 ⚠️ (2026-02-28) — REGRESSED, RUST_BACKTRACE=1 added + template env vars wiped
### :v8 ✅ (2026-02-27)
- CUDA 12.4.1 base (was 12.8.1 — failed on RTX 4090 nodes with driver <570)
- torch 2.6.0+cu124; stderr deadlock fix; torch seeds for reproducibility
- digest: `sha256:565f9e6b90d7b9f0234d2b7d0e8403e8395822cdec9aa1acf7594647f253e103`

### :v7 ✅ (2026-02-27) — handler.py S3 botocore Config fix

- **Last built:** 2026-03-01 (image size: ~21GB, ~90 min build)
- **RunPod template:** `mrgxwb470f` → `:v11` ✅ — env vars included ✅
- **Local image status:** ✅ Removed (disk freed, 35GB build cache pruned)
- **Build command (RELIABLE — use this):**
  ```bash
  # Step 1: Build (plain docker build — has network access, no DNS issues)
  docker build \
    --file gaussian-splatting/Dockerfile \
    --tag hjeij2000/video-to-3dgs:serverless \
    .
  # Step 2: Push
  docker push hjeij2000/video-to-3dgs:serverless
  # Step 3: Create versioned tag server-side (increment vN)
  docker buildx imagetools create hjeij2000/video-to-3dgs:serverless --tag hjeij2000/video-to-3dgs:vN
  # Step 4: Clean up local image
  docker image rm hjeij2000/video-to-3dgs:serverless
  ```
  **NOTE:** `docker buildx build --push` and `docker buildx build --load` both fail intermittently
  due to DNS resolution failures inside the buildx container. Use plain `docker build` instead.

---

## Training Backends
| Trainer | Description | Undistortion | Key flags |
|---------|-------------|--------------|-----------|
| `brush` | Default. Lightweight Vulkan/wgpu trainer | ✅ Required | `--total-steps`, `--max-resolution` |
| `3dgut` | gsplat MCMC + Unscented Transform. Handles camera distortion natively | ❌ Skipped | `--with_ut`, `--with_eval3d` |

### 3DGUT details
- Trainer script: `Luminance-GS/gsplat/examples/simple_trainer.py mcmc`
- PLY output: `{result_dir}/ply/point_cloud_{iterations-1}.ply` — copied to `model/export_{iterations}.ply`
- Data layout expected by gsplat COLMAP parser: `{data_dir}/sparse/0/` + `{data_dir}/images/`
- For 3DGUT, `prepare_3dgut_data()` symlinks `distorted/sparse` and `input/` into `3dgut_data/`
- **gsplat local status:** v1.5.3 on main branch ✅, `datasets/colmap.py` patched for pycolmap 3.13+ ✅, all trainer deps installed in venv ✅
- **⚠️ Not yet tested end-to-end** — local GPU in power-error state (needs reboot); v12 Docker image not yet built

---

## SfM Backends
| Backend | Description | Unique arg | Default FPS |
|---------|-------------|------------|-------------|
| `mast3r` | Best quality, slowest. Deep feature matching + GLOMAP | `window_size` | 1 |
| `fastmap` | Fast SIFT + FastMap mapper. Good default | `match_overlap` | 2 |
| `colmap` | SIFT + COLMAP incremental mapper. More reliable on tricky scenes | `match_overlap` | 2 |
| `hloc` | SuperPoint + LightGlue + COLMAP. Best for low-texture/metallic | `match_overlap` | 2 |
| `pycusfm` | GPU-SIFT + GPU matching + pycolmap mapper (**placeholder**) | `match_overlap` | 2 |

**pycusfm note:** NVIDIA pyCuSFM (github.com/nvidia-isaac/pyCuSFM) requires initial camera poses for
monocular video — not yet suitable for this pipeline. Support for unposed sequential images is listed
as a future release. `pycusfm_sfm.py` is in place for when that support arrives.

---

## Telegram Bot
A 6-step wizard that lets non-technical users submit jobs and receive results.

### Flow
1. User sends a video file (≤50 MB), Google Drive share link, or direct URL
2. Wizard: **backend → fps → resolution → backend arg → iterations → trainer**
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
| `e6f200b` | Telegram Markdown parse error — underscores in arg names break formatting; escaped properly |
| `0f41248` | Silent crash on iterations step — added try/except + error handler |
| `0f3fa93` | Full 5-step wizard implemented (backend → fps → resolution → arg → iterations) |
| `6f8af43` | Show all timing steps in bot result message |
| `b221ea1` | Bot tasks dropped silently — switched to `ctx.application.create_task()` to keep tasks alive |
| `cbb88c0` | S3 presigned URL region fix in bot — added `botocore.Config(signature_version=s3v4, addressing_style=virtual)` |
| `f9e71b7` | PLY delivered as Telegram file attachment instead of bare link |

---

## Railway Deployment (Telegram Bot)
The Telegram bot runs as a Railway **worker** (not a web service — no HTTP port needed).

### Deploy steps
1. Connect Railway project to the `gaussian-splatting/telegram_bot/` directory
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
- **Template ID:** `mrgxwb470f` (name: `video-to-3dgs-v11`, currently on `:v11`)
- **Current Endpoint ID:** `uupefx2whvkg13` (name: `video-to-3dgs`)
- **GPUs:** ADA_24 + AMPERE_24
- **Workers:** 0 min / 2 max, FlashBoot OFF, idle timeout 10 min
- **⚠️ CRITICAL:** Always include all 4 AWS env vars when calling `saveTemplate` — passing `"env": []` wipes them and all jobs fail with `'AWS_S3_BUCKET'`
- **Template env vars set:**
  - `AWS_ACCESS_KEY_ID` = `<YOUR_AWS_ACCESS_KEY_ID>`
  - `AWS_SECRET_ACCESS_KEY` = `<YOUR_AWS_SECRET_ACCESS_KEY>`
  - `AWS_S3_BUCKET` = `splats-bucket`
  - `AWS_S3_REGION` = `me-south-1`

### Useful API calls
```bash
# Submit job (Brush trainer)
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"video_url": "...", "sfm_backend": "mast3r", "trainer": "brush", "fps": 1, "iterations": 7000, "max_resolution": 960}}'

# Submit job (3DGUT trainer — requires :v12 image)
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"video_url": "...", "sfm_backend": "hloc", "trainer": "3dgut", "fps": 2, "iterations": 7000}}'

# Check status
curl "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/{JOB_ID}" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"

# Download PLY from S3
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY='...' \
  aws s3 cp s3://splats-bucket/3dgs-outputs/{JOB_ID}/export_1000.ply ./export_1000.ply --region me-south-1
```

---

## Handler Features (handler.py)
- **SfM backend-specific args:**
  - `mast3r` → `window_size` (default 10)
  - `fastmap` / `colmap` / `hloc` / `pycusfm` → `match_overlap` (default 5)
- **Trainer field:** `trainer` = `brush` (default) or `3dgut`
- **Progress updates:** Live `runpod.serverless.progress_update()` calls on each pipeline step
- **Timing summary:** Pipeline timings parsed from stdout, included in job output JSON
- **timings.txt:** Written per job and uploaded to S3 (`3dgs-outputs/<job_id>/timings.txt`)
- **Error reporting:** On failure returns both stderr and last 1500 chars of stdout

---

## Session Log

### Session 12 (2026-04-05) — 3DGUT + pycusfm + v12 prep ⏳ IN PROGRESS
- **Added pycusfm SfM backend** (`pycusfm_sfm.py`) — placeholder; NVIDIA pyCuSFM doesn't support unposed monocular video yet (listed as future release). Code is in place.
- **Added 3DGUT training backend** (`--trainer 3dgut`):
  - `train_3dgut()` + `prepare_3dgut_data()` in `video_to_3dgs.py`
  - Skips `colmap image_undistorter` — gsplat handles distortion natively
  - Symlinks `distorted/sparse` + `input/` into `3dgut_data/` for gsplat COLMAP parser
  - PLY output copied to `model/export_{iterations}.ply` (same path as Brush)
- **Upgraded gsplat** in `Luminance-GS/gsplat` from v1.0.0 tag → main branch (v1.5.3)
- **Patched `datasets/colmap.py`** in gsplat — ported from removed `SceneManager` API to modern `pycolmap.Reconstruction` (3.13+). Tested: 100 images + 20,923 3D points parsed correctly.
- **Installed all 3DGUT trainer deps** in local venv (imageio, viser, nerfview, tyro, torchmetrics, tensorboard, fused-ssim fork, fused-bilagrid, splines)
- **Telegram bot upgraded** to 6-step wizard — added Step 6: trainer selection (Brush / 3DGUT)
- **Dockerfile updated** for v12: gsplat built in builder stage, all trainer deps added, pycusfm_sfm.py added to COPY, GSPLAT_DIR path-fix sed added
- **Blocker:** Local GPU (RTX 3070 Mobile) in power-error state — needs reboot to test locally

### Session 11 (2026-03-01) — Fix v9/v10 regressions → :v11 ✅ COMPLETE
- **Root cause found:** `RUST_BACKTRACE=1` caused ALL Brush jobs to SIGSEGV consistently.
- **Second bug:** `saveTemplate` with `"env": []` wiped AWS credentials. Fixed by always including env vars.
- **:v11 deployed:** RUST_BACKTRACE removed, Brush retry logic, env vars on template ✅

### Session 10 (2026-02-27) — SfM stats, splat count, Brush diagnostics ✅ COMPLETE
- Auto-retry on Brush -11 in bot.py; SfM stats + splat count in bot output
- :v9 deployed (later found to have RUST_BACKTRACE regression — fixed in v11)

### Session 9 (2026-02-27) — Bug hunt + :v8 build ✅ COMPLETE
- Container start failure on RTX 4090 nodes with driver <570 → downgraded to CUDA 12.4.1
- stderr pipe deadlock fix in handler.py
- torch seeds for reproducibility; presigned URL Markdown fix
- :v8 deployed ✅

### Session 8 (2026-02-27) — PLY delivery fix + stale worker resolved ✅ COMPLETE
- PLY sent as Telegram file attachment instead of presigned S3 link
- Stale worker (Brush -11): all workers confirmed updated to :v7

### Session 7 (2026-02-27) — Local hloc test ✅ COMPLETE
- Video: Drive `1OvptV4p43OGIpKTpT8dQFdopDF6-7ejO` (172MB, 58s)
- Settings: hloc, fps=2, 960px, match_overlap=10, 3000 iters
- Result: 20MB PLY, 86,214 points, total 162.5s

### Session 6 (2026-02-27) — Build + deploy :v7 ✅ COMPLETE
- handler.py S3 botocore Config fix; :v7 built (~90 min, 25.6GB), pushed, template updated

### Session 5 (2026-02-26/27) — Telegram Bot ✅ COMPLETE
- Full 6-step wizard; deployed on Railway; key bugs fixed

### Session 4 (2026-02-26) — LocalVSServerless benchmark testing ✅ COMPLETE
- fastmap ✅ 121.3MB PLY, 160.64s | hloc ✅ 80MB, 100.48s | mast3r ✅ 130MB, 387.34s

### Session 3 (2026-02-26) — IMG_2188 real-world test ✅ COMPLETE
- Test job `71b86608-...`: fastmap, fps=1, 1000 iters, 1280px → 5.5MB PLY, 141.32s total

### Session 2 (2026-02-25) — First successful RunPod job ✅ COMPLETE
- fastmap, fps=2, 1000 iters, Big Buck Bunny → 1.1MB PLY, 63.47s

### Session 1 — Initial pipeline + bug fixes ✅ COMPLETE
- ffmpeg CUDA decode crash, XDG_RUNTIME_DIR error, Brush SIGSEGV (Vulkan) — all fixed

---

## Bugs Fixed

### 1. ffmpeg CUDA decode crash ✅
- **Fix:** Removed hardware decode flags from ffmpeg command

### 2. XDG_RUNTIME_DIR not set ✅
- **Fix (`run_command()`):** `env.setdefault("XDG_RUNTIME_DIR", "/tmp")`

### 3. Brush SIGSEGV (exit code -11) ✅
- **Fix (Dockerfile):** `ENV NVIDIA_DRIVER_CAPABILITIES=all` + nvidia Vulkan ICD JSON

### 4. Pre-signed S3 URL region mismatch ✅
- **Fix:** `endpoint_url=f"https://s3.{region}.amazonaws.com"` in boto3 client

### 5. Brush OOM on high-res/long videos ⚠️ (workaround known)
- **Workaround:** fps=1 and max_resolution=1280 for long/4K videos

### 6. Stale worker image cache ✅
- **Fix:** Create new versioned Docker tag to force worker pull

### 7. GLOMAP missing libboost ✅ FIXED in :v4
### 8. docker commit overwrites ENTRYPOINT ✅ FIXED in :v5
### 9. GLOMAP missing libopenblas ✅ FIXED in :v6

### 10. buildx DNS failure inside builder container ⚠️
- **Workaround:** Use plain `docker build` then `docker push`

### 11. aiohttp brotli decode error (bot) ✅
- **Fix:** `Accept-Encoding: gzip, deflate` header on all aiohttp calls

### 12. Telegram Markdown parse error (bot) ✅
- **Fix:** Replace underscores with spaces in Markdown-embedded strings

### 13. Bot tasks silently dropped ✅
- **Fix:** Use `ctx.application.create_task()` instead of `asyncio.ensure_future()`

### 14. Container start failure on RTX 4090 nodes with old drivers ✅ FIXED in :v8
- **Fix:** Downgrade base from `cuda:12.8.1` → `cuda:12.4.1`

### 15. Handler stderr pipe deadlock (stuck worker) ✅ FIXED in :v8
- **Fix:** Redirect stderr to a temp file instead of `PIPE`

### 16. Result inconsistency across runs ⚠️ PARTIALLY FIXED in :v8
- **Fix:** `torch.manual_seed(42)` in hloc_sfm.py and mast3r_sfm.py

### 17. RUST_BACKTRACE=1 causes consistent Brush SIGSEGV ✅ FIXED in :v11
- **Fix:** Removed `RUST_BACKTRACE=1` from `run_command()`

### 18. saveTemplate wipes AWS env vars ✅ FIXED in :v11
- **Fix:** Always include all 4 AWS env vars in every `saveTemplate` call

### 19. gsplat COLMAP parser incompatible with modern pycolmap ✅ FIXED (Session 12)
- **Error:** `ImportError: cannot import name 'SceneManager' from 'pycolmap'`
- **Cause:** gsplat's `datasets/colmap.py` used the old pycolmap fork API (`SceneManager`, `cam.fx`, `cam.k1`, numpy `points3D` array) — removed in pycolmap 3.x
- **Fix:** Patched `Luminance-GS/gsplat/examples/datasets/colmap.py` to use `pycolmap.Reconstruction`, `cam.focal_length_x`, `cam.params[]`, and `Point3D.xyz` dict

---

## Known Issues / To Fix

### ⏳ Build and test :v12
- All code ready. Needs Docker build (~90 min) and end-to-end test on RunPod
- After reboot: test locally first with `--trainer 3dgut` on `IMG_2188 (1).MOV`

### ⏳ End-to-end 3DGUT test
- Local: `python video_to_3dgs.py --video "Downloads/IMG_2188 (1).MOV" --output /tmp/test_3dgut --sfm-backend hloc --trainer 3dgut --fps 2 --iterations 7000`
- RunPod: requires :v12 image

### ⏳ End-to-end :v11 test
- Verify Brush still works for both 3000 and 7000 steps post-refactor

### ⏳ Auto-scale fps/max_resolution (Bug #5)
- Auto-detect video resolution/length in handler, auto-reduce to fit VRAM

### PLY delivery for large files (>45 MB)
- Falls back to presigned S3 link — tricky for non-technical users

### Bot sessions lost on restart
- Sessions are in-memory — Railway restart loses all in-progress sessions

### pycusfm placeholder
- Waiting for NVIDIA to add unposed monocular support to pyCuSFM

---

## Benchmark Results (LocalVSServerless — IMG_2188.MOV, RTX 4090)
| Backend | Trainer | PLY Size | Total Time | SfM Time | Notes |
|---------|---------|----------|------------|----------|-------|
| fastmap | brush | 121.3 MB | 160.64s | ~60s | fps=1, 960px |
| hloc | brush | 80.0 MB | 100.48s | ~40s | fps=2, 960px |
| mast3r | brush | 130.0 MB | 387.34s | 308s | fps=1, 960px, 58 imgs, 54572 tracks |
| hloc | 3dgut | — | — | — | ⏳ pending :v12 test |

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
  1. `docker image rm hjeij2000/video-to-3dgs:serverless` — remove local image
  2. `docker builder prune -f` — clear buildx builder cache
- `docker system prune -a -f` — nuclear option, only if disk critically low
