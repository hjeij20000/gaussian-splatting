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
| `gaussian-splatting/pycusfm_sfm.py` | pyCuSFM backend (placeholder — not yet functional) |
| `gaussian-splatting/telegram_bot/bot.py` | Telegram bot frontend |
| `gaussian-splatting/telegram_bot/Procfile` | Railway process definition (`worker: python bot.py`) |
| `gaussian-splatting/telegram_bot/requirements.txt` | Bot deps (python-telegram-bot, boto3, aiohttp) |
| `gaussian-splatting/telegram_bot/.env.example` | Env var template for bot |
| `Luminance-GS/gsplat/` | gsplat repo (main branch, v1.5.3, patched for modern pycolmap) |
| `/home/ibrahim/.dockerignore` | Build context filter — must include `!Luminance-GS/` |

---

## Docker Image (RunPod serverless)
- **Docker Hub:** `hjeij2000/video-to-3dgs:serverless`
- **Latest deployed tag:** `:v12` ✅ (2026-04-05)
- **Image size:** 8.34 GB

### Build command (RELIABLE — use this from /home/ibrahim):
```bash
docker build -f gaussian-splatting/Dockerfile -t hjeij2000/video-to-3dgs:v12 .
docker tag hjeij2000/video-to-3dgs:v12 hjeij2000/video-to-3dgs:serverless
docker push hjeij2000/video-to-3dgs:v12
docker push hjeij2000/video-to-3dgs:serverless
```
**NOTE:** `docker buildx build --push` fails intermittently (DNS inside buildx container). Use plain `docker build` + `docker push`.

### :v12 ✅ (2026-04-05)
- 3DGUT training backend (gsplat MCMC + `--with_ut --with_eval3d`)
- gsplat v1.5.3 built in builder stage (`TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"` — Pascal sm_61 excluded because `cooperative_groups::labeled_partition` requires sm_70+)
- fused-ssim (rahul-goel fork) + fused-bilagrid compiled in builder stage (need nvcc)
- nerfview installed in runtime stage (pure Python)
- New deps: imageio[ffmpeg], viser, tyro, scikit-learn, torchmetrics[image], tensorboard, tensorly, splines
- pycusfm_sfm.py added (placeholder)
- `trainer` input field in handler.py (`brush` default | `3dgut`)
- boto3 S3 client timeout fix: `connect_timeout=30, read_timeout=300, retries={"max_attempts": 3}`
- Telegram bot: 6-step wizard (trainer selection as Step 6), pycusfm removed from UI

### :v11 ✅ (2026-03-01)
- Removed `RUST_BACKTRACE=1` — caused all Brush jobs to SIGSEGV (-11)
- Brush retry: up to 3 attempts on -11
- Always include AWS env vars in saveTemplate calls

### :v10 ⚠️ (2026-02-28) — broken (RUST_BACKTRACE=1 present)
### :v9 ⚠️ (2026-02-28) — regressed (RUST_BACKTRACE=1 + template env vars wiped)
### :v8 ✅ (2026-02-27)
- CUDA 12.4.1 base; torch 2.6.0+cu124; stderr deadlock fix; torch seeds

---

## Training Backends
| Trainer | Description | Undistortion | Key flags |
|---------|-------------|--------------|-----------|
| `brush` | Default. Lightweight Vulkan/wgpu trainer | ✅ Required | `--total-steps`, `--max-resolution` |
| `3dgut` | gsplat MCMC + Unscented Transform. Handles camera distortion natively | ❌ Skipped | `--with_ut`, `--with_eval3d` |

### 3DGUT details
- Trainer script: `Luminance-GS/gsplat/examples/simple_trainer.py mcmc`
- PLY output: `{result_dir}/ply/point_cloud_{iterations-1}.ply` → copied to `model/export_{iterations}.ply`
- Data layout: `{data_dir}/sparse/0/` + `{data_dir}/images/`
- `prepare_3dgut_data()` symlinks `distorted/sparse` → `sparse` and `input/` → `images` inside `3dgut_data/`
- gsplat local status: v1.5.3 ✅, `datasets/colmap.py` patched for pycolmap 3.13+ ✅
- **⚠️ Not yet tested end-to-end on RunPod** — v12 image deployed but S3 upload hang occurred during first test; fix pushed in updated :v12/:serverless

---

## SfM Backends
| Backend | Description | Unique arg | Default FPS |
|---------|-------------|------------|-------------|
| `mast3r` | Best quality, slowest. Deep feature matching + GLOMAP | `window_size` | 1 |
| `fastmap` | Fast SIFT + FastMap mapper. Good default | `match_overlap` | 2 |
| `colmap` | SIFT + COLMAP incremental mapper. More reliable on tricky scenes | `match_overlap` | 2 |
| `hloc` | SuperPoint + LightGlue + COLMAP. Best for low-texture/metallic | `match_overlap` | 2 |
| `pycusfm` | GPU-SIFT + GPU matching + pycolmap mapper (**placeholder**) | `match_overlap` | 2 |

**pycusfm note:** NVIDIA pyCuSFM requires initial camera poses — not suitable for unposed monocular video yet. `pycusfm_sfm.py` is in place for when NVIDIA adds that support. **Removed from bot UI** (still in handler.py as a valid backend if called via API).

---

## Telegram Bot
A 6-step wizard that lets non-technical users submit jobs and receive results.

### Flow
1. User sends a video file (≤50 MB), Google Drive share link, or direct URL
2. Wizard: **backend → fps → resolution → backend arg → iterations → trainer**
3. Bot uploads video to S3 (`telegram-inputs/<uuid>/<filename>`) and generates presigned URL
4. Submits job to RunPod, polls every 20s, edits status message live
5. On completion: sends timings summary, then delivers `.ply` as Telegram file (≤45 MB); falls back to presigned S3 link for larger files

### Bot backends shown (3 options — pycusfm removed)
- ⚡ fastmap, 🔍 hloc, 🎯 mast3r

### Files
- **`telegram_bot/bot.py`** — full bot code (single file)
- **`telegram_bot/Procfile`** — `worker: python bot.py`
- **`telegram_bot/requirements.txt`** — `python-telegram-bot[webhooks]==21.9`, `boto3`, `aiohttp`

### Env vars required
```
TELEGRAM_BOT_TOKEN=...
RUNPOD_API_KEY=...
RUNPOD_ENDPOINT_ID=uupefx2whvkg13
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_BUCKET=splats-bucket
AWS_S3_REGION=me-south-1
```

### GitHub repo (Railway source)
- Remote: `myfork` → `https://github.com/hjeij20000/gaussian-splatting.git`
- Railway auto-deploys on push to `main`
- To push: need GitHub PAT with `repo` scope; set on remote URL temporarily then clear

---

## RunPod Setup
- **API domain:** `https://api.runpod.ai` *(NOT `.io` — returns 404)*
- **Endpoint ID:** `uupefx2whvkg13` (name: `video-to-3dgs`)
- **Template ID:** `mrgxwb470f` (now on `:v12` via `:serverless` tag)
- **GPUs:** ADA_24 + AMPERE_24
- **Workers:** 0 min / 2 max, FlashBoot OFF, idle timeout 10 min
- **⚠️ CRITICAL:** Always include all 4 AWS env vars when calling `saveTemplate` — `"env": []` wipes them
- **⚠️ To force cold start (pick up new image):** Set Max Workers to 0 → Save → wait 10s → set back → Save

### Template env vars
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_BUCKET=splats-bucket`, `AWS_S3_REGION=me-south-1`

### Useful API calls
```bash
# Submit job (Brush)
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"video_url": "...", "sfm_backend": "hloc", "trainer": "brush", "fps": 2, "iterations": 7000}}'

# Submit job (3DGUT — requires :v12)
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"video_url": "...", "sfm_backend": "hloc", "trainer": "3dgut", "fps": 2, "iterations": 7000}}'

# Check status
curl "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/{JOB_ID}" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

---

## Handler Features (handler.py)
- **SfM backend-specific args:** `mast3r` → `window_size` (default 10); others → `match_overlap` (default 5)
- **Trainer field:** `trainer` = `brush` (default) or `3dgut`
- **Progress updates:** Live `runpod.serverless.progress_update()` calls on each pipeline step
- **Timing summary:** Parsed from stdout, included in job output JSON
- **timings.txt:** Uploaded to S3 (`3dgs-outputs/<job_id>/timings.txt`)
- **Error reporting:** Returns both stderr and last 1500 chars of stdout on failure
- **boto3 S3 client:** `connect_timeout=30, read_timeout=300, retries=max_attempts=3`

---

## Session Log

### Session 13 (2026-04-05) — :v12 build, deploy, first RunPod test ⏳ IN PROGRESS
- **Fixed .dockerignore** — was missing `!Luminance-GS/`; added it + exclusions for `.git/` subdirs
- **Built Docker :v12** — 4 attempts due to new errors:
  - Bug #20: gsplat `pip install -e .` failed with `No module named 'torch'` → fixed with `--no-build-isolation`
  - Bug #21: `cooperative_groups::labeled_partition` unavailable on sm_61 (Pascal) → fixed by setting `TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"` for the gsplat build step only
  - Bug #22: fused-ssim + fused-bilagrid are CUDA extensions — failed in runtime stage (no nvcc) → moved to builder stage
  - Final build succeeded: 8.34 GB image ✅
- **Pushed** `:v12` and `:serverless` to Docker Hub ✅
- **Pushed 48 commits to GitHub** (`myfork` remote) — Railway redeployed bot ✅
- **Bot updated:** 3DGUT trainer option visible in Step 6, pycusfm removed from UI
- **First RunPod test:** Job ran (pipeline completed in ~3.5 min on RTX 4090), then **hung at S3 upload** — boto3 had no timeout, connection stalled indefinitely
- **Bug #23 fixed:** Added `connect_timeout=30, read_timeout=300, retries=3` to boto3 client in handler.py
- **Rebuilt + re-pushed** `:v12` and `:serverless` with boto3 fix ✅
- **Second test attempt:** Warm worker still running old image without fix → job stuck again
- **Resolution:** User must force cold start (set RunPod workers to 0 → save → restore)
- **3DGUT end-to-end test:** Still pending (needs cold start + retry)
- **Local GPU:** RTX 3070 Mobile still in power-error state — all tests must be on RunPod

### Session 12 (2026-04-05) — 3DGUT + pycusfm + v12 prep
- Added pycusfm SfM backend (placeholder — NVIDIA pyCuSFM doesn't support unposed video yet)
- Added 3DGUT training backend: `train_3dgut()` + `prepare_3dgut_data()` in `video_to_3dgs.py`
- Upgraded gsplat from v1.0.0 → main branch (v1.5.3) in `Luminance-GS/gsplat`
- Patched `datasets/colmap.py` for modern pycolmap 3.13+ API (`pycolmap.Reconstruction`)
- Telegram bot upgraded to 6-step wizard (trainer selection added)
- Dockerfile updated for v12 (gsplat builder stage, 3DGUT deps)
- Blocker: local GPU in power-error state

### Session 11 (2026-03-01) — Fix v9/v10 regressions → :v11 ✅
- Root cause: `RUST_BACKTRACE=1` caused ALL Brush jobs to SIGSEGV
- `saveTemplate` with `"env": []` wiped AWS credentials — fixed
- :v11 deployed ✅

### Session 10 (2026-02-27) — SfM stats, splat count, Brush diagnostics ✅
- Auto-retry on Brush -11 in bot.py; SfM stats + splat count in bot output

### Session 9 (2026-02-27) — Bug hunt + :v8 ✅
- CUDA 12.4.1 base; stderr deadlock fix; torch seeds; :v8 deployed ✅

### Session 8 (2026-02-27) — PLY delivery fix ✅
- PLY sent as Telegram file attachment

### Session 7 (2026-02-27) — Local hloc test ✅
- Video: Drive `1OvptV4p43OGIpKTpT8dQFdopDF6-7ejO` (172MB, 58s)
- hloc, fps=2, 960px, match_overlap=10, 3000 iters → 20MB PLY, 86,214 points, 162.5s

### Session 6 (2026-02-27) — :v7 build + deploy ✅
### Session 5 (2026-02-26/27) — Telegram Bot ✅
### Session 4 (2026-02-26) — LocalVSServerless benchmark ✅
### Session 3 (2026-02-26) — IMG_2188 real-world test ✅
### Session 2 (2026-02-25) — First successful RunPod job ✅
### Session 1 — Initial pipeline + bug fixes ✅

---

## Bugs Fixed

| # | Description | Status | Fixed in |
|---|-------------|--------|----------|
| 1 | ffmpeg CUDA decode crash | ✅ | Session 1 |
| 2 | XDG_RUNTIME_DIR not set | ✅ | Session 1 |
| 3 | Brush SIGSEGV (-11) on cold start | ✅ | Session 1 |
| 4 | Pre-signed S3 URL region mismatch | ✅ | :v7 |
| 5 | Brush OOM on high-res/long videos | ⚠️ workaround | use fps=1, 1280px |
| 6 | Stale worker image cache | ✅ | versioned tags |
| 7 | GLOMAP missing libboost | ✅ | :v4 |
| 8 | docker commit overwrites ENTRYPOINT | ✅ | :v5 |
| 9 | GLOMAP missing libopenblas | ✅ | :v6 |
| 10 | buildx DNS failure inside builder | ⚠️ workaround | use plain `docker build` |
| 11 | aiohttp brotli decode error (bot) | ✅ | bot fix |
| 12 | Telegram Markdown parse error (bot) | ✅ | bot fix |
| 13 | Bot tasks silently dropped | ✅ | `create_task()` |
| 14 | Container start failure on RTX 4090 (old driver) | ✅ | :v8 |
| 15 | Handler stderr pipe deadlock | ✅ | :v8 |
| 16 | Result inconsistency across runs | ⚠️ partial | torch.manual_seed |
| 17 | RUST_BACKTRACE=1 → consistent Brush SIGSEGV | ✅ | :v11 |
| 18 | saveTemplate wipes AWS env vars | ✅ | :v11 |
| 19 | gsplat COLMAP parser incompatible with pycolmap 3.x | ✅ | Session 12 |
| 20 | gsplat pip install fails — `No module named 'torch'` | ✅ | :v12 (`--no-build-isolation`) |
| 21 | gsplat CUDA build fails for sm_61 (`labeled_partition` requires sm_70+) | ✅ | :v12 (override `TORCH_CUDA_ARCH_LIST` for gsplat step) |
| 22 | fused-ssim/bilagrid fail in runtime stage (no nvcc) | ✅ | :v12 (moved to builder stage) |
| 23 | S3 upload hangs indefinitely (no boto3 timeout) | ✅ | :v12 patch (`connect_timeout=30, read_timeout=300`) |

---

## Next Steps (priority order)

1. **Force cold start on RunPod** — Set Max Workers → 0 → Save → restore → Submit test job
2. **Test Brush regression** — `{"sfm_backend": "hloc", "trainer": "brush", "fps": 2, "iterations": 7000}`
3. **Test 3DGUT** — `{"sfm_backend": "hloc", "trainer": "3dgut", "fps": 2, "iterations": 7000}`
4. **If 3DGUT fails** — check RunPod logs for the actual error; share logs.txt
5. **Add 3DGUT benchmark row** to Benchmark Results table once tested
6. **Fix local GPU** — RTX 3070 Mobile in power-error state; needs hard power cycle

---

## Benchmark Results (IMG_2188.MOV, RTX 4090, RunPod)
| Backend | Trainer | PLY Size | Total Time | SfM Time | Notes |
|---------|---------|----------|------------|----------|-------|
| fastmap | brush | 121.3 MB | 160.64s | ~60s | fps=1, 960px |
| hloc | brush | 80.0 MB | 100.48s | ~40s | fps=2, 960px |
| mast3r | brush | 130.0 MB | 387.34s | 308s | fps=1, 960px |
| hloc | 3dgut | — | — | — | ⏳ pending cold-start test |

---

## Credentials
- Stored locally in `gaussian-splatting/.env` (gitignored — never commit)
- `.env` contains: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_BUCKET`, `AWS_S3_REGION`, `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`
- Load: `export $(grep -v '^#' .env | xargs)`
- Railway bot env vars set separately in Railway dashboard

---

## Storage Management
- Disk: 192G total, ~23G free (as of 2026-04-05 post build-cache prune)
- **Post-build cleanup:**
  1. `docker builder prune -f` — clear build cache (~34GB freed)
  2. `docker image rm hjeij2000/video-to-3dgs:v12 hjeij2000/video-to-3dgs:serverless` — remove local images
- `docker system prune -a -f` — nuclear option if disk critically low
