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
- **Latest deployed tag:** `:v14` ✅ (2026-04-12)
- **Image size:** ~8.5 GB (estimated)

### Build command (RELIABLE — use this from /home/ibrahim):
```bash
docker build -f gaussian-splatting/Dockerfile -t hjeij2000/video-to-3dgs:v13 .
docker tag hjeij2000/video-to-3dgs:v13 hjeij2000/video-to-3dgs:serverless
docker push hjeij2000/video-to-3dgs:v13
docker push hjeij2000/video-to-3dgs:serverless
```
**NOTE:** `docker buildx build --push` fails intermittently (DNS inside buildx container). Use plain `docker build` + `docker push`.

### :v14 ✅ (2026-04-12)
- Added 2DGS (`2dgs`) as third training backend (hbb1/2d-gaussian-splatting)
- `diff-surfel-rasterization` CUDA extension built in Dockerfile builder stage
- Bug: `colormap()` crash at eval step — fixed with `--test_iterations 999999` (matplotlib 3.8+ removed `tostring_rgb`)
- Telegram bot: 2DGS added as third trainer button
- Local test: hloc + 2dgs, IMG_2188(1).MOV, RTX 3070 Mobile → 258 MB PLY, 43.6m total (training 41.7m)

### :v13 ✅ (2026-04-06)
- All :v12 changes included
- 3DGUT trainer fixes: `--disable-viewer`, `--disable_video`, `--eval-steps 999999`
- Symlink fix in `prepare_3dgut_data()` (`is_symlink()` instead of `exists()`)
- MASt3R mapper switched from `glomap` → `pycolmap` locally (GLOMAP binary crashes locally; Docker uses GLOMAP which still works on RunPod)
- Telegram bot: colmap added as 4th SfM backend option in UI (2×2 button grid)

### :v12 ✅ (2026-04-05)
- 3DGUT training backend (gsplat MCMC + `--with_ut --with_eval3d`)
- gsplat v1.5.3 built in builder stage (`TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"`)
- fused-ssim (rahul-goel fork) + fused-bilagrid compiled in builder stage
- nerfview installed in runtime stage (pure Python)
- boto3 S3 client timeout fix: `connect_timeout=30, read_timeout=300, retries={"max_attempts": 3}`
- Telegram bot: 6-step wizard (trainer selection as Step 6), pycusfm removed from UI

### :v11 ✅ (2026-03-01)
- Removed `RUST_BACKTRACE=1` — caused all Brush jobs to SIGSEGV (-11)
- Brush retry: up to 3 attempts on -11
- Always include AWS env vars in saveTemplate calls

### :v10 ⚠️ (2026-02-28) — broken
### :v9 ⚠️ (2026-02-28) — regressed
### :v8 ✅ (2026-02-27)
- CUDA 12.4.1 base; torch 2.6.0+cu124; stderr deadlock fix; torch seeds

---

## Training Backends
| Trainer | Description | Undistortion | Key flags |
|---------|-------------|--------------|-----------|
| `brush` | Default. Lightweight Vulkan/wgpu trainer | ✅ Required | `--total-steps`, `--max-resolution` |
| `3dgut` | gsplat MCMC + Unscented Transform. Handles camera distortion natively | ❌ Skipped | `--with_ut`, `--with_eval3d`, `--disable-viewer`, `--disable_video`, `--eval-steps 999999` |
| `2dgs` | 2D Gaussian Splatting (hbb1/2d-gaussian-splatting). Better surface geometry | ✅ Required | `--lambda_dist 1000`, `--lambda_normal 0.05`, `--test_iterations 999999` |

### 3DGUT details
- Trainer script: `Luminance-GS/gsplat/examples/simple_trainer.py mcmc`
- PLY output: `{result_dir}/ply/point_cloud_{iterations-1}.ply` → copied to `model/export_{iterations}.ply`
- Data layout: `{data_dir}/sparse/0/` + `{data_dir}/images/`
- `prepare_3dgut_data()` symlinks `distorted/sparse` → `sparse` and `input/` → `images` inside `3dgut_data/`
- gsplat local status: v1.5.3 ✅, `datasets/colmap.py` patched for pycolmap 3.13+ ✅
- **⚠️ `--disable-viewer` is critical** — without it, trainer sleeps for 11 days after training (`time.sleep(1000000)` in simple_trainer.py:1189)
- **⚠️ `--eval-steps 999999` is critical** — default eval_steps=[7000,30000] triggers trajectory rendering + DataLoader that hangs
- **⚠️ RunPod end-to-end test still pending** — needs :v13 cold start

---

## SfM Backends
| Backend | Description | Unique arg | Default FPS |
|---------|-------------|------------|-------------|
| `fastmap` | Fast SIFT + FastMap mapper. Good default | `match_overlap` | 2 |
| `colmap` | SIFT + COLMAP incremental mapper. More reliable on tricky scenes | `match_overlap` | 2 |
| `hloc` | SuperPoint + LightGlue + COLMAP. Best for low-texture/metallic | `match_overlap` | 2 |
| `mast3r` | Best quality, slowest. Deep feature matching + pycolmap (local) / GLOMAP (RunPod) | `window_size` | 1 |
| `pycusfm` | GPU-SIFT + GPU matching + pycolmap mapper (**placeholder**) | `match_overlap` | 2 |

**pycusfm note:** NVIDIA pyCuSFM requires initial camera poses — not suitable for unposed monocular video yet. `pycusfm_sfm.py` is in place for when NVIDIA adds that support. **Removed from bot UI** (still in handler.py as a valid backend if called via API).

**mast3r local note:** GLOMAP binary crashes locally (`colmap::Database::Open` SIGABRT — version mismatch with kapture-generated db). Switched to `--mapper pycolmap` in `video_to_3dgs.py` for local runs. Docker image still uses GLOMAP which works fine on RunPod.

---

## Telegram Bot
A 6-step wizard that lets non-technical users submit jobs and receive results.

### Flow
1. User sends a video file (≤50 MB), Google Drive share link, or direct URL
2. Wizard: **backend → fps → resolution → backend arg → iterations → trainer**
3. Bot uploads video to S3 (`telegram-inputs/<uuid>/<filename>`) and generates presigned URL
4. Submits job to RunPod, polls every 20s, edits status message live
5. On completion: sends timings summary, then delivers `.ply` as Telegram file (≤45 MB); falls back to presigned S3 link for larger files

### Bot backends shown (4 options — 2×2 grid)
- ⚡ fastmap, 🗺️ colmap, 🔍 hloc, 🎯 mast3r

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
AWS_S3_BUCKET=frankfurt-splats-bucket
AWS_S3_REGION=eu-central-1
```

### GitHub repo (Railway source)
- Remote: `myfork` → `https://github.com/hjeij20000/gaussian-splatting.git`
- Railway auto-deploys on push to `main`
- To push: need GitHub PAT with `repo` scope; set on remote URL temporarily then clear

---

## RunPod Setup
- **API domain:** `https://api.runpod.ai` *(NOT `.io` — returns 404)*
- **Endpoint ID:** `uupefx2whvkg13` (name: `video-to-3dgs`)
- **Template ID:** `mrgxwb470f` (needs update to `:v13` via `:serverless` tag)
- **GPUs:** ADA_24 + AMPERE_24
- **Workers:** 0 min / 2 max, FlashBoot OFF, idle timeout 10 min
- **⚠️ CRITICAL:** Always include all 4 AWS env vars when calling `saveTemplate` — `"env": []` wipes them
- **⚠️ To force cold start (pick up new image):** Set Max Workers to 0 → Save → wait 10s → set back → Save

### Template env vars
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_BUCKET=frankfurt-splats-bucket`, `AWS_S3_REGION=eu-central-1`

### Useful API calls
```bash
# Submit job (Brush)
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"video_url": "...", "sfm_backend": "hloc", "trainer": "brush", "fps": 2, "iterations": 7000}}'

# Submit job (3DGUT — requires :v13)
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

### Session 14 (2026-04-05/06) — Local full test + 3DGUT fixes + :v13
- **Local benchmark** — all 4 Brush SfM backends tested end-to-end on IMG_2188(1).MOV:
  - RTX 3070 Mobile back online (was in power-error state)
  - Fixed 12 missing Python packages in `gs` conda env
  - Fixed `np.fromstring` → `np.frombuffer` in fastmap (numpy 2.x breaking change)
  - Fixed GLOMAP → pycolmap mapper for mast3r locally (GLOMAP binary incompatible with kapture db)
- **3DGUT local test** — discovered and fixed 3 hangs:
  - Bug #24: `--with_eval3d` triggers trajectory rendering → `writer.close()` hangs (imageio ffmpeg) → fixed with `--disable_video`
  - Bug #25: `--eval-steps` defaults to [7000, 30000] → triggers eval DataLoader hang → fixed with `--eval-steps 999999`
  - Bug #26: After training, viser viewer `time.sleep(1000000)` keeps process alive 11 days → fixed with `--disable-viewer`
  - Bug #27: Symlink in `prepare_3dgut_data()` uses `exists()` (follows symlinks, returns False for broken links) → fixed with `is_symlink()`
- **Telegram bot** — colmap added as 4th SfM backend (2×2 grid)
- **Laptop** — screen lock/sleep disabled while on charger
- **:v13 build + push** — in progress

### Session 13 (2026-04-05) — :v12 build, deploy, first RunPod test
- Fixed .dockerignore, built :v12, fixed boto3 S3 timeout (Bug #23)
- First RunPod test hung at S3 upload → fixed → second test hit warm worker
- 3DGUT end-to-end on RunPod still pending (needs :v13 cold start)

### Session 12 (2026-04-05) — 3DGUT + pycusfm + v12 prep
- Added 3DGUT trainer, pycusfm placeholder, upgraded gsplat v1.0→v1.5.3

### Session 11 (2026-03-01) — Fix v9/v10 regressions → :v11 ✅
### Session 10 (2026-02-27) — SfM stats, splat count, Brush diagnostics ✅
### Session 9 (2026-02-27) — Bug hunt + :v8 ✅
### Session 8 (2026-02-27) — PLY delivery fix ✅
### Session 7 (2026-02-27) — Local hloc test ✅
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
| 21 | gsplat CUDA build fails for sm_61 | ✅ | :v12 (override `TORCH_CUDA_ARCH_LIST`) |
| 22 | fused-ssim/bilagrid fail in runtime stage (no nvcc) | ✅ | :v12 (moved to builder stage) |
| 23 | S3 upload hangs indefinitely (no boto3 timeout) | ✅ | :v12 patch |
| 24 | 3DGUT trajectory rendering hangs (`writer.close()` ffmpeg) | ✅ | :v13 (`--disable_video`) |
| 25 | 3DGUT eval DataLoader hangs at step 6999 | ✅ | :v13 (`--eval-steps 999999`) |
| 26 | 3DGUT viser viewer sleeps 11 days after training | ✅ | :v13 (`--disable-viewer`) |
| 27 | 3DGUT symlink fails on retry (`exists()` vs `is_symlink()`) | ✅ | :v13 |
| 28 | 2DGS `colormap()` crashes at eval step — `tostring_rgb` removed in matplotlib 3.8+ | ✅ | :v14 (`--test_iterations 999999`) |

---

## Next Steps (priority order)

1. **Force cold start on RunPod** — Set Max Workers → 0 → Save → restore (picks up :v13)
2. **Test Brush regression on RunPod** — `{"sfm_backend": "hloc", "trainer": "brush", "fps": 2, "iterations": 7000}`
3. **Test 3DGUT on RunPod** — `{"sfm_backend": "hloc", "trainer": "3dgut", "fps": 2, "iterations": 7000}`
4. **Add 3DGUT benchmark row** to Benchmark Results once RunPod test passes
5. ~~**Update S3 bucket name**~~ ✅ — `frankfurt-splats-bucket` / `eu-central-1` updated in RunPod template, Railway, and local `.env`

---

## Benchmark Results (IMG_2188.MOV / IMG_2188(1).MOV)

### RTX 4090 (RunPod)
| Backend | Trainer | PLY Size | Total Time | SfM Time | Notes |
|---------|---------|----------|------------|----------|-------|
| fastmap | brush | 121.3 MB | 160.64s | ~60s | fps=1, 960px |
| hloc | brush | 80.0 MB | 100.48s | ~40s | fps=2, 960px |
| mast3r | brush | 130.0 MB | 387.34s | 308s | fps=1, 960px |
| hloc | 3dgut | — | — | — | ⏳ pending :v13 cold-start test |

### RTX 3070 Mobile (local, fps=2, 960px, 7000 iters)
| Backend | Trainer | PLY Size | Total Time | SfM Time | Train Time |
|---------|---------|----------|------------|----------|------------|
| colmap | brush | 112 MB | 3:74m | 68.6s | 101.8s |
| fastmap | brush | 161 MB | 3:75m | 72.7s | 114.6s |
| hloc | brush | 80 MB | 3:09m | 60.6s | 89.8s |
| hloc | 2dgs | 258 MB | 43:6m | 78.3s | 2503s |
| mast3r | brush | 250 MB | 2:89m* | —* | 144.0s |
| colmap | 3dgut | 185 MB | 13:06m | 70.1s | 664s |

*mast3r SfM ran separately (pycolmap mapper); timing excludes SfM

---

## Credentials
- Stored locally in `gaussian-splatting/.env` (gitignored — never commit)
- `.env` contains: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_BUCKET`, `AWS_S3_REGION`, `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`
- Load: `export $(grep -v '^#' .env | xargs)`
- Railway bot env vars set separately in Railway dashboard

---

## Storage Management
- Disk: 192G total, ~25G free (as of 2026-04-06 pre :v13 build)
- **Post-build cleanup:**
  1. `docker builder prune -f` — clear build cache
  2. `docker image rm hjeij2000/video-to-3dgs:v13 hjeij2000/video-to-3dgs:serverless` — remove local images
- `docker system prune -a -f` — nuclear option if disk critically low
