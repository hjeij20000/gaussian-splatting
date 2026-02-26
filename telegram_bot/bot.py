#!/usr/bin/env python3
"""
Telegram bot for Video → 3D Gaussian Splatting pipeline.

Users send a video (file, Google Drive link, or direct URL) and choose
Fast (fastmap, ~2.5 min) or Best Quality (mast3r, ~6.5 min).
The bot submits to RunPod, polls for completion, and replies with
a download link for the .ply file + a timing summary.
"""

import asyncio
import logging
import os
import re
import uuid

import aiohttp
import boto3
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN    = os.environ["TELEGRAM_BOT_TOKEN"]
RUNPOD_API_KEY    = os.environ["RUNPOD_API_KEY"]
RUNPOD_ENDPOINT   = os.environ["RUNPOD_ENDPOINT_ID"]
AWS_KEY_ID        = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET        = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_BUCKET        = os.environ["AWS_S3_BUCKET"]
AWS_REGION        = os.environ["AWS_S3_REGION"]

RUNPOD_BASE = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}"
HEADERS     = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    endpoint_url=f"https://s3.{AWS_REGION}.amazonaws.com",
    aws_access_key_id=AWS_KEY_ID,
    aws_secret_access_key=AWS_SECRET,
)

# user_id → video_url waiting for backend choice
pending: dict[int, str] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_gdrive_url(text: str) -> str | None:
    """Convert a Google Drive share link to a direct download URL."""
    patterns = [
        r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"docs\.google\.com/[^/]+/d/([a-zA-Z0-9_-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            fid = m.group(1)
            return f"https://drive.google.com/uc?export=download&id={fid}&confirm=t"
    return None


async def upload_to_s3(data: bytes, filename: str) -> str:
    """Upload raw bytes to S3 and return a 24-hour pre-signed URL."""
    key = f"telegram-inputs/{uuid.uuid4()}/{filename}"
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=data,
        ContentType="video/mp4",
    )
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": AWS_BUCKET, "Key": key},
        ExpiresIn=86400,
    )


async def runpod_submit(video_url: str, backend: str) -> str:
    fps     = 1 if backend == "mast3r" else 2
    max_res = 960
    payload = {
        "input": {
            "video_url":   video_url,
            "sfm_backend": backend,
            "fps":         fps,
            "iterations":  7000,
            "max_resolution": max_res,
        }
    }
    async with aiohttp.ClientSession(headers={"Accept-Encoding": "gzip, deflate"}) as s:
        async with s.post(f"{RUNPOD_BASE}/run", json=payload, headers=HEADERS) as r:
            data = await r.json()
    return data["id"]


async def runpod_poll(job_id: str) -> dict:
    """Poll every 20 s until COMPLETED / FAILED / CANCELLED."""
    async with aiohttp.ClientSession(headers={"Accept-Encoding": "gzip, deflate"}) as s:
        while True:
            async with s.get(f"{RUNPOD_BASE}/status/{job_id}", headers=HEADERS) as r:
                data = await r.json()
            if data.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
                return data
            await asyncio.sleep(20)


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


# ── Handlers ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Video → 3D Gaussian Splatting*\n\n"
        "Send me a video and I'll turn it into a 3D model you can view and download!\n\n"
        "*I accept:*\n"
        "📹  A video file (up to 50 MB)\n"
        "🔗  A Google Drive share link\n"
        "🌐  Any direct video URL\n\n"
        "Once ready you'll get a `.ply` download link you can open in "
        "[SuperSplat](https://supersplat.xyz) or any 3DGS viewer.",
        parse_mode="Markdown",
        disable_web_page_preview=True,
    )


async def ask_backend(update: Update, video_url: str):
    """Store the URL and ask the user which backend they want."""
    pending[update.effective_user.id] = video_url
    kb = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🚀  Fast  (~2.5 min)",        callback_data="backend:fastmap"),
            InlineKeyboardButton("🎯  Best Quality  (~6.5 min)", callback_data="backend:mast3r"),
        ]
    ])
    await update.message.reply_text(
        "✅ Got your video!\n\n"
        "Which mode do you want?\n\n"
        "• *Fast* — great quality, done in ~2.5 minutes\n"
        "• *Best Quality* — deep AI matching, ~6.5 minutes",
        parse_mode="Markdown",
        reply_markup=kb,
    )


async def handle_video(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle direct video / document upload."""
    media = update.message.video or update.message.document
    if not media:
        await update.message.reply_text("❌ Couldn't read that file. Please try again.")
        return

    size_mb = (media.file_size or 0) / 1_048_576
    if size_mb > 50:
        await update.message.reply_text(
            f"⚠️ That file is {size_mb:.0f} MB — Telegram bots are limited to 50 MB downloads.\n\n"
            "Please upload the video to Google Drive and send me the share link instead."
        )
        return

    msg = await update.message.reply_text("⬇️ Downloading your video…")
    tg_file = await media.get_file()
    data    = bytes(await tg_file.download_as_bytearray())

    await msg.edit_text("☁️ Uploading to cloud storage…")
    name    = getattr(media, "file_name", None) or "video.mp4"
    url     = await upload_to_s3(data, name)

    await msg.delete()
    await ask_backend(update, url)


async def handle_text(update: Update, _: ContextTypes.DEFAULT_TYPE):
    """Handle Google Drive links and direct video URLs."""
    text = update.message.text.strip()

    url = extract_gdrive_url(text)
    if url:
        await ask_backend(update, url)
        return

    if text.lower().startswith(("http://", "https://")):
        await ask_backend(update, text)
        return

    await update.message.reply_text(
        "🤔 I didn't recognise that.\n\n"
        "Please send:\n"
        "• A video file\n"
        "• A Google Drive share link\n"
        "• A direct video URL\n\n"
        "Type /start to see the instructions again."
    )


async def handle_backend_choice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Inline-keyboard callback: backend chosen → submit job."""
    query = update.callback_query
    await query.answer()

    user_id   = update.effective_user.id
    video_url = pending.pop(user_id, None)
    if not video_url:
        await query.edit_message_text("❌ Session expired. Please send your video again.")
        return

    backend = query.data.split(":")[1]   # "fastmap" or "mast3r"
    label   = "Fast (fastmap)" if backend == "fastmap" else "Best Quality (mast3r)"

    await query.edit_message_text(
        f"⏳ Queuing your job on a GPU… ({label})\n"
        "I'll update this message as it progresses."
    )

    # Run job in background so the handler returns immediately
    ctx.application.create_task(
        _run_job(query, video_url, backend, label),
        update=update,
    )


async def _run_job(query, video_url: str, backend: str, label: str):
    """Submit → poll → reply. Runs as a background task."""
    try:
        job_id = await runpod_submit(video_url, backend)
        await query.edit_message_text(
            f"🔄 Processing on GPU… ({label})\n"
            f"Job ID: `{job_id}`\n\n"
            "_This usually takes 2–7 minutes. Hang tight!_",
            parse_mode="Markdown",
        )

        result = await runpod_poll(job_id)

        if result["status"] == "COMPLETED":
            out      = result.get("output", {})
            timings  = out.get("timings", {})
            ply_url  = out.get("ply_url", "")
            ply_mb   = out.get("ply_size_mb", 0)
            total    = timings.get("total", 0)
            frames   = timings.get("frame_extraction", 0)
            feat_ext = timings.get("feature_extraction", 0)
            feat_mat = timings.get("feature_matching", 0)
            sfm      = timings.get("sfm_reconstruction", 0)
            undist   = timings.get("undistortion", 0)
            train    = timings.get("training", 0)

            def row(label, secs):
                return f"  {label}: {fmt_time(secs)}\n" if secs > 0 else ""

            text = (
                f"🎉 *Your 3D model is ready!*\n\n"
                f"⏱ *Timings:*\n"
                + row("Frame extraction", frames)
                + row("Feature extraction", feat_ext)
                + row("Feature matching", feat_mat)
                + row("SfM reconstruction", sfm)
                + row("Undistortion", undist)
                + row("3DGS training", train)
                + f"  *Total: {fmt_time(total)}*\n\n"
                f"📦 *File size:* {ply_mb:.1f} MB\n\n"
                f"⬇️ *Download your `.ply`:*\n"
                f"{ply_url}\n\n"
                f"_Link valid for 24 hours._\n"
                f"_Open at_ [supersplat.xyz](https://supersplat.xyz) _→ drag & drop the file_"
            )
            await query.edit_message_text(
                text,
                parse_mode="Markdown",
                disable_web_page_preview=True,
            )
        else:
            err = str(result.get("error", "Unknown error"))
            # Show last 400 chars of error (usually the most relevant part)
            snippet = err[-400:].strip() if len(err) > 400 else err.strip()
            await query.edit_message_text(
                f"❌ *Processing failed.*\n\n```\n{snippet}\n```",
                parse_mode="Markdown",
            )

    except Exception as exc:
        logger.exception("Unhandled error in _run_job")
        await query.edit_message_text(f"❌ An unexpected error occurred:\n{exc}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_start))

    # Video files sent as compressed video or as document
    app.add_handler(MessageHandler(
        filters.VIDEO | filters.Document.MimeType("video/mp4")
            | filters.Document.MimeType("video/quicktime")
            | filters.Document.MimeType("video/x-msvideo")
            | filters.Document.MimeType("video/x-matroska"),
        handle_video,
    ))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_backend_choice, pattern=r"^backend:"))

    logger.info("Bot starting…")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
