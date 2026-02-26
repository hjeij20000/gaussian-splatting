#!/usr/bin/env python3
"""
Telegram bot for Video → 3D Gaussian Splatting pipeline.

Wizard flow:
  video/link → backend → fps → resolution → backend arg → iterations → submit
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
TELEGRAM_TOKEN  = os.environ["TELEGRAM_BOT_TOKEN"]
RUNPOD_API_KEY  = os.environ["RUNPOD_API_KEY"]
RUNPOD_ENDPOINT = os.environ["RUNPOD_ENDPOINT_ID"]
AWS_KEY_ID      = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET      = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_BUCKET      = os.environ["AWS_S3_BUCKET"]
AWS_REGION      = os.environ["AWS_S3_REGION"]

RUNPOD_BASE = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}"
HEADERS     = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
NO_BR       = {"Accept-Encoding": "gzip, deflate"}

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    endpoint_url=f"https://s3.{AWS_REGION}.amazonaws.com",
    aws_access_key_id=AWS_KEY_ID,
    aws_secret_access_key=AWS_SECRET,
)

# user_id → session dict
sessions: dict[int, dict] = {}

BACKEND_INFO = {
    "fastmap": {"label": "⚡ fastmap",  "arg_name": "match_overlap", "default_fps": 2},
    "hloc":    {"label": "🔍 hloc",     "arg_name": "match_overlap", "default_fps": 2},
    "mast3r":  {"label": "🎯 mast3r",   "arg_name": "window_size",   "default_fps": 1},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def kb(buttons: list[list[tuple[str, str]]]) -> InlineKeyboardMarkup:
    """Build InlineKeyboardMarkup from list of rows of (label, callback_data)."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(label, callback_data=data) for label, data in row]
        for row in buttons
    ])


def extract_gdrive_url(text: str) -> str | None:
    for pat in [
        r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"docs\.google\.com/[^/]+/d/([a-zA-Z0-9_-]+)",
    ]:
        m = re.search(pat, text)
        if m:
            return f"https://drive.google.com/uc?export=download&id={m.group(1)}&confirm=t"
    return None


async def upload_to_s3(data: bytes, filename: str) -> str:
    key = f"telegram-inputs/{uuid.uuid4()}/{filename}"
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=data, ContentType="video/mp4")
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": AWS_BUCKET, "Key": key},
        ExpiresIn=86400,
    )


async def runpod_submit(sess: dict) -> str:
    payload = {
        "input": {
            "video_url":      sess["video_url"],
            "sfm_backend":    sess["backend"],
            "fps":            sess["fps"],
            "iterations":     sess["iterations"],
            "max_resolution": sess["resolution"],
            sess["arg_name"]: sess["arg_value"],
        }
    }
    async with aiohttp.ClientSession(headers=NO_BR) as s:
        async with s.post(f"{RUNPOD_BASE}/run", json=payload, headers=HEADERS) as r:
            data = await r.json()
    return data["id"]


async def runpod_poll(job_id: str) -> dict:
    async with aiohttp.ClientSession(headers=NO_BR) as s:
        while True:
            async with s.get(f"{RUNPOD_BASE}/status/{job_id}", headers=HEADERS) as r:
                data = await r.json()
            if data.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
                return data
            await asyncio.sleep(20)


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def session_summary(sess: dict) -> str:
    backend = sess.get("backend", "?")
    label   = BACKEND_INFO.get(backend, {}).get("label", backend)
    arg     = sess.get("arg_name", "").replace("_", " ")   # underscores break Markdown
    val     = sess.get("arg_value", "")
    arg_str = f"  {arg}: {val}\n" if arg else ""
    return (
        f"*Settings so far:*\n"
        f"  Backend: {label}\n"
        f"  FPS: {sess.get('fps', '?')}\n"
        f"  Resolution: {sess.get('resolution', '?')}px\n"
        f"{arg_str}"
        f"  Iterations: {sess.get('iterations', '?')}\n"
    )


# ── Wizard steps ──────────────────────────────────────────────────────────────

async def ask_backend(query_or_msg, user_id: int):
    sessions[user_id]["step"] = "backend"
    text = (
        "🎬 *Step 1 / 5 — Reconstruction method*\n\n"
        "• *fastmap* — Fast SIFT matching. Great all-rounder (~2.5 min)\n"
        "• *hloc* — SuperPoint + LightGlue. Best for shiny/low-texture scenes (~3 min)\n"
        "• *mast3r* — AI deep matching. Highest quality, slowest (~6.5 min)"
    )
    markup = kb([
        [("⚡ fastmap", "w:backend:fastmap"),
         ("🔍 hloc",    "w:backend:hloc"),
         ("🎯 mast3r",  "w:backend:mast3r")],
    ])
    if hasattr(query_or_msg, "edit_message_text"):
        await query_or_msg.edit_message_text(text, parse_mode="Markdown", reply_markup=markup)
    else:
        await query_or_msg.reply_text(text, parse_mode="Markdown", reply_markup=markup)


async def ask_fps(query, sess: dict):
    sess["step"] = "fps"
    default = BACKEND_INFO[sess["backend"]]["default_fps"]
    text = (
        f"📸 *Step 2 / 5 — Frames per second*\n\n"
        f"How many frames to extract per second of video.\n"
        f"More frames = better quality but slower.\n"
        f"_(Recommended: {default} fps for {sess['backend']})_"
    )
    markup = kb([[
        ("1 fps",  "w:fps:1"),
        ("2 fps",  "w:fps:2"),
        ("3 fps",  "w:fps:3"),
    ]])
    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=markup)


async def ask_resolution(query, sess: dict):
    sess["step"] = "resolution"
    text = (
        "🖼 *Step 3 / 5 — Max resolution*\n\n"
        "Maximum image size used during reconstruction and training.\n"
        "Higher = more detail but uses more GPU memory.\n\n"
        "• *640px* — Fast, lower detail\n"
        "• *960px* — Balanced _(recommended)_\n"
        "• *1280px* — High detail, more memory"
    )
    markup = kb([[
        ("640px",  "w:resolution:640"),
        ("960px",  "w:resolution:960"),
        ("1280px", "w:resolution:1280"),
    ]])
    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=markup)


async def ask_backend_arg(query, sess: dict):
    sess["step"] = "arg"
    backend  = sess["backend"]
    arg_name = BACKEND_INFO[backend]["arg_name"]
    sess["arg_name"] = arg_name

    if arg_name == "match_overlap":
        text = (
            "🔗 *Step 4 / 5 — Matching coverage*\n\n"
            "How many nearby frames each frame is matched against.\n"
            "Higher = more connections between frames, better for complex scenes.\n\n"
            "• *3* — Minimal (fast)\n"
            "• *5* — Standard _(recommended)_\n"
            "• *10* — Thorough (slower)"
        )
        markup = kb([[
            ("3 — Minimal",   "w:arg:3"),
            ("5 — Standard",  "w:arg:5"),
            ("10 — Thorough", "w:arg:10"),
        ]])
    else:  # window_size for mast3r
        text = (
            "🪟 *Step 4 / 5 — Matching window*\n\n"
            "Number of neighboring frames used in MASt3R's deep matching.\n"
            "Larger window = better accuracy on hard scenes.\n\n"
            "• *5* — Fast\n"
            "• *10* — Standard _(recommended)_\n"
            "• *20* — Thorough (slower)"
        )
        markup = kb([[
            ("5 — Fast",      "w:arg:5"),
            ("10 — Standard", "w:arg:10"),
            ("20 — Thorough", "w:arg:20"),
        ]])
    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=markup)


async def ask_iterations(query, sess: dict):
    sess["step"] = "iterations"
    text = (
        "🔁 *Step 5 / 5 — Training iterations*\n\n"
        "How long to train the 3D Gaussian Splatting model.\n"
        "More iterations = sharper, more detailed result.\n\n"
        "• *3 000* — Quick preview (~20s)\n"
        "• *7 000* — Standard _(recommended)_ (~1 min)\n"
        "• *15 000* — High quality (~2 min)\n"
        "• *30 000* — Ultra (~4 min)"
    )
    markup = kb([
        [("3 000 — Preview",  "w:iterations:3000"),
         ("7 000 — Standard", "w:iterations:7000")],
        [("15 000 — High",    "w:iterations:15000"),
         ("30 000 — Ultra",   "w:iterations:30000")],
    ])
    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=markup)


# ── Telegram handlers ─────────────────────────────────────────────────────────

async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Video → 3D Gaussian Splatting*\n\n"
        "Send me a video and I'll guide you through the settings to create a 3D model!\n\n"
        "*I accept:*\n"
        "📹  A video file (up to 50 MB)\n"
        "🔗  A Google Drive share link\n"
        "🌐  Any direct video URL\n\n"
        "Open the result at [supersplat.xyz](https://supersplat.xyz) — just drag and drop.",
        parse_mode="Markdown",
        disable_web_page_preview=True,
    )


async def handle_video(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    media = update.message.video or update.message.document
    if not media:
        await update.message.reply_text("❌ Couldn't read that file. Please try again.")
        return

    size_mb = (media.file_size or 0) / 1_048_576
    if size_mb > 50:
        await update.message.reply_text(
            f"⚠️ That file is {size_mb:.0f} MB — Telegram bots max out at 50 MB.\n\n"
            "Please upload to Google Drive and send me the share link instead."
        )
        return

    msg = await update.message.reply_text("⬇️ Downloading your video…")
    tg_file = await media.get_file()
    data    = bytes(await tg_file.download_as_bytearray())

    await msg.edit_text("☁️ Uploading to cloud storage…")
    name    = getattr(media, "file_name", None) or "video.mp4"
    url     = await upload_to_s3(data, name)
    await msg.delete()

    sessions[update.effective_user.id] = {"video_url": url}
    await ask_backend(update.message, update.effective_user.id)


async def handle_text(update: Update, _: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    url = extract_gdrive_url(text)
    if not url:
        if text.lower().startswith(("http://", "https://")):
            url = text
    if url:
        sessions[update.effective_user.id] = {"video_url": url}
        await ask_backend(update.message, update.effective_user.id)
        return

    await update.message.reply_text(
        "🤔 I didn't recognise that.\n\n"
        "Please send a video file, a Google Drive link, or a direct video URL.\n"
        "Type /start to see the instructions again."
    )


async def handle_wizard(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Single callback handler for all wizard steps (callback_data starts with 'w:')."""
    query   = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    sess    = sessions.get(user_id)
    if not sess:
        await query.edit_message_text("❌ Session expired. Please send your video again.")
        return

    try:
        _, step, value = query.data.split(":", 2)

        if step == "backend":
            sess["backend"]  = value
            sess["arg_name"] = BACKEND_INFO[value]["arg_name"]
            await ask_fps(query, sess)

        elif step == "fps":
            sess["fps"] = int(value)
            await ask_resolution(query, sess)

        elif step == "resolution":
            sess["resolution"] = int(value)
            await ask_backend_arg(query, sess)

        elif step == "arg":
            sess["arg_value"] = int(value)
            await ask_iterations(query, sess)

        elif step == "iterations":
            sess["iterations"] = int(value)
            sessions.pop(user_id, None)
            await query.edit_message_text(
                f"✅ *All set! Submitting your job…*\n\n{session_summary(sess)}",
                parse_mode="Markdown",
            )
            ctx.application.create_task(_run_job(query, sess.copy()))

    except Exception as exc:
        logger.exception("Wizard error")
        await query.edit_message_text(f"❌ Error: {exc}")


async def _run_job(query, sess: dict):
    try:
        job_id = await runpod_submit(sess)
        backend_label = BACKEND_INFO.get(sess["backend"], {}).get("label", sess["backend"])
        await query.edit_message_text(
            f"🔄 *Processing on GPU…*\n\n"
            f"{session_summary(sess)}"
            f"\n_Job ID: `{job_id}`_\n"
            f"_I'll update this message when it's done._",
            parse_mode="Markdown",
        )

        result = await runpod_poll(job_id)

        if result["status"] == "COMPLETED":
            out     = result.get("output", {})
            t       = out.get("timings", {})
            ply_url = out.get("ply_url", "")
            ply_mb  = out.get("ply_size_mb", 0)

            def row(label, key):
                v = t.get(key, 0)
                return f"  {label}: {fmt_time(v)}\n" if v > 0 else ""

            timings_str = (
                row("Frame extraction",   "frame_extraction")
                + row("Feature extraction", "feature_extraction")
                + row("Feature matching",   "feature_matching")
                + row("SfM reconstruction", "sfm_reconstruction")
                + row("Undistortion",       "undistortion")
                + row("3DGS training",      "training")
                + f"  *Total: {fmt_time(t.get('total', 0))}*"
            )

            await query.edit_message_text(
                f"🎉 *Your 3D model is ready!*\n\n"
                f"⏱ *Timings:*\n{timings_str}\n\n"
                f"📦 *File size:* {ply_mb:.1f} MB\n\n"
                f"⬇️ *Download your `.ply`:*\n{ply_url}\n\n"
                f"_Link valid 24 h — open at_ [supersplat.xyz](https://supersplat.xyz)",
                parse_mode="Markdown",
                disable_web_page_preview=True,
            )
        else:
            err = str(result.get("error", "Unknown error"))
            snippet = err[-400:].strip()
            await query.edit_message_text(
                f"❌ *Processing failed.*\n\n```\n{snippet}\n```",
                parse_mode="Markdown",
            )

    except Exception as exc:
        logger.exception("Unhandled error in _run_job")
        await query.edit_message_text(f"❌ An unexpected error occurred:\n{exc}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def error_handler(update: object, ctx: ContextTypes.DEFAULT_TYPE):
    logger.error("Unhandled exception", exc_info=ctx.error)


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_error_handler(error_handler)

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_start))
    app.add_handler(MessageHandler(
        filters.VIDEO
        | filters.Document.MimeType("video/mp4")
        | filters.Document.MimeType("video/quicktime")
        | filters.Document.MimeType("video/x-msvideo")
        | filters.Document.MimeType("video/x-matroska"),
        handle_video,
    ))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_wizard, pattern=r"^w:"))

    logger.info("Bot starting…")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
