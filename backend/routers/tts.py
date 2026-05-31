"""
═══════════════════════════════════
📄 FILE 15/42: backend/routers/tts.py
═══════════════════════════════════

BrailleVision AI — TTS API Router
Endpoints for neural speech synthesis and voice listing.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Optional

from fastapi import APIRouter, HTTPException  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
from pydantic import BaseModel  # type: ignore

from core.tts_engine import BrailleTTSEngine, VOICE_MAP  # type: ignore

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tts", tags=["tts"])

_tts_engine: Optional[BrailleTTSEngine] = None


def get_tts_engine() -> BrailleTTSEngine:
    """Return singleton TTS engine."""
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = BrailleTTSEngine()
    return _tts_engine


# ─────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────


class SpeakRequest(BaseModel):
    """Request body for /tts/speak."""
    text: str
    lang: str = "en"
    rate: str = "+0%"


class GuidanceRequest(BaseModel):
    """Request body for /tts/guidance."""
    message: str
    lang: str = "en"


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────


@router.post("/speak", summary="Generate speech audio for decoded Braille text")
async def speak_text(request: SpeakRequest) -> StreamingResponse:
    """
    Convert text to MP3 speech using a neural voice.

    Returns:
        StreamingResponse with audio/mpeg content.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    engine = get_tts_engine()

    try:
        audio_bytes = await engine.generate_speech_bytes(
            text=request.text,
            lang=request.lang,
            rate=request.rate,
        )
    except RuntimeError as exc:
        logger.error("/tts/speak failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"TTS generation failed: {exc}") from exc

    if not audio_bytes:
        raise HTTPException(status_code=503, detail="TTS returned empty audio.")

    logger.info("/tts/speak: %d bytes for lang=%s", len(audio_bytes), request.lang)
    return StreamingResponse(
        BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": 'attachment; filename="braille_speech.mp3"',
            "Content-Length": str(len(audio_bytes)),
        },
    )


@router.post("/guidance", summary="Generate fast-rate guidance audio")
async def speak_guidance(request: GuidanceRequest) -> StreamingResponse:
    """
    Generate faster-rate audio for camera guidance messages.

    Uses 25% faster speech rate for snappier feedback.
    Returns:
        StreamingResponse with audio/mpeg content.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    engine = get_tts_engine()
    engine.set_language(request.lang)

    try:
        audio_bytes = await engine.generate_guidance_bytes(request.message)
    except RuntimeError as exc:
        logger.error("/tts/guidance failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Guidance TTS failed: {exc}") from exc

    return StreamingResponse(
        BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Content-Length": str(len(audio_bytes))},
    )


@router.get("/voices", summary="List all available neural TTS voices")
async def get_voices() -> dict:
    """
    Return all available neural TTS voices and their language codes.

    Returns:
        Dict with voices list and total count.
    """
    voices = [
        {"code": code, "voice": voice, "language": code}
        for code, voice in VOICE_MAP.items()
    ]
    return {
        "voices": voices,
        "total": len(voices),
    }
