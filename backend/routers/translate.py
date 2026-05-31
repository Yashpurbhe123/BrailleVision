"""
═══════════════════════════════════
📄 FILE 16/42: backend/routers/translate.py
═══════════════════════════════════

BrailleVision AI — Translation API Router
Endpoints for text translation and supported language lists.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException  # type: ignore
from pydantic import BaseModel, Field  # type: ignore

from core.translator import BrailleTranslator  # type: ignore

# ─────────────────────────────────────────────────────────────
# CONSTANTS & SETUP
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/translate", tags=["translate"])

_translator: Optional[BrailleTranslator] = None


def get_translator() -> BrailleTranslator:
    """Return singleton BrailleTranslator instance."""
    global _translator
    if _translator is None:
        _translator = BrailleTranslator()
    return _translator


# ─────────────────────────────────────────────────────────────
# REQUEST & RESPONSE MODELS
# ─────────────────────────────────────────────────────────────


class TranslateRequest(BaseModel):
    """Request body for text translation."""
    text: str = Field(..., description="English text decoded from Braille to translate")
    target_lang: str = Field(..., description="BCP-47 target language code (e.g., 'hi', 'es', 'fr')")


class TranslationResponse(BaseModel):
    """Response returned after a translation request."""
    success: bool
    original: str
    translated: str
    target_language: str
    language_name: str
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────


@router.post("/", response_model=TranslationResponse, summary="Translate text to a target language")
async def translate_text(request: TranslateRequest) -> TranslationResponse:
    """
    Translate English (Braille decoded) text to one of the 16+ supported languages.

    Uses deep-translator with MD5-based query caching for rapid responses.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    translator = get_translator()

    try:
        result = translator.translate(request.text, request.target_lang)
    except Exception as exc:
        logger.error("/translate failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Translation process failed: {exc}") from exc

    if not result.get("success", False):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Translation failed due to unsupported language or service issue.")
        )

    return TranslationResponse(
        success=result["success"],
        original=result["original"],
        translated=result["translated"],
        target_language=result["target_language"],
        language_name=result["language_name"],
        error=result.get("error"),
    )


@router.get("/languages", summary="List all supported languages and codes")
async def get_supported_languages() -> dict:
    """
    List all supported languages with their human-readable names and BCP-47 codes.

    Returns:
        Dict mapping language names to language codes.
    """
    translator = get_translator()
    languages = translator.get_supported_languages()
    return {
        "languages": languages,
        "total": len(languages),
    }
