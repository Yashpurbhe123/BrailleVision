"""
BrailleVision AI — Multi-Language Translator
Google Translate via deep-translator with caching.
Supports 16 languages + maps each to its edge-tts neural voice.
"""

from __future__ import annotations

import hashlib
import logging

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: dict[str, str] = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
}

# Inverse mapping: code → name
CODE_TO_NAME: dict[str, str] = {v: k for k, v in SUPPORTED_LANGUAGES.items()}

# edge-tts neural voice per language code
TTS_VOICE_MAP: dict[str, str] = {
    "en": "en-US-JennyNeural",
    "hi": "hi-IN-SwaraNeural",
    "ta": "ta-IN-PallaviNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
}


# ─────────────────────────────────────────────────────────────
# TRANSLATOR CLASS
# ─────────────────────────────────────────────────────────────


class BrailleTranslator:
    """
    Translates Braille-decoded text to any of 16+ target languages.

    Uses Google Translate via deep-translator with a simple hash cache
    to avoid duplicate network calls for repeated identical text.
    """

    def __init__(self) -> None:
        """Initialise translator with an empty result cache."""
        self._cache: dict[str, dict] = {}
        logger.info(
            "BrailleTranslator ready. Supported languages: %d",
            len(SUPPORTED_LANGUAGES),
        )

    # ------------------------------------------------------------------
    # TRANSLATE
    # ------------------------------------------------------------------

    def translate(self, text: str, target_lang: str) -> dict:
        """
        Translate text to the specified language code.

        Args:
            text: Source text (assumed English from Braille decode).
            target_lang: BCP-47 language code, e.g. 'hi', 'fr', 'zh-CN'.

        Returns:
            Dict with original, translated, target_language, language_name,
            success fields. On failure, includes an 'error' key.
        """
        if not text or not text.strip():
            return {
                "original": text,
                "translated": text,
                "target_language": target_lang,
                "language_name": CODE_TO_NAME.get(target_lang, target_lang),
                "success": True,
                "note": "empty_input",
            }

        # Validate language code
        valid_codes = set(SUPPORTED_LANGUAGES.values())
        if target_lang not in valid_codes:
            logger.warning("translate: unsupported language code '%s'", target_lang)
            return {
                "original": text,
                "translated": text,
                "target_language": target_lang,
                "language_name": "Unknown",
                "success": False,
                "error": f"Unsupported language code: {target_lang}. "
                         f"Supported: {sorted(valid_codes)}",
            }

        # English to English — skip API
        if target_lang == "en":
            return {
                "original": text,
                "translated": text,
                "target_language": "en",
                "language_name": "English",
                "success": True,
            }

        # Cache check
        cache_key = hashlib.md5(f"{text}|{target_lang}".encode()).hexdigest()
        if cache_key in self._cache:
            logger.debug("translate: cache hit for %s:%s", target_lang, cache_key[:8])
            return self._cache[cache_key]

        try:
            from deep_translator import GoogleTranslator  # type: ignore

            translated = GoogleTranslator(
                source="auto", target=target_lang
            ).translate(text)

            lang_name = CODE_TO_NAME.get(target_lang, target_lang)
            result: dict = {
                "original": text,
                "translated": translated,
                "target_language": target_lang,
                "language_name": lang_name,
                "success": True,
            }
            self._cache[cache_key] = result
            logger.info(
                "translate: '%s...' → [%s] '%s...'",
                text[:30],
                target_lang,
                translated[:30],
            )
            return result

        except Exception as exc:
            logger.error("translate: failed for lang=%s: %s", target_lang, exc)
            return {
                "original": text,
                "translated": text,
                "target_language": target_lang,
                "language_name": CODE_TO_NAME.get(target_lang, target_lang),
                "success": False,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # TTS VOICE MAPPING
    # ------------------------------------------------------------------

    def get_tts_voice_for_lang(self, lang_code: str) -> str:
        """
        Return the edge-tts neural voice name for a language code.

        Falls back to English voice if the code is not mapped.

        Args:
            lang_code: BCP-47 language code.

        Returns:
            edge-tts voice name string.
        """
        voice = TTS_VOICE_MAP.get(lang_code, TTS_VOICE_MAP["en"])
        logger.debug("get_tts_voice_for_lang: %s → %s", lang_code, voice)
        return voice

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------

    def get_supported_languages(self) -> dict[str, str]:
        """
        Return the full map of language name → language code.

        Returns:
            Dict like {'Hindi': 'hi', 'French': 'fr', ...}
        """
        return dict(SUPPORTED_LANGUAGES)

    def is_supported(self, lang_code: str) -> bool:
        """Check whether a language code is supported."""
        return lang_code in set(SUPPORTED_LANGUAGES.values())

    def clear_cache(self) -> None:
        """Clear the translation cache."""
        self._cache.clear()
        logger.info("BrailleTranslator: cache cleared")


# ─────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("\n" + "=" * 50)
    print("  BrailleTranslator Smoke Test")
    print("=" * 50)

    translator = BrailleTranslator()

    # Test supported language lookup
    langs = translator.get_supported_languages()
    print(f"  Supported languages: {len(langs)}")
    print(f"  Sample: {list(langs.items())[:4]}")

    # Test voice mapping
    for code in ["en", "hi", "fr", "zh-CN", "xx"]:
        voice = translator.get_tts_voice_for_lang(code)
        print(f"  Voice [{code}]: {voice}")

    # Test is_supported
    assert translator.is_supported("hi") is True
    assert translator.is_supported("xx") is False
    print("  ✓ is_supported works")

    # Test unsupported language
    result = translator.translate("Hello world", "xx")
    assert result["success"] is False
    print(f"  ✓ Unsupported lang returns success=False: {result['error'][:50]}")

    # Test empty input
    result = translator.translate("", "hi")
    assert result["success"] is True
    print("  ✓ Empty input handled")

    # Test English→English passthrough
    result = translator.translate("Hello world", "en")
    assert result["translated"] == "Hello world"
    print("  ✓ English passthrough works")

    # Attempt live translation (may fail without network)
    try:
        result = translator.translate("Hello world", "hi")
        if result["success"]:
            print(f"  ✓ Live translation: '{result['translated']}'")
        else:
            print(f"  ⚠ Live translation unavailable: {result.get('error', 'unknown')}")
    except Exception as e:
        print(f"  ⚠ Translation network test skipped: {e}")

    print("\n✅ Smoke test complete.\n")
