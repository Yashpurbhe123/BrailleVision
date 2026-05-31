"""
BrailleVision AI — Neural Text-to-Speech Engine
Microsoft edge-tts for high-quality neural voices in 17 languages.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from io import BytesIO
from typing import Optional

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

VOICE_MAP: dict[str, str] = {
    "en":    "en-US-JennyNeural",
    "hi":    "hi-IN-SwaraNeural",
    "ta":    "ta-IN-PallaviNeural",
    "mr":    "mr-IN-AarohiNeural",
    "te":    "te-IN-MohanNeural",
    "kn":    "kn-IN-SapnaNeural",
    "bn":    "bn-IN-TanishaaNeural",
    "gu":    "gu-IN-DhwaniNeural",
    "pa":    "pa-IN-OjasvNeural",
    "es":    "es-ES-ElviraNeural",
    "fr":    "fr-FR-DeniseNeural",
    "de":    "de-DE-KatjaNeural",
    "ar":    "ar-SA-ZariyahNeural",
    "ja":    "ja-JP-NanamiNeural",
    "zh-CN": "zh-CN-XiaoxiaoNeural",
    "pt":    "pt-BR-FranciscaNeural",
    "ru":    "ru-RU-SvetlanaNeural",
}

DEFAULT_RATE = "+0%"
DEFAULT_VOLUME = "+100%"
GUIDANCE_RATE = "+25%"   # faster for camera guidance messages
FALLBACK_VOICE = "en-US-JennyNeural"


# ─────────────────────────────────────────────────────────────
# TTS ENGINE CLASS
# ─────────────────────────────────────────────────────────────


class BrailleTTSEngine:
    """
    Async neural Text-to-Speech engine using Microsoft edge-tts.

    Generates MP3 audio bytes in memory (no temp files required).
    Supports 17 languages with natural-sounding neural voices.
    Provides a faster-rate mode for camera guidance messages.
    """

    def __init__(
        self,
        default_lang: str = "en",
        rate: str = DEFAULT_RATE,
        volume: str = DEFAULT_VOLUME,
    ) -> None:
        """
        Initialise the TTS engine.

        Args:
            default_lang: Default language code (default 'en').
            rate: Speech rate adjustment, e.g. '+0%', '+20%', '-10%'.
            volume: Volume adjustment, e.g. '+100%', '+50%'.
        """
        self.default_lang = default_lang
        self.rate = rate
        self.volume = volume
        self.cache_dir = "./data/tts_cache"
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception as e:
            logger.warning("Could not create tts cache directory: %s", e)
        logger.info(
            "BrailleTTSEngine ready. lang=%s rate=%s volume=%s cache_dir=%s",
            default_lang, rate, volume, self.cache_dir,
        )

    # ------------------------------------------------------------------
    # GENERATE SPEECH
    # ------------------------------------------------------------------

    async def generate_speech_bytes(
        self,
        text: str,
        lang: Optional[str] = None,
        rate: Optional[str] = None,
    ) -> bytes:
        """
        Generate MP3 audio bytes for the given text.

        Args:
            text: Text to synthesise.
            lang: Language code (uses default_lang if None).
            rate: Speech rate override (uses engine default if None).

        Returns:
            MP3 audio as bytes object.

        Raises:
            RuntimeError: If edge-tts fails to generate audio.
        """
        if not text or not text.strip():
            logger.warning("generate_speech_bytes: empty text, returning silence stub")
            return b""

        # Check if the text contains any speakable characters (letters or numbers)
        if not any(c.isalnum() for c in text):
            logger.warning("generate_speech_bytes: no speakable characters in text '%s', returning silence", text)
            return b""

        resolved_lang = lang or self.default_lang
        resolved_rate = rate or self.rate
        voice = VOICE_MAP.get(resolved_lang, FALLBACK_VOICE)

        # Generate unique cache key based on voice parameters and clean text
        clean_text_key = text.strip()
        key_source = f"{voice}_{resolved_rate}_{self.volume}_{clean_text_key}"
        cache_key = hashlib.md5(key_source.encode("utf-8")).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.mp3")

        # 1. Check local file-based cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached_audio = f.read()
                if cached_audio:
                    logger.info(
                        "generate_speech_bytes: CACHE HIT for lang=%s voice=%s text='%s...' → %d bytes",
                        resolved_lang, voice, clean_text_key[:20], len(cached_audio),
                    )
                    return cached_audio
            except Exception as e:
                logger.warning("generate_speech_bytes: Failed to read from TTS cache file: %s", e)

        # 2. Cache Miss: Call edge_tts network API
        try:
            import edge_tts  # type: ignore

            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=resolved_rate,
                volume=self.volume,
            )

            buf = BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])

            audio_bytes = buf.getvalue()
            
            # Save generated audio to local cache
            if audio_bytes:
                try:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    with open(cache_file, "wb") as f:
                        f.write(audio_bytes)
                except Exception as e:
                    logger.warning("generate_speech_bytes: Failed to write to TTS cache: %s", e)

            logger.info(
                "generate_speech_bytes: lang=%s voice=%s → %d bytes (CACHE MISS - generated & saved)",
                resolved_lang, voice, len(audio_bytes),
            )
            return audio_bytes

        except Exception as exc:
            logger.error("generate_speech_bytes: edge-tts failed: %s", exc)
            raise RuntimeError(f"TTS generation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # GUIDANCE AUDIO
    # ------------------------------------------------------------------

    async def generate_guidance_bytes(self, message: str) -> bytes:
        """
        Generate fast-rate guidance audio (e.g. 'Move closer', 'Hold steady').

        Uses a 25% faster speech rate for snappier camera feedback.

        Args:
            message: Short guidance string.

        Returns:
            MP3 bytes for the guidance message.
        """
        return await self.generate_speech_bytes(
            text=message,
            lang=self.default_lang,
            rate=GUIDANCE_RATE,
        )

    # ------------------------------------------------------------------
    # SAVE TO FILE
    # ------------------------------------------------------------------

    async def save_to_file(
        self,
        text: str,
        output_path: str,
        lang: str = "en",
    ) -> str:
        """
        Generate speech and write MP3 to a file on disk.

        Args:
            text: Text to synthesise.
            output_path: Absolute path for output MP3 file.
            lang: Language code.

        Returns:
            Absolute path to the written file.

        Raises:
            IOError: If file cannot be written.
        """
        audio_bytes = await self.generate_speech_bytes(text, lang=lang)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        logger.info("save_to_file: wrote %d bytes to '%s'", len(audio_bytes), output_path)
        return output_path

    # ------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------

    def set_language(self, lang_code: str) -> None:
        """
        Update the default output language.

        Args:
            lang_code: BCP-47 language code, e.g. 'hi', 'fr'.
        """
        if lang_code not in VOICE_MAP:
            logger.warning(
                "set_language: '%s' not in VOICE_MAP — keeping '%s'",
                lang_code, self.default_lang,
            )
            return
        self.default_lang = lang_code
        logger.info("set_language: → %s (%s)", lang_code, VOICE_MAP[lang_code])

    def get_available_voices(self) -> list[dict]:
        """
        Return all configured voices as a list of info dicts.

        Returns:
            List of dicts: {language_name, code, voice}.
        """
        from backend.core.translator import CODE_TO_NAME  # type: ignore
        result = []
        for code, voice in VOICE_MAP.items():
            result.append(
                {
                    "language_name": CODE_TO_NAME.get(code, code),
                    "code": code,
                    "voice": voice,
                }
            )
        return result


# ─────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("\n" + "=" * 50)
    print("  BrailleTTSEngine Smoke Test")
    print("=" * 50)

    engine = BrailleTTSEngine(default_lang="en")

    async def run_tests() -> None:
        # Test voice map coverage
        print(f"  Voice map entries: {len(VOICE_MAP)}")
        for code, voice in list(VOICE_MAP.items())[:4]:
            print(f"    [{code}] -> {voice}")

        # Test set_language
        engine.set_language("hi")
        assert engine.default_lang == "hi"
        engine.set_language("en")
        assert engine.default_lang == "en"
        print("  [OK] set_language works")

        # Test unknown language fallback
        engine.set_language("xx")
        assert engine.default_lang == "en"  # unchanged
        print("  [OK] Unknown language ignored")

        # Attempt live TTS (may fail without edge-tts installed)
        try:
            audio = await engine.generate_speech_bytes("Hello from BrailleVision AI")
            if audio:
                print(f"  [OK] generate_speech_bytes: {len(audio)} bytes MP3")
            else:
                print("  [WARN] generate_speech_bytes returned empty (edge-tts not installed?)")
        except Exception as e:
            print(f"  [WARN] TTS unavailable in test env: {e}")

        # Guidance
        try:
            audio = await engine.generate_guidance_bytes("Move closer to the page")
            print(f"  [OK] generate_guidance_bytes: {len(audio)} bytes")
        except Exception as e:
            print(f"  [WARN] Guidance TTS skipped: {e}")

        print("\n  Empty text test:")
        audio = await engine.generate_speech_bytes("")
        assert audio == b""
        print("  [OK] Empty text returns b''")

    asyncio.run(run_tests())
    print("\n[OK] Smoke test complete.\n")
