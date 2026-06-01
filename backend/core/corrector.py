"""
BrailleVision AI — AI Error Corrector
Groq primary + pyspellchecker fallback for fixing
OCR-style errors in camera-decoded Braille text.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import os
from typing import Optional

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

GROQ_DEFAULT_MODEL = "llama-3.1-8b-instant"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

LLM_MAX_TOKENS = 300
LLM_TEMPERATURE = 0.1

SYSTEM_PROMPT = (
    "You are an expert at correcting text that was decoded from physical Braille "
    "paper using a camera-based AI system. The text may contain errors due to "
    "imperfect dot detection (similar to OCR errors) or over-segmentation repetition errors.\n\n"
    "Your job: Fix ONLY likely detection errors. Preserve meaning exactly. "
    "Do not add or remove words unless they are clearly wrong or repeated.\n\n"
    "IMPORTANT: You may be provided with previously decoded text context for topic/domain reference. "
    "Do NOT merge, prepend, or append this context to the corrected text. "
    "Your output must ONLY contain the corrected version of the current text to correct.\n\n"
    "Common errors to fix:\n"
    "- Repeated words/substrings (e.g. 'olly olly olly oxen' → 'olly oxen', 'hello hello' → 'hello')\n"
    "- Missing letters: 'helo' → 'hello'\n"
    "- Swapped letters: 'teh' → 'the'\n"
    "- Wrong letters from similar dot patterns: 'i' ↔ 'e', 'h' ↔ 'b'\n"
    "- Missing spaces between words creating run-on words\n"
    "- Random extra characters from noise dots\n\n"
    "Return ONLY the corrected text. No explanation. No quotes. Just the fixed text."
)

LLM_CONFIDENCE = 0.95
SPELLCHECK_CONFIDENCE = 0.70
NO_CORRECTION_CONFIDENCE = 1.0


# ─────────────────────────────────────────────────────────────
# CORRECTOR CLASS
# ─────────────────────────────────────────────────────────────


class AIErrorCorrector:
    """
    Two-tier error correction for camera-decoded Braille text.

    Tier 1 (preferred): Groq Llama-3.1 with Braille-aware system prompt.
    Tier 2 (fallback):  pyspellchecker offline word-by-word correction.

    Results are cached by MD5 hash to avoid redundant API calls.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialise the corrector.

        Args:
            api_key: Optional API key override. If None, checks GROQ_API_KEY env var.
                     If key is absent/invalid, falls back to spellcheck only.
        """
        self._cache: dict[str, tuple[str, str]] = {}
        self.llm_available = False
        self._client = None
        self._model = GROQ_DEFAULT_MODEL

        # Try to resolve keys
        groq_key = api_key or os.getenv("GROQ_API_KEY", "")

        # Check if we should use Groq
        if groq_key and groq_key not in ("your_groq_key_here", "your_groq_api_key_here", ""):
            try:
                from openai import OpenAI  # type: ignore
                self._model = os.getenv("GROQ_MODEL", GROQ_DEFAULT_MODEL)
                self._client = OpenAI(
                    api_key=groq_key,
                    base_url=GROQ_BASE_URL
                )
                self.llm_available = True
                logger.info("AIErrorCorrector: Groq %s ready", self._model)
            except Exception as exc:
                logger.warning("AIErrorCorrector: Groq init failed: %s", exc)

        # Initialise spell checker
        try:
            from spellchecker import SpellChecker  # type: ignore
            self._spell = SpellChecker()
            logger.info("AIErrorCorrector: pyspellchecker ready")
        except Exception as exc:
            self._spell = None
            logger.warning("AIErrorCorrector: pyspellchecker unavailable: %s", exc)

        mode = "llm+spellcheck" if self.llm_available else "spellcheck-only" if self._spell else "passthrough"
        logger.info("AIErrorCorrector active mode: %s", mode)

    # ------------------------------------------------------------------
    # LLM CORRECTION
    # ------------------------------------------------------------------

    def correct_with_llm(
        self, text: str, context: str = ""
    ) -> tuple[str, str]:
        """
        Correct text using Groq with Braille-aware prompting.

        Results are memoised by MD5 hash of the input text.

        Args:
            text: Raw decoded Braille text to correct.
            context: Optional context string from ContextMemory.

        Returns:
            Tuple of (corrected_text, method_string).
            method_string is 'llm' on success, 'llm_failed' on error.
        """
        if not self.llm_available or self._client is None:
            return text, "llm_unavailable"

        cache_key = hashlib.md5(f"{text}|{context}".encode()).hexdigest()
        if cache_key in self._cache:
            logger.debug("correct_with_llm: cache hit for key %s", cache_key[:8])
            return self._cache[cache_key]

        try:
            if context:
                user_msg = (
                    f"--- CONTEXT OF PREVIOUSLY DECODED SENTENCES ---\n"
                    f"{context}\n"
                    f"--- END OF CONTEXT ---\n\n"
                    f"Text to correct: {text}"
                )
            else:
                user_msg = f"Text to correct: {text}"

            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
            corrected = response.choices[0].message.content.strip()
            result = (corrected, "llm")
            self._cache[cache_key] = result
            logger.info("correct_with_llm: '%s' → '%s'", text[:40], corrected[:40])
            return result

        except Exception as exc:
            logger.error("correct_with_llm: API call failed: %s", exc)
            return text, "llm_failed"

    # ------------------------------------------------------------------
    # SPELLCHECK CORRECTION
    # ------------------------------------------------------------------

    def correct_with_spellcheck(self, text: str) -> tuple[str, str]:
        """
        Word-by-word spell correction using pyspellchecker.

        Preserves punctuation and spacing structure.

        Args:
            text: Raw decoded Braille text.

        Returns:
            Tuple of (corrected_text, 'spellcheck').
        """
        if self._spell is None:
            logger.debug("correct_with_spellcheck: spell checker unavailable")
            return text, "spellcheck_unavailable"

        try:
            words = text.split()
            corrected_words: list[str] = []

            for word in words:
                # Strip punctuation for lookup
                stripped = word.strip(".,!?;:'\"()-")
                if not stripped or not stripped.isalpha():
                    corrected_words.append(word)
                    continue

                misspelled = self._spell.unknown([stripped.lower()])
                if misspelled:
                    candidate = self._spell.correction(stripped.lower())
                    if candidate and candidate != stripped.lower():
                        # Preserve original capitalisation
                        if stripped[0].isupper():
                            candidate = candidate.capitalize()
                        # Re-attach punctuation
                        prefix = word[: len(word) - len(word.lstrip(".,!?;:'\"()-"))]
                        suffix = word[len(stripped) + len(prefix):]
                        corrected_words.append(prefix + candidate + suffix)
                        logger.debug(
                            "spellcheck: '%s' → '%s'", stripped, candidate
                        )
                        continue
                corrected_words.append(word)

            corrected = " ".join(corrected_words)
            return corrected, "spellcheck"

        except Exception as exc:
            logger.error("correct_with_spellcheck: %s", exc)
            return text, "spellcheck_failed"

    # ------------------------------------------------------------------
    # DIFF CALCULATION
    # ------------------------------------------------------------------

    def get_diff(self, original: str, corrected: str) -> list[dict]:
        """
        Compute word-level diff between original and corrected text.

        Args:
            original: Pre-correction text.
            corrected: Post-correction text.

        Returns:
            List of change dicts: {original, corrected, position}.
            Only includes words that actually changed.
        """
        orig_words = original.split()
        corr_words = corrected.split()

        matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
        changes: list[dict] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ("replace", "delete", "insert"):
                changes.append(
                    {
                        "original": " ".join(orig_words[i1:i2]),
                        "corrected": " ".join(corr_words[j1:j2]),
                        "position": i1,
                        "type": tag,
                    }
                )

        return changes

    # ------------------------------------------------------------------
    # MAIN CORRECT
    # ------------------------------------------------------------------

    def correct(
        self,
        text: str,
        context: str = "",
        prefer_llm: bool = True,
    ) -> dict:
        """
        Apply best-available correction to decoded Braille text.

        Decision flow:
            1. Return unchanged if text is empty/whitespace.
            2. Apply pre-correction deduplication to collapse repeated segments.
            3. Try LLM if available and prefer_llm=True.
            4. Fall back to spellcheck on LLM failure/unavailability.
            5. Return passthrough if neither available.

        Args:
            text: Raw decoded text to correct.
            context: Context string from ContextMemory.
            prefer_llm: Whether to try LLM before spellcheck.

        Returns:
            Dict with original, corrected, method, was_corrected,
            changes, and confidence fields.
        """
        if not text or not text.strip() or len(text.strip()) == 1:
            return {
                "original": text,
                "corrected": text,
                "method": "none",
                "was_corrected": False,
                "changes": [],
                "confidence": NO_CORRECTION_CONFIDENCE,
            }

        import re
        dedup_text = text
        # Collapse 2+ repetitions of words >= 3 chars, and 3+ repetitions of any word
        dedup_text = re.sub(r"\b(\w{3,})(?:\s+\1\b)+", r"\1", dedup_text, flags=re.IGNORECASE)
        dedup_text = re.sub(r"\b(\w+)(?:\s+\1\b){2,}", r"\1", dedup_text, flags=re.IGNORECASE)

        corrected_text = dedup_text
        method = "none"
        confidence = NO_CORRECTION_CONFIDENCE

        # If deduplication did some work, we record it as pre_dedup
        if dedup_text != text:
            method = "pre_dedup"

        if prefer_llm and self.llm_available:
            llm_corrected, llm_method = self.correct_with_llm(dedup_text, context)
            if llm_method == "llm":
                corrected_text = llm_corrected
                method = "llm"
                confidence = LLM_CONFIDENCE
            else:
                # LLM failed — fall through to spellcheck
                spell_corrected, spell_method = self.correct_with_spellcheck(dedup_text)
                corrected_text = spell_corrected
                method = "spellcheck" if spell_method == "spellcheck" else method
                confidence = SPELLCHECK_CONFIDENCE

        elif self._spell is not None:
            spell_corrected, spell_method = self.correct_with_spellcheck(dedup_text)
            corrected_text = spell_corrected
            method = "spellcheck" if spell_method == "spellcheck" else method
            confidence = SPELLCHECK_CONFIDENCE

        was_corrected = corrected_text != text
        changes = self.get_diff(text, corrected_text) if was_corrected else []

        logger.info(
            "correct: method=%s was_corrected=%s  '%s' → '%s'",
            method,
            was_corrected,
            text[:30],
            corrected_text[:30],
        )

        return {
            "original": text,
            "corrected": corrected_text,
            "method": method,
            "was_corrected": was_corrected,
            "changes": changes,
            "confidence": confidence,
        }

    def clear_cache(self) -> None:
        """Clear the LLM response cache."""
        self._cache.clear()
        logger.info("AIErrorCorrector: cache cleared")


# ─────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("\n" + "=" * 50)
    print("  AIErrorCorrector Smoke Test")
    print("=" * 50)

    corrector = AIErrorCorrector()
    print(f"  llm_available: {corrector.llm_available}")
    print(f"  spellcheck: {corrector._spell is not None}")

    # Test with no API key — should use spellcheck
    test_cases = [
        "helo wrold",
        "i lvoe braille",
        "teh quck brwon fox",
        "",
        "   ",
        "Perfect sentence with no errors.",
    ]

    for text in test_cases:
        result = corrector.correct(text, prefer_llm=False)
        print(
            f"\n  Input:     '{text}'"
            f"\n  Corrected: '{result['corrected']}'"
            f"\n  Method:    {result['method']}"
            f"\n  Changes:   {result['changes']}"
        )

    # Test diff
    diff = corrector.get_diff("i lvoe braille", "i love braille")
    print(f"\n  Diff test: {diff}")

    # Test empty input
    r = corrector.correct("")
    assert r["method"] == "none"
    assert r["was_corrected"] is False
    print("\n  [OK] Empty input handled correctly")

    print("\n[SUCCESS] Smoke test complete.\n")
