"""
═══════════════════════════════════
📄 FILE 07/42: backend/ai/context_memory.py
═══════════════════════════════════

BrailleVision AI — Sentence Context Memory
Maintains a sliding window of decoded sentences to improve
LLM error correction accuracy via topic-aware context.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, deque
from typing import Optional

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SIZE = 5
DEFAULT_WORD_BUFFER = 50
TOP_TOPIC_WORDS = 3

# Common English stopwords to exclude from topic detection
STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "this", "that", "these", "those", "it", "its", "i", "me", "my",
    "we", "our", "you", "your", "he", "his", "she", "her", "they",
    "their", "what", "which", "who", "whom", "how", "when", "where",
    "why", "all", "any", "every", "just", "also", "very", "more",
})


# ─────────────────────────────────────────────────────────────
# CONTEXT MEMORY CLASS
# ─────────────────────────────────────────────────────────────


class ContextMemory:
    """
    Maintains a sliding window of recent Braille-decoded sentences.

    Used to provide the AI error corrector with recent context so it
    can infer the topic/domain and make more accurate corrections.
    For example, if recent text is about "medicine dosage", the corrector
    can better resolve "tke 2 tblets" → "take 2 tablets".
    """

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        """
        Initialise context memory.

        Args:
            window_size: Maximum number of sentences to retain (default 5).
        """
        self.sentences: deque[str] = deque(maxlen=window_size)
        self.words: deque[str] = deque(maxlen=DEFAULT_WORD_BUFFER)
        self.topic: Optional[str] = None
        self._window_size = window_size
        logger.info("ContextMemory initialised (window=%d)", window_size)

    # ------------------------------------------------------------------
    # ADD SENTENCE
    # ------------------------------------------------------------------

    def add_sentence(self, sentence: str) -> None:
        """
        Add a decoded sentence to the context window.

        Automatically updates word buffer and topic hint.

        Args:
            sentence: Raw decoded/corrected text from one scan.
        """
        cleaned = self._clean(sentence)
        if not cleaned:
            logger.debug("add_sentence: empty after cleaning, skipped")
            return

        self.sentences.append(cleaned)

        # Extract content words
        words = self._tokenize(cleaned)
        for word in words:
            if word not in STOPWORDS and len(word) > 2:
                self.words.append(word)

        # Update topic
        self.topic = self.get_topic_hint()
        logger.debug("add_sentence: added '%s...'  topic='%s'", cleaned[:40], self.topic)

    # ------------------------------------------------------------------
    # GETTERS
    # ------------------------------------------------------------------

    def get_context_string(self) -> str:
        """
        Return all retained sentences joined into a single string.

        Returns:
            Context string for inclusion in LLM prompts.
        """
        return ". ".join(self.sentences)

    def get_topic_hint(self) -> str:
        """
        Identify the most common content words across recent sentences.

        Excludes stopwords. Returns the top-3 most frequent words
        comma-joined as a topic hint.

        Returns:
            Comma-joined topic hint string, or '' if no words retained.
        """
        if not self.words:
            return ""

        counter = Counter(self.words)
        top = [word for word, _ in counter.most_common(TOP_TOPIC_WORDS)]
        hint = ", ".join(top)
        logger.debug("get_topic_hint: '%s'", hint)
        return hint

    def get_correction_context(self) -> str:
        """
        Build a context string suitable for inclusion in the LLM correction prompt.

        Combines recent sentence history with the inferred topic hint to
        give the LLM maximum disambiguation power.

        Returns:
            Context prompt string.
        """
        ctx = self.get_context_string()
        topic = self.get_topic_hint()

        parts: list[str] = []
        if ctx:
            parts.append(f"Previous text context: {ctx}")
        if topic:
            parts.append(f"Topic hints: {topic}")

        result = ". ".join(parts)
        logger.debug("get_correction_context: '%s'", result[:80])
        return result

    def get_sentence_count(self) -> int:
        """Return the number of sentences currently retained."""
        return len(self.sentences)

    def get_all_sentences(self) -> list[str]:
        """Return a copy of all retained sentences in order."""
        return list(self.sentences)

    # ------------------------------------------------------------------
    # CLEAR
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset all retained context (sentences, words, topic)."""
        self.sentences.clear()
        self.words.clear()
        self.topic = None
        logger.info("ContextMemory cleared")

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _clean(self, text: str) -> str:
        """Strip extra whitespace and control characters from text."""
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def _tokenize(self, text: str) -> list[str]:
        """Split text into lowercase alpha words."""
        return re.findall(r"[a-zA-Z]+", text.lower())

    # ------------------------------------------------------------------
    # REPR
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ContextMemory(window={self._window_size}, "
            f"sentences={len(self.sentences)}, "
            f"words={len(self.words)}, "
            f"topic='{self.topic}')"
        )


# ─────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    print("\n" + "=" * 50)
    print("  ContextMemory Smoke Test")
    print("=" * 50)

    mem = ContextMemory(window_size=3)
    print(f"  Initial: {mem}")

    mem.add_sentence("The patient should take two tablets daily.")
    mem.add_sentence("Dosage must not exceed four tablets per day.")
    mem.add_sentence("Consult your doctor before changing medication.")

    print(f"  After 3 sentences: {mem}")
    print(f"  Context: '{mem.get_context_string()}'")
    print(f"  Topic hint: '{mem.get_topic_hint()}'")
    print(f"  Correction context: '{mem.get_correction_context()}'")

    # Test window overflow (window_size=3, add 4th sentence)
    mem.add_sentence("Keep tablets away from children.")
    print(f"\n  After 4th sentence (window=3): {mem}")
    print(f"  Sentence count: {mem.get_sentence_count()} (should be 3)")
    assert mem.get_sentence_count() == 3, "Window overflow not working!"
    print("  ✓ Window overflow correct")

    # Test clear
    mem.clear()
    print(f"\n  After clear: {mem}")
    assert mem.get_sentence_count() == 0
    assert mem.get_context_string() == ""
    print("  ✓ Clear works")

    # Test empty input
    mem.add_sentence("")
    mem.add_sentence("   ")
    assert mem.get_sentence_count() == 0
    print("  ✓ Empty/whitespace inputs ignored")

    print("\n✅ Smoke test complete.\n")
