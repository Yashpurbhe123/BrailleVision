"""
BrailleVision AI — Grade 1 & Grade 2 Braille Decoder
Decodes 6-dot Braille cell patterns into characters/text.
Supports capital indicator, number indicator, fuzzy Hamming
matching for error-tolerant recognition, and Grade 2 contractions.
"""

from __future__ import annotations

import logging
from typing import Union

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# Dot layout:
#  dot1  dot4
#  dot2  dot5
#  dot3  dot6
#
# Tuple format: (dot1, dot2, dot3, dot4, dot5, dot6) → 0 or 1

GRADE1_TABLE: dict[tuple[int, ...], str] = {
    # Letters a-z
    (1, 0, 0, 0, 0, 0): "a",
    (1, 1, 0, 0, 0, 0): "b",
    (1, 0, 0, 1, 0, 0): "c",
    (1, 0, 0, 1, 1, 0): "d",
    (1, 0, 0, 0, 1, 0): "e",
    (1, 1, 0, 1, 0, 0): "f",
    (1, 1, 0, 1, 1, 0): "g",
    (1, 1, 0, 0, 1, 0): "h",
    (0, 1, 0, 1, 0, 0): "i",
    (0, 1, 0, 1, 1, 0): "j",
    (1, 0, 1, 0, 0, 0): "k",
    (1, 1, 1, 0, 0, 0): "l",
    (1, 0, 1, 1, 0, 0): "m",
    (1, 0, 1, 1, 1, 0): "n",
    (1, 0, 1, 0, 1, 0): "o",
    (1, 1, 1, 1, 0, 0): "p",
    (1, 1, 1, 1, 1, 0): "q",
    (1, 1, 1, 0, 1, 0): "r",
    (0, 1, 1, 1, 0, 0): "s",
    (0, 1, 1, 1, 1, 0): "t",
    (1, 0, 1, 0, 0, 1): "u",
    (1, 1, 1, 0, 0, 1): "v",
    (0, 1, 0, 1, 1, 1): "w",
    (1, 0, 1, 1, 0, 1): "x",
    (1, 0, 1, 1, 1, 1): "y",
    (1, 0, 1, 0, 1, 1): "z",
    # Special indicators
    (0, 0, 0, 0, 0, 1): "[CAP]",   # dot 6 only
    (0, 0, 1, 1, 1, 1): "[NUM]",   # dots 3-4-5-6 (correct Braille number indicator)
    (0, 0, 0, 0, 0, 0): " ",
    # Punctuation
    (0, 1, 0, 0, 1, 1): ".",
    (0, 1, 0, 0, 0, 0): ",",
    (0, 1, 1, 0, 0, 1): "?",
    (0, 1, 1, 0, 1, 0): "!",
    (0, 0, 0, 1, 0, 1): "-",
    (0, 0, 0, 0, 1, 1): "'",
    (0, 1, 0, 0, 1, 0): ":",
    (0, 1, 1, 0, 0, 0): ";",
    (1, 1, 0, 0, 1, 1): "(",   # dots 1-2-5-6
    (0, 0, 1, 1, 0, 1): ")",   # dots 3-4-6
    (0, 0, 0, 1, 1, 0): "\"",
    (1, 0, 0, 0, 0, 1): "@",
    (0, 0, 1, 0, 0, 1): "#",
    (0, 0, 1, 0, 1, 0): "*",
    (1, 1, 0, 0, 0, 1): "&",
    (0, 1, 0, 1, 0, 1): "/",
    (0, 0, 1, 0, 0, 0): "~",
    (0, 0, 0, 1, 0, 0): "[ITER]",  # Iteration sign / apostrophe variant
}

# Number mode: same patterns as letters a-j but interpreted as digits 1-0
NUMBER_TABLE: dict[tuple[int, ...], str] = {
    (1, 0, 0, 0, 0, 0): "1",
    (1, 1, 0, 0, 0, 0): "2",
    (1, 0, 0, 1, 0, 0): "3",
    (1, 0, 0, 1, 1, 0): "4",
    (1, 0, 0, 0, 1, 0): "5",
    (1, 1, 0, 1, 0, 0): "6",
    (1, 1, 0, 1, 1, 0): "7",
    (1, 1, 0, 0, 1, 0): "8",
    (0, 1, 0, 1, 0, 0): "9",
    (0, 1, 0, 1, 1, 0): "0",
}

# Grade 2 contractions: multi-cell → word/sequence
# Each key is a tuple of cell patterns, value is the contraction
GRADE2_CONTRACTIONS: dict[tuple[int, ...], str] = {
    # Single-cell whole-word contractions
    (1, 1, 1, 1, 0, 1): "and",    # and-pattern used as contraction
}

# Single-cell Grade 2 whole-word contractions (pattern → word)
# IMPORTANT: Only patterns that do NOT overlap with Grade 1 letters a-z.
# The letter-based contractions (b=but, c=can, d=do, etc.) are deliberately
# excluded here because they share identical dot patterns with Grade 1 letters
# and would cause every occurrence of those letters to decode incorrectly.
# Grade 2 letter-based contractions are only valid in full Braille Grade 2 text
# where every word boundary is explicitly marked — not in general scanning.
GRADE2_SINGLE: dict[tuple[int, ...], str] = {
    # True unique whole-word contractions (patterns not used by a-z letters)
    (1, 1, 1, 1, 0, 1): "and",    # dots 1-2-3-4-6
    (1, 1, 1, 1, 1, 1): "for",    # dots 1-2-3-4-5-6
    (1, 1, 1, 0, 1, 1): "of",     # dots 1-2-3-5-6
    (0, 1, 1, 1, 1, 1): "with",   # dots 2-3-4-5-6
    (0, 1, 1, 1, 0, 1): "the",    # dots 2-3-4-6
}

# Grade 2 prefix/suffix contractions (pattern → affix)
# IMPORTANT: Only patterns NOT already in GRADE1_TABLE.
# Patterns that overlap Grade1 letters or punctuation are excluded
# because in_grade1 check fires first and bypasses affix lookup.
GRADE2_AFFIXES: dict[tuple[int, ...], str] = {
    (1, 0, 0, 0, 1, 1): "ch",     # dots 1-5-6
    # (1, 1, 0, 1, 0, 1): "gh"   ← REMOVED: same as '/'  in Grade1
    # (1, 1, 0, 0, 1, 1): "ed"   ← REMOVED: same as '('  in Grade1
    # (0, 1, 0, 1, 0, 1): "ow"   ← REMOVED: same as '/'  in Grade1
    (0, 1, 1, 0, 1, 1): "sh",
    (0, 1, 1, 1, 0, 1): "th",
    (0, 1, 1, 0, 0, 0): "wh",
    (1, 0, 0, 1, 1, 1): "er",
    (0, 1, 0, 0, 1, 1): "ou",
    (0, 0, 1, 0, 1, 1): "en",
    (0, 0, 1, 1, 0, 0): "in",
    (0, 0, 0, 1, 1, 1): "st",
    (0, 0, 1, 1, 1, 0): "ar",
}


# ─────────────────────────────────────────────────────────────
# DECODER CLASS
# ─────────────────────────────────────────────────────────────

class BrailleDecoder:
    """
    Decodes 6-dot Braille cell patterns into readable text.

    Supports:
    - Grade 1: Full alphabet, punctuation, numbers
    - Grade 2: Common contractions and affixes
    - Capital indicator and number mode
    - Fuzzy Hamming-distance matching for error tolerance
    """

    def __init__(self) -> None:
        """Initialise decoder with Grade 1 and Grade 2 lookup tables."""
        self.grade1 = GRADE1_TABLE
        self.number_table = NUMBER_TABLE
        self.grade2_single = GRADE2_SINGLE
        self.grade2_affixes = GRADE2_AFFIXES
        logger.info("BrailleDecoder initialised. Grade1 entries: %d", len(self.grade1))

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def detect_blank_cell(self, pattern: tuple[int, ...]) -> bool:
        """Return True if pattern is all zeros (blank/space cell)."""
        return pattern == (0, 0, 0, 0, 0, 0)

    def decode_cell(self, pattern: tuple[int, ...]) -> tuple[str, float]:
        """
        Decode a single Braille cell pattern to a character.

        Args:
            pattern: 6-element tuple of 0/1 dot values.

        Returns:
            Tuple of (character_string, confidence_float).
            confidence = 1.0 for exact match, 0.75 for 1-dot error,
            0.5 for 2-dot error, 0.0 if unresolvable.
        """
        pattern = tuple(int(p) for p in pattern)
        if len(pattern) != 6:
            logger.warning("Invalid pattern length %d, expected 6", len(pattern))
            return ("?", 0.0)

        # Exact lookup first
        if pattern in self.grade1:
            return (self.grade1[pattern], 1.0)

        # Fuzzy fallback
        return self.fuzzy_match(pattern)

    def fuzzy_match(
        self, pattern: tuple[int, ...], threshold: int = 1
    ) -> tuple[str, float]:
        """
        Find closest Grade 1 entry by Hamming distance.

        Args:
            pattern: 6-element dot pattern.
            threshold: Maximum Hamming distance to accept (1 or 2).

        Returns:
            Best (character, confidence) pair, or ('?', 0.0) if none found.
        """
        best_char = "?"
        best_dist = 999
        best_confidence = 0.0

        for key, char in self.grade1.items():
            dist = sum(a != b for a, b in zip(pattern, key))
            if dist < best_dist:
                best_dist = dist
                best_char = char

        if best_dist == 0:
            best_confidence = 1.0
        elif best_dist == 1:
            best_confidence = 0.75
        elif best_dist == 2 and threshold >= 2:
            best_confidence = 0.5
        else:
            return ("?", 0.0)

        logger.debug(
            "Fuzzy match: pattern=%s → '%s' dist=%d conf=%.2f",
            pattern,
            best_char,
            best_dist,
            best_confidence,
        )
        return (best_char, best_confidence)

    def decode(self, cells: list[dict]) -> str:
        """
        Wrapper to decode cells and return only the text string.

        Args:
            cells: Ordered list of cell dicts from segmenter.

        Returns:
            Decoded text string.
        """
        return self.decode_sequence(cells)[0]

    def decode_sequence(
        self, cells: list[dict]
    ) -> tuple[str, list[float]]:
        """
        Decode an ordered list of Braille cell dicts into text.

        Each cell dict must contain a 'pattern' key with a 6-element
        list/tuple of 0/1 values. Additional keys (x, y, bbox, etc.)
        are ignored.

        Args:
            cells: Ordered list of cell dicts from segmenter.

        Returns:
            Tuple of (decoded_text, per_character_confidence_list).
        """
        text_parts: list[str] = []
        confidences: list[float] = []

        capitalize_next = False
        number_mode = False


        def is_standalone(i_idx: int) -> bool:
            # Find contiguous non-space segment boundaries
            start = i_idx
            while start > 0:
                p = tuple(int(val) for val in cells[start - 1].get("pattern", [0,0,0,0,0,0]))
                if p == (0, 0, 0, 0, 0, 0):
                    break
                start -= 1
            
            end = i_idx
            while end < len(cells) - 1:
                p = tuple(int(val) for val in cells[end + 1].get("pattern", [0,0,0,0,0,0]))
                if p == (0, 0, 0, 0, 0, 0):
                    break
                end += 1
            
            # Count content-bearing cells in this segment
            content_count = 0
            for i in range(start, end + 1):
                p = tuple(int(val) for val in cells[i].get("pattern", [0,0,0,0,0,0]))
                if p == (0, 0, 0, 0, 0, 0):
                    continue
                char_val, _ = self.decode_cell(p)
                if char_val not in ["[CAP]", "[NUM]", "[ITER]"]:
                    content_count += 1
            
            return content_count == 1

        for idx, cell in enumerate(cells):
            raw_pattern = cell.get("pattern", [0, 0, 0, 0, 0, 0])
            pattern = tuple(int(p) for p in raw_pattern)

            # Explicit blank cell detection FIRST
            if self.detect_blank_cell(pattern):
                text_parts.append(" ")
                confidences.append(1.0)
                number_mode = False  # space exits number mode
                continue

            # ── Step A: Number mode ─────────────────────────────────
            if number_mode:
                if pattern in self.number_table:
                    digit = self.number_table[pattern]
                    text_parts.append(digit)
                    confidences.append(1.0)
                    continue
                elif pattern == (0, 0, 0, 0, 0, 0): # space pattern
                    number_mode = False  # space exits number mode
                else:
                    number_mode = False  # any non-digit/non-space exits number mode

            # ── Step B: Exact Grade 1 lookup ────────────────────────
            in_grade1 = pattern in self.grade1
            if in_grade1:
                char, conf = self.grade1[pattern], 1.0
            else:
                char, conf = "?", 0.0  # will resolve in later steps

            if char == "[CAP]":
                capitalize_next = True
                logger.debug("Cell %d: Capital indicator", idx)
                continue

            if char == "[NUM]":
                number_mode = True
                logger.debug("Cell %d: Number indicator", idx)
                continue

            if char == "[ITER]":
                logger.debug("Cell %d: Iteration sign, skipping", idx)
                continue

            # ── Step C: Grade 2 whole-word contractions ─────────────
            # These have patterns NOT in Grade 1, so they only apply when
            # exact Grade 1 lookup failed. Standalone check ensures they
            # only fire when the cell is the single content cell in a word.
            if not number_mode and not in_grade1:
                # Check that previous and next cells are spaces (or start/end of sequence, allowing indicators on the left)
                prev_ok = False
                if idx == 0:
                    prev_ok = True
                else:
                    prev_pat = tuple(int(p) for p in cells[idx - 1].get("pattern", [0, 0, 0, 0, 0, 0]))
                    if prev_pat == (0, 0, 0, 0, 0, 0):
                        prev_ok = True
                    elif prev_pat == (0, 0, 0, 0, 0, 1): # [CAP]
                        if idx - 1 == 0:
                            prev_ok = True
                        else:
                            prev_prev_pat = tuple(int(p) for p in cells[idx - 2].get("pattern", [0, 0, 0, 0, 0, 0]))
                            if prev_prev_pat == (0, 0, 0, 0, 0, 0):
                                prev_ok = True

                next_ok = False
                if idx == len(cells) - 1:
                    next_ok = True
                else:
                    next_pat = tuple(int(p) for p in cells[idx + 1].get("pattern", [0, 0, 0, 0, 0, 0]))
                    if next_pat == (0, 0, 0, 0, 0, 0):
                        next_ok = True

                if pattern in self.grade2_single and prev_ok and next_ok and is_standalone(idx):
                    word = self.grade2_single[pattern]
                    if capitalize_next:
                        word = word.capitalize()
                        capitalize_next = False
                    text_parts.append(word)
                    confidences.append(cell.get("confidence", 0.85))
                    logger.debug("Cell %d: Grade2 whole-word '%s'", idx, word)
                    continue

                # ── Step D: Grade 2 affixes ──────────────────────────
                if pattern in self.grade2_affixes:
                    affix = self.grade2_affixes[pattern]
                    if capitalize_next:
                        affix = affix.capitalize()
                        capitalize_next = False
                    text_parts.append(affix)
                    confidences.append(cell.get("confidence", 0.85))
                    logger.debug("Cell %d: Grade2 affix '%s'", idx, affix)
                    continue

                # ── Step E: Fuzzy Grade 1 fallback ──────────────────
                char, conf = self.fuzzy_match(pattern)

            # ── Step F: Normal Grade 1 character output ──────────────
            if capitalize_next and char.isalpha():
                char = char.upper()
                capitalize_next = False

            text_parts.append(char)
            confidences.append(conf)

        decoded = "".join(text_parts)
        logger.info(
            "Decoded %d cells → %d characters. Avg conf=%.2f",
            len(cells),
            len(decoded),
            sum(confidences) / max(len(confidences), 1),
        )
        return decoded, confidences

    def decode_with_stats(self, cells: list[dict]) -> dict:
        """
        Decode cells and return text plus detailed statistics.

        Args:
            cells: List of cell dicts from segmenter.

        Returns:
            Dict containing decoded text, confidence metrics, and unknown positions.
        """
        text, confidences = self.decode_sequence(cells)

        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
        else:
            avg_confidence = 0.0
            min_confidence = 0.0

        unknown_indices = [i for i, c in enumerate(confidences) if c == 0.0]
        unknown_count = len(unknown_indices)

        stats = {
            "decoded_text": text,
            "character_count": len(text),
            "cell_count": len(cells),
            "confidences": confidences,
            "avg_confidence": round(avg_confidence, 4),
            "min_confidence": round(min_confidence, 4),
            "unknown_count": unknown_count,
            "unknown_indices": unknown_indices,
            "quality": self._quality_label(avg_confidence),
        }
        logger.info("decode_with_stats: %s", stats)
        return stats

    def decode_from_predictions(
        self, predictions: list[dict]
    ) -> tuple[str, list[float]]:
        """
        Decode a sequence of classifier predictions directly into text.

        Takes the output of ``CellClassifier.predict_batch()`` — a list of
        ``{char, confidence, pattern, ...}`` dicts — and applies capital/number-mode
        modifier logic, producing the same output format as ``decode_sequence``.

        This method intentionally mirrors the state machine in ``decode_sequence``
        so that Grade-1 indicator handling is consistent regardless of whether
        the input came from the dot-pattern table or from the neural classifier.

        Args:
            predictions: List of dicts, each with at minimum:
                - ``char``       (str)   — predicted character or indicator token
                - ``confidence`` (float) — classifier confidence (0.0–1.0)
                - ``pattern``    (tuple) — optional original 6-dot pattern (tuple of 0/1)

        Returns:
            Tuple of (decoded_text, per_character_confidence_list).
        """
        text_parts: list[str] = []
        confidences: list[float] = []

        capitalize_next = False
        number_mode = False

        def is_standalone_pred(i_idx: int) -> bool:
            # Find contiguous non-space segment boundaries
            start = i_idx
            while start > 0:
                p_char = predictions[start - 1].get("char", " ")
                p_pat = predictions[start - 1].get("pattern")
                if p_char == " " or p_pat == (0, 0, 0, 0, 0, 0):
                    break
                start -= 1
            
            end = i_idx
            while end < len(predictions) - 1:
                p_char = predictions[end + 1].get("char", " ")
                p_pat = predictions[end + 1].get("pattern")
                if p_char == " " or p_pat == (0, 0, 0, 0, 0, 0):
                    break
                end += 1
            
            # Count content-bearing cells in this segment
            content_count = 0
            for i in range(start, end + 1):
                p_char = predictions[i].get("char", "")
                p_pat = predictions[i].get("pattern")
                if p_pat:
                    p_char_val, _ = self.decode_cell(p_pat)
                else:
                    p_char_val = p_char

                if p_char_val not in ["[CAP]", "[NUM]", "[ITER]", " ", ""]:
                    content_count += 1
            
            return content_count == 1

        for idx, pred in enumerate(predictions):
            char: str = pred.get("char", "?")
            conf: float = float(pred.get("confidence", 0.0))
            raw_pattern = pred.get("pattern")
            pattern = tuple(int(p) for p in raw_pattern) if raw_pattern else None

            # Space cell handling: immediately append and continue, exiting number mode
            if char == " " or (pattern is not None and pattern == (0, 0, 0, 0, 0, 0)):
                text_parts.append(" ")
                confidences.append(conf)
                number_mode = False
                continue

            # ── Capital indicator ────────────────────────────────────────
            if char == "[CAP]":
                capitalize_next = True
                logger.debug("decode_from_predictions: [CAP] indicator")
                continue

            # ── Number indicator ─────────────────────────────────────────
            if char == "[NUM]":
                number_mode = True
                logger.debug("decode_from_predictions: [NUM] indicator")
                continue

            # ── Number mode: digits 0-9 pass through; anything else exits ─
            if number_mode:
                if char.isdigit():
                    text_parts.append(char)
                    confidences.append(conf)
                    continue
                elif char.isalpha():
                    # Classifier may output 'a'-'j'; map to digits via number table
                    letter_to_digit: dict[str, str] = {
                        "a": "1", "b": "2", "c": "3", "d": "4", "e": "5",
                        "f": "6", "g": "7", "h": "8", "i": "9", "j": "0",
                    }
                    mapped = letter_to_digit.get(char.lower())
                    if mapped:
                        text_parts.append(mapped)
                        confidences.append(conf)
                        continue
                    else:
                        number_mode = False  # non a-j letter exits number mode
                else:
                    number_mode = False

            # ── Step C: Grade 2 whole-word contractions ─────────────
            if not number_mode and pattern is not None:
                # Check that previous and next cells are spaces (or start/end of sequence)
                prev_ok = False
                if idx == 0:
                    prev_ok = True
                else:
                    prev_pat = predictions[idx - 1].get("pattern")
                    prev_char = predictions[idx - 1].get("char")
                    if prev_char == " " or prev_pat == (0, 0, 0, 0, 0, 0):
                        prev_ok = True
                    elif prev_char == "[CAP]" or prev_pat == (0, 0, 0, 0, 0, 1): # [CAP]
                        if idx - 1 == 0:
                            prev_ok = True
                        else:
                            prev_prev_pat = predictions[idx - 2].get("pattern")
                            prev_prev_char = predictions[idx - 2].get("char")
                            if prev_prev_char == " " or prev_prev_pat == (0, 0, 0, 0, 0, 0):
                                prev_ok = True

                next_ok = False
                if idx == len(predictions) - 1:
                    next_ok = True
                else:
                    next_pat = predictions[idx + 1].get("pattern")
                    next_char = predictions[idx + 1].get("char")
                    if next_char == " " or next_pat == (0, 0, 0, 0, 0, 0):
                        next_ok = True

                if pattern in self.grade2_single and prev_ok and next_ok and is_standalone_pred(idx):
                    word = self.grade2_single[pattern]
                    if capitalize_next:
                        word = word.capitalize()
                        capitalize_next = False
                    text_parts.append(word)
                    confidences.append(conf)
                    logger.debug("decode_from_predictions: Grade2 whole-word '%s'", word)
                    continue

                # ── Step D: Grade 2 affixes ──────────────────────────
                if pattern in self.grade2_affixes:
                    affix = self.grade2_affixes[pattern]
                    if capitalize_next:
                        affix = affix.capitalize()
                        capitalize_next = False
                    text_parts.append(affix)
                    confidences.append(conf)
                    logger.debug("decode_from_predictions: Grade2 affix '%s'", affix)
                    continue

            # ── Map digits back to letters if not in number mode ─────────
            if not number_mode and char.isdigit():
                digit_to_letter: dict[str, str] = {
                    "1": "a", "2": "b", "3": "c", "4": "d", "5": "e",
                    "6": "f", "7": "g", "8": "h", "9": "i", "0": "j",
                }
                char = digit_to_letter.get(char, char)

            # ── Capitalise ───────────────────────────────────────────────
            if capitalize_next and char.isalpha():
                char = char.upper()
                capitalize_next = False

            text_parts.append(char)
            confidences.append(conf)

        decoded = "".join(text_parts)
        logger.info(
            "decode_from_predictions: %d predictions → %d characters. Avg conf=%.2f",
            len(predictions),
            len(decoded),
            sum(confidences) / max(len(confidences), 1),
        )
        return decoded, confidences

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _quality_label(self, avg_conf: float) -> str:
        """Map average confidence to a human-readable quality label."""
        if avg_conf >= 0.85:
            return "excellent"
        if avg_conf >= 0.65:
            return "good"
        if avg_conf >= 0.45:
            return "fair"
        return "poor"

    def pattern_to_dots(self, pattern: tuple[int, ...]) -> str:
        """
        Return a human-readable dot string e.g. '1-4-5' for pattern (1,0,0,1,1,0).

        Args:
            pattern: 6-element dot pattern.

        Returns:
            String like '1-4-5' indicating which dots are raised.
        """
        dot_labels = [str(i + 1) for i, v in enumerate(pattern) if v]
        return "-".join(dot_labels) if dot_labels else "blank"

    def make_cell(self, pattern: Union[list, tuple], **kwargs) -> dict:
        """
        Helper to construct a cell dict compatible with decode_sequence.

        Args:
            pattern: 6-element dot pattern.
            **kwargs: Optional extra fields (x, y, confidence, bbox, etc.).

        Returns:
            Cell dict ready for decode_sequence.
        """
        return {"pattern": tuple(int(p) for p in pattern), "confidence": kwargs.get("confidence", 1.0), **kwargs}


# ─────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    decoder = BrailleDecoder()

    print("\n" + "=" * 50)
    print("  BrailleDecoder Smoke Test")
    print("=" * 50)

    # Test all 26 letters
    letter_patterns = [
        ((1, 0, 0, 0, 0, 0), "a"),
        ((1, 1, 0, 0, 0, 0), "b"),
        ((1, 0, 0, 1, 0, 0), "c"),
        ((1, 0, 0, 1, 1, 0), "d"),
        ((1, 0, 0, 0, 1, 0), "e"),
        ((1, 1, 0, 1, 0, 0), "f"),
        ((1, 1, 0, 1, 1, 0), "g"),
        ((1, 1, 0, 0, 1, 0), "h"),
        ((0, 1, 0, 1, 0, 0), "i"),
        ((0, 1, 0, 1, 1, 0), "j"),
        ((1, 0, 1, 0, 0, 0), "k"),
        ((1, 1, 1, 0, 0, 0), "l"),
        ((1, 0, 1, 1, 0, 0), "m"),
        ((1, 0, 1, 1, 1, 0), "n"),
        ((1, 0, 1, 0, 1, 0), "o"),
        ((1, 1, 1, 1, 0, 0), "p"),
        ((1, 1, 1, 1, 1, 0), "q"),
        ((1, 1, 1, 0, 1, 0), "r"),
        ((0, 1, 1, 1, 0, 0), "s"),
        ((0, 1, 1, 1, 1, 0), "t"),
        ((1, 0, 1, 0, 0, 1), "u"),
        ((1, 1, 1, 0, 0, 1), "v"),
        ((0, 1, 0, 1, 1, 1), "w"),
        ((1, 0, 1, 1, 0, 1), "x"),
        ((1, 0, 1, 1, 1, 1), "y"),
        ((1, 0, 1, 0, 1, 1), "z"),
    ]

    passed = 0
    for pattern, expected in letter_patterns:
        char, conf = decoder.decode_cell(pattern)
        status = "✓" if char == expected else f"✗ (got '{char}')"
        print(f"  {expected}: {status}  conf={conf:.2f}")
        if char == expected:
            passed += 1

    print(f"\n  Letters: {passed}/26 passed")

    # Test number mode
    print("\n--- Number mode ---")
    cells = [
        decoder.make_cell((0, 1, 1, 1, 1, 1)),  # [NUM]
        decoder.make_cell((1, 0, 0, 0, 0, 0)),  # 1
        decoder.make_cell((1, 1, 0, 0, 0, 0)),  # 2
        decoder.make_cell((1, 0, 0, 1, 0, 0)),  # 3
    ]
    text, confs = decoder.decode_sequence(cells)
    print(f"  Decoded: '{text}'  (expected '123')")

    # Test capital indicator
    print("\n--- Capital indicator ---")
    cells2 = [
        decoder.make_cell((0, 0, 0, 0, 0, 1)),  # [CAP]
        decoder.make_cell((1, 0, 0, 0, 0, 0)),  # A (capitalised)
        decoder.make_cell((1, 0, 1, 0, 1, 0)),  # o
    ]
    text2, _ = decoder.decode_sequence(cells2)
    print(f"  Decoded: '{text2}'  (expected 'Ao')")

    # Test fuzzy match
    print("\n--- Fuzzy matching ---")
    # Introduce 1 dot error in 'a' pattern: (1,0,0,0,0,0) → (1,1,0,0,0,0) = 'b'
    wrong = (1, 1, 0, 0, 0, 1)  # 2 bits different from 'b'
    ch, cf = decoder.fuzzy_match(wrong, threshold=2)
    print(f"  Fuzzy (1,1,0,0,0,1) → '{ch}' conf={cf:.2f}")

    # Test decode_with_stats
    print("\n--- decode_with_stats ---")
    stats = decoder.decode_with_stats(cells)
    print(f"  Text: '{stats['decoded_text']}'  avg_conf={stats['avg_confidence']}  quality={stats['quality']}")

    print("\n✅ Smoke test complete.\n")
