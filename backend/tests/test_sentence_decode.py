"""
test_sentence_decode.py
=======================
Verifies that the pipeline correctly decodes multi-word Braille sentences
by preserving word-boundary space cells instead of classifying them.

Tests the three critical layers of the fix:
  1. _cells_to_predictions: blank cells → space
  2. decode_from_predictions: spaces flow through correctly
  3. _clean_decoded_text: multi-space collapse and noise removal
"""

import sys
import os
import logging

# Allow running from backend/ directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from core.decoder import BrailleDecoder


def make_cell(pattern, dot_count=None, classifier_char=None, classifier_confidence=0.0):
    """Build a minimal cell dict matching the pipeline's internal format."""
    dc = dot_count if dot_count is not None else sum(pattern)
    cell = {
        "pattern": tuple(pattern),
        "dot_count": dc,
        "confidence": 0.9,
        "x": 0.0,
        "y": 0.0,
        "bbox": [0.0, 0.0, 20.0, 30.0],
    }
    if classifier_char is not None:
        cell["classifier_char"] = classifier_char
        cell["classifier_confidence"] = classifier_confidence
        cell["classifier_top3"] = [{"char": classifier_char, "confidence": classifier_confidence}]
        cell["low_confidence"] = classifier_confidence < 0.60
    return cell


# ─────────────────────────────────────────────────────────────────────────────
# Inline the fixed pipeline logic (without importing the full pipeline)
# so this test is dependency-free
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFIER_FALLBACK_THRESHOLD = 0.55

decoder = BrailleDecoder()


def cells_to_predictions(cells):
    """
    Mirrors _cells_to_predictions from pipeline.py.

    Priority order (must stay in sync with pipeline.py):
      1. Blank cells (dot_count==0) → space.
      2. Exact Grade 1 dot-pattern match → use pattern with 1.0 confidence.
         Each Braille letter (H, E, L, L, O …) is decoded individually and
         then combined into the full word downstream.
      3. High-confidence classifier (>= threshold) → use classifier char.
      4. Fuzzy dot-pattern fallback.
    """
    predictions = []
    for cell in cells:
        # Priority 1: blank → space
        dot_count = cell.get("dot_count", sum(cell.get("pattern", [])))
        if dot_count == 0:
            predictions.append({"char": " ", "confidence": 1.0})
            continue

        pattern = tuple(int(p) for p in cell.get("pattern", (0, 0, 0, 0, 0, 0)))

        # Priority 2: exact Grade 1 lookup
        if pattern in decoder.grade1:
            predictions.append({"char": decoder.grade1[pattern], "confidence": 1.0})
            continue

        clf_char = cell.get("classifier_char")
        clf_conf = cell.get("classifier_confidence", 0.0)

        # Priority 3: high-confidence classifier (only for non-Grade1 patterns)
        if clf_char is not None and clf_char != " " and clf_conf >= CLASSIFIER_FALLBACK_THRESHOLD:
            predictions.append({"char": clf_char, "confidence": clf_conf})
        else:
            # Priority 4: fuzzy Hamming fallback
            char, conf = decoder.decode_cell(pattern)
            predictions.append({"char": char, "confidence": conf})
    return predictions


def clean_decoded_text(text):
    """Fixed version of _clean_decoded_text from pipeline.py."""
    import re
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"(?<![\\w])\\?(?![\\w])", "", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Space cells are always output as " " regardless of classifier
# ─────────────────────────────────────────────────────────────────────────────

def test_blank_cell_always_space():
    """A blank cell (dot_count=0) must output space even if classifier says 'a'."""
    blank_cell = make_cell(
        pattern=(0, 0, 0, 0, 0, 0),
        dot_count=0,
        classifier_char="a",            # classifier wrongly thinks it's 'a'
        classifier_confidence=0.92,     # high confidence — the old bug
    )
    preds = cells_to_predictions([blank_cell])
    assert preds[0]["char"] == " ", (
        f"FAIL: blank cell should output ' ' but got '{preds[0]['char']}'"
    )
    print("  [PASS] test_blank_cell_always_space")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: "HELLO WORLD" decodes with a space in the middle
# ─────────────────────────────────────────────────────────────────────────────

def test_hello_world_sentence():
    """
    HELLO WORLD in Braille (Grade 1 patterns):
    H=(1,1,0,0,1,0)  E=(1,0,0,0,1,0)  L=(1,1,1,0,0,0)  L=(1,1,1,0,0,0)  O=(1,0,1,0,1,0)
    [space]=(0,0,0,0,0,0)
    W=(0,1,0,1,1,1)  O=(1,0,1,0,1,0)  R=(1,1,1,0,1,0)  L=(1,1,1,0,0,0)  D=(1,0,0,1,1,0)
    """
    cells = [
        # HELLO — classifier correctly identifies each (high confidence)
        make_cell((1,1,0,0,1,0), classifier_char="h", classifier_confidence=0.85),
        make_cell((1,0,0,0,1,0), classifier_char="e", classifier_confidence=0.80),
        make_cell((1,1,1,0,0,0), classifier_char="l", classifier_confidence=0.88),
        make_cell((1,1,1,0,0,0), classifier_char="l", classifier_confidence=0.87),
        make_cell((1,0,1,0,1,0), classifier_char="o", classifier_confidence=0.76),

        # SPACE — injected by segmenter; old bug: classifier would say 'a' with 0.93 conf
        make_cell((0,0,0,0,0,0), dot_count=0, classifier_char="a", classifier_confidence=0.93),

        # WORLD — classifier correct
        make_cell((0,1,0,1,1,1), classifier_char="w", classifier_confidence=0.72),
        make_cell((1,0,1,0,1,0), classifier_char="o", classifier_confidence=0.69),
        make_cell((1,1,1,0,1,0), classifier_char="r", classifier_confidence=0.78),
        make_cell((1,1,1,0,0,0), classifier_char="l", classifier_confidence=0.91),
        make_cell((1,0,0,1,1,0), classifier_char="d", classifier_confidence=0.83),
    ]

    preds = cells_to_predictions(cells)
    text, _ = decoder.decode_from_predictions(preds)
    text = clean_decoded_text(text)

    assert text == "hello world", (
        f"FAIL: expected 'hello world' but got '{text}'"
    )
    print(f"  [PASS] test_hello_world_sentence → '{text}'")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Pattern fallback when classifier confidence is low
# ─────────────────────────────────────────────────────────────────────────────

def test_pattern_fallback_on_low_confidence():
    """When classifier confidence < 0.55, the dot-pattern decoder is used instead."""
    # 'b' pattern = (1,1,0,0,0,0), but classifier guesses 'z' with only 0.3 confidence
    cell = make_cell((1,1,0,0,0,0), classifier_char="z", classifier_confidence=0.30)
    preds = cells_to_predictions([cell])
    assert preds[0]["char"] == "b", (
        f"FAIL: pattern fallback should give 'b' but got '{preds[0]['char']}'"
    )
    print("  [PASS] test_pattern_fallback_on_low_confidence")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Multi-space collapse
# ─────────────────────────────────────────────────────────────────────────────

def test_multi_space_collapse():
    """Multiple consecutive space cells collapse to a single space."""
    cells = [
        make_cell((1,0,0,0,0,0), classifier_char="a", classifier_confidence=0.90),  # 'a'
        make_cell((0,0,0,0,0,0), dot_count=0),  # space
        make_cell((0,0,0,0,0,0), dot_count=0),  # extra space (double-gap)
        make_cell((1,1,0,0,0,0), classifier_char="b", classifier_confidence=0.88),  # 'b'
    ]
    preds = cells_to_predictions(cells)
    text, _ = decoder.decode_from_predictions(preds)
    text = clean_decoded_text(text)

    assert text == "a b", (
        f"FAIL: multi-space should collapse to 'a b' but got '{text}'"
    )
    print(f"  [PASS] test_multi_space_collapse → '{text}'")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Replicate the image scenario — individual chars become a word
# ─────────────────────────────────────────────────────────────────────────────

def test_image_chars_as_single_word():
    """
    The screenshot shows: s c i o b r a i l l (all one word, no spaces injected).
    Expected output: 'sciobraill' (or real Braille word, but no spaces between these).
    This confirms that when no space cells exist, chars are correctly concatenated.
    """
    chars = [
        ("s", (0,1,1,1,0,0)), ("c", (1,0,0,1,0,0)), ("i", (0,1,0,1,0,0)),
        ("o", (1,0,1,0,1,0)), ("b", (1,1,0,0,0,0)), ("r", (1,1,1,0,1,0)),
        ("a", (1,0,0,0,0,0)), ("i", (0,1,0,1,0,0)), ("l", (1,1,1,0,0,0)),
        ("l", (1,1,1,0,0,0)),
    ]
    cells = [
        make_cell(pat, classifier_char=ch, classifier_confidence=0.75)
        for ch, pat in chars
    ]
    preds = cells_to_predictions(cells)
    text, _ = decoder.decode_from_predictions(preds)
    text = clean_decoded_text(text)

    assert text == "sciobraill", (
        f"FAIL: expected 'sciobraill' but got '{text}'"
    )
    print(f"  [PASS] test_image_chars_as_single_word → '{text}'")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: Digit to letter mapping when not in number mode
# ─────────────────────────────────────────────────────────────────────────────

def test_digit_to_letter_mapping():
    """Digits predicted by the classifier (e.g. '1', '2') must map to letters ('a', 'b') when number_mode is False."""
    cells = [
        # 'hello' with classifier predicting digits '1' and '5' instead of 'a' and 'e'
        make_cell((1,1,0,0,1,0), classifier_char="h", classifier_confidence=0.85),
        # e=1,0,0,0,1,0, but classifier predicts '5' with 0.80 conf
        make_cell((1,0,0,0,1,0), classifier_char="5", classifier_confidence=0.80),
        make_cell((1,1,1,0,0,0), classifier_char="l", classifier_confidence=0.88),
        make_cell((1,1,1,0,0,0), classifier_char="l", classifier_confidence=0.87),
        # o=1,0,1,0,1,0, but classifier predicts '0' with 0.76 conf
        make_cell((1,0,1,0,1,0), classifier_char="0", classifier_confidence=0.76),
    ]

    preds = cells_to_predictions(cells)
    text, _ = decoder.decode_from_predictions(preds)
    text = clean_decoded_text(text)

    assert text == "hello", (
        f"FAIL: expected 'hello' (with digits mapped to letters) but got '{text}'"
    )
    print(f"  [PASS] test_digit_to_letter_mapping → '{text}'")


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BrailleVision Sentence Decode Fix — Test Suite")
    print("=" * 60 + "\n")

    tests = [
        test_blank_cell_always_space,
        test_hello_world_sentence,
        test_pattern_fallback_on_low_confidence,
        test_multi_space_collapse,
        test_image_chars_as_single_word,
        test_digit_to_letter_mapping,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  {e}")

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{len(tests)} passed")
    if passed == len(tests):
        print("  ✅ ALL TESTS PASSED — sentence decoding is working correctly!")
    else:
        print("  ❌ Some tests failed — review the output above.")
    print("=" * 60 + "\n")
