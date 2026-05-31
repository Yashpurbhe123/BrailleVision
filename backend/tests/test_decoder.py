"""
═══════════════════════════════════
📄 FILE 40/42: backend/tests/test_decoder.py
═══════════════════════════════════

BrailleVision AI — Pytest suite for BrailleDecoder
Covers Grade 1, Grade 2 contractions, number mode, capitalization rules,
fuzzy Hamming-distance corrections, sequence structures, and stat generation.
"""

import sys
import os
import pytest

# Ensure backend root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.decoder import BrailleDecoder

@pytest.fixture
def decoder():
    return BrailleDecoder()

# ─────────────────────────────────────────────────────────────
# GRADE 1 TESTS
# ─────────────────────────────────────────────────────────────

def test_decode_single_letter_a(decoder):
    """Test decoding letter 'a' with exact match."""
    char, conf = decoder.decode_cell((1, 0, 0, 0, 0, 0))
    assert char == "a"
    assert conf == 1.0

def test_decode_single_letter_z(decoder):
    """Test decoding letter 'z' with exact match."""
    char, conf = decoder.decode_cell((1, 0, 1, 0, 1, 1))
    assert char == "z"
    assert conf == 1.0

def test_decode_invalid_length(decoder):
    """Test handling of invalid pattern length."""
    char, conf = decoder.decode_cell((1, 0, 0))
    assert char == "?"
    assert conf == 0.0

def test_decode_punctuation_period(decoder):
    """Test decoding period punctuation."""
    char, conf = decoder.decode_cell((0, 1, 0, 0, 1, 1))
    assert char == "."
    assert conf == 1.0

def test_decode_punctuation_exclamation(decoder):
    """Test decoding exclamation mark punctuation."""
    char, conf = decoder.decode_cell((0, 1, 1, 0, 1, 0))
    assert char == "!"
    assert conf == 1.0

# ─────────────────────────────────────────────────────────────
# CAPITALIZATION & NUMBER INDICATORS
# ─────────────────────────────────────────────────────────────

def test_capital_indicator(decoder):
    """Test capital indicator capitalized the subsequent letter."""
    cells = [
        decoder.make_cell((0, 0, 0, 0, 0, 1)),  # [CAP]
        decoder.make_cell((1, 0, 0, 0, 0, 0)),  # a
        decoder.make_cell((1, 1, 0, 0, 0, 0)),  # b
    ]
    text, confs = decoder.decode_sequence(cells)
    assert text == "Ab"
    assert len(confs) == 2  # CAP cell is consumed

def test_multiple_capitals(decoder):
    """Test multiple cap indicators apply to respective letters."""
    cells = [
        decoder.make_cell((0, 0, 0, 0, 0, 1)),  # [CAP]
        decoder.make_cell((1, 0, 0, 0, 0, 0)),  # a
        decoder.make_cell((0, 0, 0, 0, 0, 1)),  # [CAP]
        decoder.make_cell((1, 1, 0, 0, 0, 0)),  # b
    ]
    text, _ = decoder.decode_sequence(cells)
    assert text == "AB"

def test_number_indicator(decoder):
    """Test number indicator sets number mode and remaps letters to digits."""
    cells = [
        decoder.make_cell((0, 0, 1, 1, 1, 1)),  # [NUM]
        decoder.make_cell((1, 0, 0, 0, 0, 0)),  # 1 (normally 'a')
        decoder.make_cell((1, 1, 0, 0, 0, 0)),  # 2 (normally 'b')
        decoder.make_cell((0, 1, 0, 1, 1, 0)),  # 0 (normally 'j')
    ]
    text, _ = decoder.decode_sequence(cells)
    assert text == "120"

def test_number_mode_reset_by_space(decoder):
    """Test space character terminates number mode."""
    cells = [
        decoder.make_cell((0, 0, 1, 1, 1, 1)),  # [NUM]
        decoder.make_cell((1, 0, 0, 0, 0, 0)),  # 1
        decoder.make_cell((0, 0, 0, 0, 0, 0)),  # [SPACE]
        decoder.make_cell((1, 0, 0, 0, 0, 0)),  # a (back to letter mode)
    ]
    text, _ = decoder.decode_sequence(cells)
    assert text == "1 a"

# ─────────────────────────────────────────────────────────────
# GRADE 2 CONTRACTIONS & AFFIXES
# ─────────────────────────────────────────────────────────────

def test_grade2_single_word_and(decoder):
    """Test Grade 2 single-cell word contraction 'and'."""
    cells = [decoder.make_cell((1, 1, 1, 1, 0, 1))]  # dots 1-2-3-4-6
    text, _ = decoder.decode_sequence(cells)
    assert text == "and"

def test_grade2_single_word_the(decoder):
    """Test Grade 2 contraction 'the'."""
    cells = [decoder.make_cell((0, 1, 1, 1, 0, 1))]  # dots 2-3-4-6
    text, _ = decoder.decode_sequence(cells)
    assert text == "the"

def test_grade2_affix_ch(decoder):
    """Test Grade 2 affix contraction 'ch'."""
    cells = [decoder.make_cell((1, 0, 0, 0, 1, 1))]
    text, _ = decoder.decode_sequence(cells)
    assert text == "ch"

def test_grade2_affix_st(decoder):
    """Test Grade 2 affix contraction 'st'."""
    cells = [decoder.make_cell((0, 0, 0, 1, 1, 1))]
    text, _ = decoder.decode_sequence(cells)
    assert text == "st"

# ─────────────────────────────────────────────────────────────
# FUZZY HAMMING MATCHING
# ─────────────────────────────────────────────────────────────

def test_fuzzy_hamming_distance_one(decoder):
    """Test correcting a 1-dot error in letter 'c' pattern."""
    # 'c' is (1,0,0,1,0,0). Let's test (1,0,0,1,0,1) which has dist 1 from 'c' and is not in Grade 1 table.
    wrong_pattern = (1, 0, 0, 1, 0, 1)
    char, conf = decoder.decode_cell(wrong_pattern)
    assert char == "c"
    assert conf == 0.75

def test_fuzzy_hamming_distance_two(decoder):
    """Test correcting a 2-dot error in letter 'a' pattern with isolated table."""
    decoder.grade1 = {(1, 0, 0, 0, 0, 0): "a"}
    # (1, 0, 0, 0, 1, 1) is distance 2 from 'a'
    char, conf = decoder.fuzzy_match((1, 0, 0, 0, 1, 1), threshold=2)
    assert char == "a"
    assert conf == 0.5

def test_fuzzy_hamming_distance_exceeded(decoder):
    """Test that matches with distance > threshold return unknown."""
    decoder.grade1 = {(1, 0, 0, 0, 0, 0): "a"}
    # (1, 0, 0, 0, 1, 1) is distance 2 from 'a'. Threshold is 1, so it should return unknown '?'
    char, conf = decoder.fuzzy_match((1, 0, 0, 0, 1, 1), threshold=1)
    assert char == "?"
    assert conf == 0.0

# ─────────────────────────────────────────────────────────────
# SEQUENCE & DECODE WITH STATS
# ─────────────────────────────────────────────────────────────

def test_decode_sequence_empty(decoder):
    """Test decoding an empty cell list."""
    text, confs = decoder.decode_sequence([])
    assert text == ""
    assert confs == []

def test_decode_sequence_simple_word(decoder):
    """Test decoding standard sequence word."""
    cells = [
        decoder.make_cell((1, 0, 1, 0, 0, 0)),  # k
        decoder.make_cell((1, 0, 0, 0, 1, 0)),  # e
        decoder.make_cell((1, 1, 1, 0, 1, 0)),  # r
        decoder.make_cell((1, 1, 1, 0, 1, 0)),  # r
    ]
    text, confs = decoder.decode_sequence(cells)
    assert text == "kerr"
    assert len(confs) == 4
    assert all(c == 1.0 for c in confs)

def test_decode_with_stats_excellent_quality(decoder):
    """Test detailed stats generation with excellent quality indicator."""
    cells = [
        decoder.make_cell((1, 0, 0, 0, 0, 0), confidence=0.95),  # a
        decoder.make_cell((1, 1, 0, 0, 0, 0), confidence=0.90),  # b
    ]
    # In sequence context, 'ab' is not standalone so it decodes as letters
    stats = decoder.decode_with_stats(cells)
    assert stats["decoded_text"] == "ab"
    assert stats["cell_count"] == 2
    assert stats["avg_confidence"] == 1.0
    assert stats["quality"] == "excellent"
    assert stats["unknown_count"] == 0

def test_decode_with_stats_poor_quality(decoder):
    """Test stats generation with low quality indicator."""
    decoder.grade1 = {} # clear grade 1 to force bad matches
    decoder.grade2_single = {}
    decoder.grade2_affixes = {}
    stats = decoder.decode_with_stats([{"pattern": (1,1,1,1,1,1)}])
    assert stats["quality"] == "poor"
    assert stats["unknown_count"] == 1
    assert stats["unknown_indices"] == [0]

# ─────────────────────────────────────────────────────────────
# HELPER METHODS
# ─────────────────────────────────────────────────────────────

def test_pattern_to_dots_normal(decoder):
    """Test formatting pattern to dot numbers."""
    assert decoder.pattern_to_dots((1, 0, 0, 1, 1, 0)) == "1-4-5"

def test_pattern_to_dots_blank(decoder):
    """Test formatting blank pattern."""
    assert decoder.pattern_to_dots((0, 0, 0, 0, 0, 0)) == "blank"

def test_make_cell_creation(decoder):
    """Test helper cell creator returns expected dictionary layout."""
    c = decoder.make_cell((1, 1, 0, 0, 0, 0), confidence=0.88, extra="info")
    assert c["pattern"] == (1, 1, 0, 0, 0, 0)
    assert c["confidence"] == 0.88
    assert c["extra"] == "info"
