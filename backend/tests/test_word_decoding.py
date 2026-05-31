"""
BrailleVision AI — Test suite for word-level decoding correctness.
Covers single word, spacing, capitalization, numbers, punctuation,
and Grade 2 contraction boundary constraints.
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

def test_single_word_hello(decoder):
    """Test single word: 'hello' (no spaces)"""
    cells = [
        decoder.make_cell((1, 1, 0, 0, 1, 0)),  # h
        decoder.make_cell((1, 0, 0, 0, 1, 0)),  # e
        decoder.make_cell((1, 1, 1, 0, 0, 0)),  # l
        decoder.make_cell((1, 1, 1, 0, 0, 0)),  # l
        decoder.make_cell((1, 0, 1, 0, 1, 0)),  # o
    ]
    text, _ = decoder.decode_sequence(cells)
    assert text == "hello"

def test_two_words_hello_world(decoder):
    """Test two words: 'hello world' (with space)"""
    cells = [
        decoder.make_cell((1, 1, 0, 0, 1, 0)),  # h
        decoder.make_cell((1, 0, 0, 0, 1, 0)),  # e
        decoder.make_cell((1, 1, 1, 0, 0, 0)),  # l
        decoder.make_cell((1, 1, 1, 0, 0, 0)),  # l
        decoder.make_cell((1, 0, 1, 0, 1, 0)),  # o
        decoder.make_cell((0, 0, 0, 0, 0, 0)),  # space
        decoder.make_cell((0, 1, 0, 1, 1, 1)),  # w
        decoder.make_cell((1, 0, 1, 0, 1, 0)),  # o
        decoder.make_cell((1, 1, 1, 0, 1, 0)),  # r
        decoder.make_cell((1, 1, 1, 0, 0, 0)),  # l
        decoder.make_cell((1, 0, 0, 1, 1, 0)),  # d
    ]
    text, _ = decoder.decode_sequence(cells)
    assert text == "hello world"

def test_with_capitals_hello_world(decoder):
    """Test capitalization preservation: 'Hello World'"""
    cells = [
        decoder.make_cell((0, 0, 0, 0, 0, 1)),  # [CAP]
        decoder.make_cell((1, 1, 0, 0, 1, 0)),  # H
        decoder.make_cell((1, 0, 0, 0, 1, 0)),  # e
        decoder.make_cell((1, 1, 1, 0, 0, 0)),  # l
        decoder.make_cell((1, 1, 1, 0, 0, 0)),  # l
        decoder.make_cell((1, 0, 1, 0, 1, 0)),  # o
        decoder.make_cell((0, 0, 0, 0, 0, 0)),  # space
        decoder.make_cell((0, 0, 0, 0, 0, 1)),  # [CAP]
        decoder.make_cell((0, 1, 0, 1, 1, 1)),  # W
        decoder.make_cell((1, 0, 1, 0, 1, 0)),  # o
        decoder.make_cell((1, 1, 1, 0, 1, 0)),  # r
        decoder.make_cell((1, 1, 1, 0, 0, 0)),  # l
        decoder.make_cell((1, 0, 0, 1, 1, 0)),  # d
    ]
    text, _ = decoder.decode_sequence(cells)
    assert text == "Hello World"

def test_with_numbers_123(decoder):
    """Test with numbers: '123'"""
    cells = [
        decoder.make_cell((0, 0, 1, 1, 1, 1)),  # [NUM]
        decoder.make_cell((1, 0, 0, 0, 0, 0)),  # 1
        decoder.make_cell((1, 1, 0, 0, 0, 0)),  # 2
        decoder.make_cell((1, 0, 0, 1, 0, 0)),  # 3
    ]
    text, _ = decoder.decode_sequence(cells)
    assert text == "123"

def test_with_punctuation_hello_period(decoder):
    """Test with punctuation: 'hello.'"""
    cells = [
        decoder.make_cell((1, 1, 0, 0, 1, 0)),  # h
        decoder.make_cell((1, 0, 0, 0, 1, 0)),  # e
        decoder.make_cell((1, 1, 1, 0, 0, 0)),  # l
        decoder.make_cell((1, 1, 1, 0, 0, 0)),  # l
        decoder.make_cell((1, 0, 1, 0, 1, 0)),  # o
        decoder.make_cell((0, 1, 0, 0, 1, 1)),  # .
    ]
    text, _ = decoder.decode_sequence(cells)
    assert text == "hello."

def test_grade2_contract_not_in_words(decoder):
    """Verify that Grade 2 contractions are not applied when part of a longer word."""
    # Pattern (1, 1, 1, 1, 0, 1) is 'and'.
    # If part of 'candy': c=(1,0,0,1,0,0), and=(1,1,1,1,0,1), y=(1,0,1,1,1,1).
    # Since they are not separated by spaces, it should not decode the contraction 'and'.
    cells = [
        decoder.make_cell((1, 0, 0, 1, 0, 0)),  # c
        decoder.make_cell((1, 1, 1, 1, 0, 1)),  # and-pattern (1,1,1,1,0,1)
        decoder.make_cell((1, 0, 1, 1, 1, 1)),  # y
    ]
    text, _ = decoder.decode_sequence(cells)
    # The 'and' contraction should NOT be used. It should fall back to fuzzy match 'p'.
    assert "and" not in text
