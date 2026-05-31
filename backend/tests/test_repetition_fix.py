"""
BrailleVision AI — Pytest suite for Repetition Fixes
Covers repetition detection, deduplication, and pipeline retry logic.
"""

import sys
import os
import pytest

# Ensure backend root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.corrector import AIErrorCorrector
from ai.pipeline import BrailleAIPipeline

@pytest.fixture
def corrector():
    return AIErrorCorrector()

def test_detect_repetition():
    """Test the _detect_repetition helper in BrailleAIPipeline."""
    # Repeated words
    assert BrailleAIPipeline._detect_repetition("hello hello hello") is True
    
    # Substring repetition
    assert BrailleAIPipeline._detect_repetition("abcabcabc") is True
    
    # Normal text without repetition
    assert BrailleAIPipeline._detect_repetition("hello world this is braille") is False
    assert BrailleAIPipeline._detect_repetition("the cat sat on the mat") is False

def test_clean_decoded_text_collapsing():
    """Test that _clean_decoded_text collapses repeated patterns."""
    # Word repetitions
    assert BrailleAIPipeline._clean_decoded_text("hello hello hello") == "hello"
    
    # Normal text
    assert BrailleAIPipeline._clean_decoded_text("hello world") == "hello world"

def test_corrector_pre_deduplication(corrector):
    """Test that the corrector deduplicates text before LLM or spell check."""
    # Word repetitions
    res = corrector.correct("hello hello hello")
    assert res["corrected"] == "hello"
    assert res["was_corrected"] is True
    
    # Spell check combined with dedup
    res_spell = corrector.correct("hello hello hello wordl", prefer_llm=False)
    # Deduplicates to "hello wordl", then spellcheck corrects wordl -> world
    assert res_spell["corrected"] == "hello world"
    assert res_spell["was_corrected"] is True
