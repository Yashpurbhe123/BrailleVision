"""
BrailleVision AI — Pytest suite for BrailleCellSegmenter
Covers spacing estimation accuracy, cell-splitting prevention, and word space injection.
"""

import sys
import os
import pytest
import numpy as np

# Ensure backend root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.segmenter import BrailleCellSegmenter
from core.decoder import BrailleDecoder

@pytest.fixture
def segmenter():
    return BrailleCellSegmenter()

@pytest.fixture
def decoder():
    return BrailleDecoder()

def test_estimate_spacing_exact(segmenter):
    """Test that estimate_spacing accurately retrieves spacing when k=2 is used."""
    # Simulate a regular grid of dots with exact spacing of 25.0
    dot_spacing = 25.0
    dots = []
    # Create two cells with some patterns
    for cell_idx in range(2):
        base_x = 100.0 + cell_idx * 60.0
        base_y = 100.0
        # dot 1 and 2
        dots.append({"x": base_x - dot_spacing * 0.5, "y": base_y, "confidence": 1.0})
        dots.append({"x": base_x - dot_spacing * 0.5, "y": base_y + dot_spacing, "confidence": 1.0})

    spacing = segmenter.estimate_spacing(dots)
    # The estimated spacing must be exactly dot_spacing
    assert abs(spacing["dot_spacing"] - dot_spacing) < 1e-3

def test_segment_hello_world(segmenter, decoder):
    """Test segmenting the phrase 'hello world' and checking for cell splitting and space injection."""
    dot_spacing = 20.0
    cell_gap = 50.0
    test_dots = []

    hello_patterns = [
        (1, 1, 0, 0, 1, 0), # h
        (1, 0, 0, 0, 1, 0), # e
        (1, 1, 1, 0, 0, 0), # l
        (1, 1, 1, 0, 0, 0), # l
        (1, 0, 1, 0, 1, 0), # o
    ]

    world_patterns = [
        (0, 1, 0, 1, 1, 1), # w
        (1, 0, 1, 0, 1, 0), # o
        (1, 1, 1, 0, 1, 0), # r
        (1, 1, 1, 0, 0, 0), # l
        (1, 0, 0, 1, 1, 0), # d
    ]

    def add_dots(base_x, base_y, pattern):
        expected_rel = [
            (-dot_spacing * 0.5, 0.0),             # dot 1
            (-dot_spacing * 0.5, dot_spacing),       # dot 2
            (-dot_spacing * 0.5, dot_spacing * 2.0), # dot 3
            (dot_spacing * 0.5, 0.0),              # dot 4
            (dot_spacing * 0.5, dot_spacing),       # dot 5
            (dot_spacing * 0.5, dot_spacing * 2.0), # dot 6
        ]
        for idx, v in enumerate(pattern):
            if v:
                dx, dy = expected_rel[idx]
                test_dots.append({
                    "x": base_x + dx,
                    "y": base_y + dy,
                    "confidence": 1.0,
                    "source": "yolo"
                })

    # Add 'hello' dots (cells 0-4)
    base_y = 100.0
    for col_idx, pattern in enumerate(hello_patterns):
        base_x = 100.0 + col_idx * cell_gap
        add_dots(base_x, base_y, pattern)

    # Add 'world' dots (cells 6-10) with cell 5 being a space gap
    for col_idx, pattern in enumerate(world_patterns):
        base_x = 100.0 + (col_idx + 6) * cell_gap
        add_dots(base_x, base_y, pattern)

    cells = segmenter.segment(test_dots)

    # Should detect 11 cells total (5 for hello, 1 for space, 5 for world)
    assert len(cells) == 11
    
    # 6th cell (index 5) must be a space
    assert cells[5]["pattern"] == (0, 0, 0, 0, 0, 0)

    # Decode and verify the result is exactly "hello world"
    text, _ = decoder.decode_sequence(cells)
    assert text == "hello world"

def test_segmenter_duplicate_removal(segmenter, decoder):
    """Test that duplicate cells (close in coordinate space) are removed and sorted correctly."""
    # We will simulate dots that spell "CAT".
    # C = (1, 0, 0, 1, 0, 0) -> dots 1, 4
    # A = (1, 0, 0, 0, 0, 0) -> dot 1
    # T = (0, 1, 1, 1, 1, 0) -> dots 2, 3, 4, 5
    
    dot_spacing = 20.0
    test_dots = []
    
    # helper to add cell dots
    def add_cell(base_x, base_y, pattern):
        expected_rel = [
            (-dot_spacing * 0.5, 0.0),             # dot 1
            (-dot_spacing * 0.5, dot_spacing),       # dot 2
            (-dot_spacing * 0.5, dot_spacing * 2.0), # dot 3
            (dot_spacing * 0.5, 0.0),              # dot 4
            (dot_spacing * 0.5, dot_spacing),       # dot 5
            (dot_spacing * 0.5, dot_spacing * 2.0), # dot 6
        ]
        for idx, v in enumerate(pattern):
            if v:
                dx, dy = expected_rel[idx]
                test_dots.append({
                    "x": base_x + dx,
                    "y": base_y + dy,
                    "confidence": 1.0,
                    "source": "yolo"
                })

    # C
    add_cell(50.0, 100.0, (1, 0, 0, 1, 0, 0))
    # Duplicate C (slightly shifted, within 10px / 15px)
    add_cell(52.0, 101.0, (1, 0, 0, 1, 0, 0))
    
    # A
    add_cell(100.0, 100.0, (1, 0, 0, 0, 0, 0))
    # Duplicate A (slightly shifted, within 10px / 15px)
    add_cell(101.0, 99.0, (1, 0, 0, 0, 0, 0))
    
    # T
    add_cell(150.0, 100.0, (0, 1, 1, 1, 1, 0))
    
    cells = segmenter.segment(test_dots)
    
    # Should only return 3 cells after deduplication (C, A, T)
    assert len(cells) == 3
    
    # Let's decode them
    text, _ = decoder.decode_sequence(cells)
    assert text.lower() == "cat"
