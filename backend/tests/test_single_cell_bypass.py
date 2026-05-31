"""
BrailleVision AI — Test suite for single-cell bypass and polarity fixes.
Verifies that small squareish images are correctly routed to the single-cell classifier
and that crop/image polarity is correctly handled dynamically.
"""

import sys
import os
import pytest
import numpy as np
import cv2

# Ensure backend root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.pipeline import BrailleAIPipeline
from ai.models.cell_classifier import crop_cell_to_pil

@pytest.fixture
def pipeline():
    return BrailleAIPipeline()

def test_crop_cell_to_pil_dynamic_polarity():
    """Verify crop_cell_to_pil dynamically inverts only when background is light."""
    # 1. Dark background, light dots (mean < 127) -> should not invert
    dark_bg = np.ones((64, 64), dtype=np.uint8) * 20
    # Add a white dot in the center
    cv2.circle(dark_bg, (32, 32), 8, 255, -1)
    
    pil_dark = crop_cell_to_pil(dark_bg, (0, 0, 64, 64), padding=0)
    assert pil_dark is not None
    # Convert back to numpy to check pixel values
    arr_dark = np.array(pil_dark)
    # Background should stay dark (around 20, or rather, the inverted_crop logic was 255 - crop when mean > 127,
    # so since mean <= 127, it stayed dark/original, but wait, crop_cell_to_pil did inverted_crop = 255 - crop previously.
    # Now it dynamically leaves it as-is for dark background).
    assert np.mean(arr_dark) < 127
    assert arr_dark[0, 0] == 20  # preserved original dark background
    
    # 2. Light background, dark dots (mean > 127) -> should invert
    light_bg = np.ones((64, 64), dtype=np.uint8) * 235
    # Add a dark dot in the center
    cv2.circle(light_bg, (32, 32), 8, 40, -1)
    
    pil_light = crop_cell_to_pil(light_bg, (0, 0, 64, 64), padding=0)
    assert pil_light is not None
    arr_light = np.array(pil_light)
    # Background should become dark (255 - 235 = 20) and dot should become light (255 - 40 = 215)
    assert np.mean(arr_light) < 127
    assert arr_light[0, 0] == 20  # inverted to dark background

@pytest.mark.asyncio
async def test_single_cell_image_bypass(pipeline):
    """Verify that a small, squareish image runs through the single-cell bypass."""
    # Create a small 128x128 image with a dark background and white dot (simulating dots 1-2-3 for letter 'l')
    img = np.ones((128, 128, 3), dtype=np.uint8) * 15
    # Draw white dot pattern for letter 'l' (vertical line of dots 1, 2, 3)
    cv2.circle(img, (40, 30), 6, (250, 250, 250), -1)
    cv2.circle(img, (40, 64), 6, (250, 250, 250), -1)
    cv2.circle(img, (40, 98), 6, (250, 250, 250), -1)
    
    _, buf = cv2.imencode(".jpg", img)
    image_bytes = buf.tobytes()
    
    result = await pipeline.process_image(image_bytes, options={"correct": False, "speak": False})
    
    assert result["success"] is True
    assert result["classifier_used"] is True
    assert result["cell_count"] == 1
    # Bounding box should match full image size
    assert result["cells"][0]["bbox"] == [0.0, 0.0, 128.0, 128.0]
    assert len(result["heatmap"]) == 1
    assert result["heatmap"][0]["char"] == result["raw_text"]
