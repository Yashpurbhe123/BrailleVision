"""
═══════════════════════════════════
📄 FILE 08/42: backend/ai/pipeline.py
═══════════════════════════════════

BrailleVision AI — Master AI Pipeline Orchestrator
Wires every module together: preprocess → detect → segment →
classify (EfficientNet-B3) → decode → correct → translate → TTS → DB.

NEW in this revision:
  • CellClassifier (EfficientNet-B3) integrated after segmentation.
  • process_live_frame() groups cells into lines, returns lines[] + line_count.
  • ScanResponse enriched with heatmap[] data per cell.
  • Dual-side auto-correction already handled by preprocessor (detect_side +
    mirror_if_back); reflected in side_detected field.
  • Graceful fallback to pattern-lookup decoder when classifier unavailable or
    cell confidence < CLASSIFIER_FALLBACK_THRESHOLD.
"""

from __future__ import annotations

import torch
import asyncio
import base64
import logging
import time
from typing import Optional

import cv2
import numpy as np

from core.preprocess import ImagePreprocessor  # type: ignore
from core.detector import HybridBrailleDetector  # type: ignore
from core.segmenter import BrailleCellSegmenter  # type: ignore
from core.decoder import BrailleDecoder  # type: ignore
from core.corrector import AIErrorCorrector  # type: ignore
from core.translator import BrailleTranslator  # type: ignore
from core.tts_engine import BrailleTTSEngine  # type: ignore
from ai.context_memory import ContextMemory  # type: ignore
from ai.models.cell_classifier import get_classifier, crop_cell_to_pil  # type: ignore

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

LIVE_FRAME_TIMEOUT_MS = 200       # target max ms for live frame processing
ANNOTATED_JPEG_QUALITY = 80       # JPEG quality for base64 annotated image export

# Minimum classifier confidence to trust the model over the pattern decoder.
# Set to 0.55 to align with test suite expectations and ensure high-confidence
# neural overrides, falling back to exact dot-pattern lookup when uncertain.
CLASSIFIER_FALLBACK_THRESHOLD = 0.55

# Dot density sanity gate: max dots allowed per 1000 px² of image area.
# Braille text at any normal density produces well under 1 dot per 500 px².
# If exceeded, the image is noisy / blurry and we clamp before segmenting.
MAX_DOT_DENSITY_PER_KPX2 = 0.8     # dots per 1000 px²
DOT_DENSITY_HARD_CAP = 400          # absolute max dots sent to segmenter

# For live scan: target Braille lines to decode per frame
# Process 2 lines first; extend to 4 if timing budget allows (adaptive).
LIVE_LINES_MIN = 2
LIVE_LINES_MAX = 4
LIVE_BUDGET_EXTEND_MS = 120  # if 2-line decode finishes in <120ms, try 2 more lines


# ─────────────────────────────────────────────────────────────
# PIPELINE CLASS
# ─────────────────────────────────────────────────────────────


class BrailleAIPipeline:
    """
    Master orchestrator that chains all BrailleVision AI modules.

    Single pipeline object is created at application startup and
    reused for all requests (thread-safe for async usage).

    Processing flow:
        image_bytes
            → ImagePreprocessor.full_pipeline()
            → HybridBrailleDetector.detect()
            → BrailleCellSegmenter.segment()
            → CellClassifier.predict_batch()        [NEW — EfficientNet-B3]
            → BrailleDecoder.decode_from_predictions()  [NEW path]
            → AIErrorCorrector.correct()  [optional]
            → BrailleTranslator.translate()  [optional]
            → BrailleTTSEngine.generate_speech_bytes()  [optional]
            → annotated image base64  [optional]
    """

    def __init__(self) -> None:
        """
        Initialise and wire all pipeline modules.

        Each module is lazily ready — heavy models (YOLOv8, EfficientNet-B3)
        load during __init__ so the first request is not penalised.
        """
        logger.info("BrailleAIPipeline: initialising all modules...")

        self.preprocessor = ImagePreprocessor()
        self.detector = HybridBrailleDetector()
        self.segmenter = BrailleCellSegmenter()
        self.decoder = BrailleDecoder()
        self.corrector = AIErrorCorrector()
        self.translator = BrailleTranslator()
        self.tts = BrailleTTSEngine()
        self.memory = ContextMemory()

        # Load EfficientNet-B3 classifier singleton (non-blocking: graceful fallback)
        # Prefers braille_scripted.pt (TorchScript), falls back to best_model.pth
        self.classifier = get_classifier()
        if self.classifier.is_available():
            info = self.classifier.model_info()
            logger.info(
                "BrailleAIPipeline: classifier READY — type=%s, val_acc=%.1f%%, "
                "classes=%d, device=%s, threshold=%.2f",
                info["model_type"],
                info["val_accuracy"],
                info["num_classes"],
                info["device"],
                CLASSIFIER_FALLBACK_THRESHOLD,
            )
        else:
            logger.warning(
                "BrailleAIPipeline: classifier unavailable — using pattern-lookup decoder only."
            )

        self.last_result: Optional[dict] = None

        logger.info("BrailleAIPipeline: all modules ready.")

    # ------------------------------------------------------------------
    # CLASSIFIER INTEGRATION HELPERS
    # ------------------------------------------------------------------

    def _classify_cells(self, cells: list[dict], processed_img: np.ndarray) -> tuple[list[dict], bool]:
        """Classify cells with standard preprocessing."""
        from PIL import Image  # type: ignore
        
        if not self.classifier.is_available():
            return cells, False
        
        crop_indices: list[int] = []
        pil_crops: list = []
        
        logger.info(f"[CLASSIFIER] Processing {len(cells)} cells")
        
        for i, cell in enumerate(cells):
            # Skip blank cells
            if cell.get("dot_count", 0) == 0:
                cell["classifier_char"] = " "
                cell["classifier_confidence"] = 1.0
                continue
            
            # Get cell image
            cell_image = cell.get('image')
            if cell_image is None:
                logger.warning(f"  Cell {i}: No image data")
                continue
            
            # Convert to PIL Image
            if isinstance(cell_image, np.ndarray):
                pil_image = Image.fromarray(cell_image.astype(np.uint8))
            else:
                pil_image = cell_image
            
            # Predict
            prediction = self.classifier.predict_single(pil_image)
            
            cell["classifier_char"] = prediction['char']
            cell["classifier_confidence"] = prediction['confidence']
            
            crop_indices.append(i)
            pil_crops.append(pil_image)
            
            logger.info(f"  Cell {i}: '{prediction['char']}' (conf={prediction['confidence']:.3f})")
        
        return cells, len(crop_indices) > 0

    def _cells_to_predictions(self, cells: list[dict]) -> list[dict]:
        """
        Convert enriched cell dicts to the predictions format expected by
        BrailleDecoder.decode_from_predictions().

        Priority order:
          1. Blank cells (dot_count==0) → always space.
          2. High-confidence neural classifier (conf >= CLASSIFIER_FALLBACK_THRESHOLD)
             → used because classifier reads pixels directly (immune to geometry misalignment).
          3. Exact Grade 1 dot-pattern match → fallback for when model is uncertain.
          4. Fuzzy dot-pattern match → last resort.

        Args:
            cells: Enriched cell dicts (output of _classify_cells).

        Returns:
            List of {char, confidence, pattern} dicts for decode_from_predictions.
        """
        predictions: list[dict] = []
        for cell in cells:
            pattern = tuple(int(p) for p in cell.get("pattern", (0, 0, 0, 0, 0, 0)))

            # ── Priority 1: Blank cells are ALWAYS spaces ─────────────────
            # The segmenter injects cells with pattern (0,0,0,0,0,0) at word
            # boundaries. The classifier has no space class, so handle first.
            dot_count = cell.get("dot_count", sum(pattern))
            if dot_count == 0 or sum(pattern) == 0:
                predictions.append({
                    "char": " ",
                    "confidence": 1.0,
                    "pattern": pattern,
                })
                continue

            clf_char = cell.get("classifier_char")
            clf_conf = cell.get("classifier_confidence", 0.0)

            # ── Priority 2: High-confidence neural classifier ─────────────
            # The classifier reads actual pixel data from the cropped cell image.
            # It is NOT affected by geometric cell-center misalignment that can
            # corrupt the dot-pattern (proven by debug logs: classifier correctly
            # identified 't','a','c' while pattern decoded to 'd','[ITER]','c').
            # Trust the classifier whenever its confidence is above threshold.
            if clf_char is not None and clf_char != " " and clf_conf >= CLASSIFIER_FALLBACK_THRESHOLD:
                predictions.append({
                    "char": clf_char,
                    "confidence": clf_conf,
                    "pattern": pattern,
                })
                continue

            # ── Priority 3: Exact Grade 1 dot-pattern lookup ──────────────
            # Fallback when classifier is unavailable or below threshold.
            if pattern in self.decoder.grade1:
                predictions.append({
                    "char": self.decoder.grade1[pattern],
                    "confidence": 1.0,
                    "pattern": pattern,
                })
                continue

            # ── Priority 4: Fuzzy dot-pattern fallback ────────────────────
            char, conf = self.decoder.decode_cell(pattern)
            predictions.append({
                "char": char,
                "confidence": conf,
                "pattern": pattern,
            })
        return predictions

    # ------------------------------------------------------------------
    # WORD-LEVEL CHARACTER-BY-CHARACTER DECODING  (per handwritten notes)
    # ------------------------------------------------------------------

    def _decode_word_by_characters(
        self,
        cells: list[dict],
        processed_img: np.ndarray,
    ) -> tuple[list[dict], list[dict]]:
        """
        Decode a Braille word by classifying each cell individually.

        Implements the approach from the BrailleVision handwritten design notes:
            Detect full word → separate cells → decode each cell one-by-one
            → combine all decoded characters → final word.

        Why this is better than whole-word classification:
          • The EfficientNet-B3 was trained on *single-cell* images.
          • Feeding it the whole word at once causes confusion (96.8% acc
            is measured on individual cells, not multi-cell strips).
          • Classifying cell-by-cell and concatenating preserves full accuracy
            for any word length.

        Args:
            cells:         Segmented cell dicts from BrailleCellSegmenter.segment().
            processed_img: Full preprocessed image (numpy ndarray).

        Returns:
            Tuple of:
              enriched_cells     – original cells list with classifier_char /
                                   classifier_confidence fields added in-place.
              decoded_characters – list of per-cell result dicts:
                                   {char, confidence, cell_index, source}.
        """
        img_h, img_w = processed_img.shape[:2]
        aspect_ratio = img_w / img_h
        
        # Check if the cells span multiple lines by looking at their Y-coordinates spread
        is_multiline = False
        if cells:
            ys = [c.get("y", 0.0) for c in cells]
            if len(ys) > 1:
                y_spread = max(ys) - min(ys)
                if y_spread > 70.0:
                    is_multiline = True

        # ── Pathway A: Multi-Line Document Grid Segmentation ──────────
        if is_multiline:
            logger.info("[WORD-DECODE] Multiline document detected. Applying high-reliability multi-line grid segmentation.")
            # Group cells into lines
            lines_cells = self._group_cells_into_lines(cells, spacing_ds=20.0)
            logger.info("[WORD-DECODE] Grouped into %d lines.", len(lines_cells))
            
            grid_cells = []
            grid_decoded_chars = []
            
            global_cell_idx = 0
            for line_idx, line in enumerate(lines_cells):
                if not line:
                    continue
                # Find bounding box of this line
                x_min = min(c["bbox"][0] for c in line)
                x_max = max(c["bbox"][2] for c in line)
                y_min = min(c["bbox"][1] for c in line)
                y_max = max(c["bbox"][3] for c in line)
                
                # Add vertical padding
                y_margin = 5
                y_min_padded = max(0, int(y_min - y_margin))
                y_max_padded = min(img_h, int(y_max + y_margin))
                line_h = y_max_padded - y_min_padded
                
                word_width = x_max - x_min
                
                # Estimate expected cell count N for this line
                expected_n = max(1, round(word_width / (line_h * 0.72)))
                logger.info("[WORD-DECODE] Line %d: Grid bounds X=[%d, %d], Y=[%d, %d], expected cells N=%d", 
                            line_idx, x_min, x_max, y_min_padded, y_max_padded, expected_n)
                
                cell_width = word_width / expected_n
                for i in range(expected_n):
                    cx1 = x_min + i * cell_width
                    cx2 = x_min + (i + 1) * cell_width
                    
                    # Crop cell from the horizontal band
                    cell_strip = processed_img[y_min_padded:y_max_padded, int(cx1):int(cx2)]
                    h_c, w_c = cell_strip.shape[:2]
                    if cell_strip.size == 0:
                        continue
                    
                    # Pad to square
                    max_dim = max(h_c, w_c)
                    square_image = np.ones((max_dim, max_dim), dtype=np.uint8) * 255
                    y_offset = (max_dim - h_c) // 2
                    x_offset = (max_dim - w_c) // 2
                    square_image[y_offset:y_offset+h_c, x_offset:x_offset+w_c] = cell_strip
                    
                    # bbox and center coordinates
                    bbox = [round(cx1, 1), float(y_min_padded), round(cx2, 1), float(y_max_padded)]
                    cx = cx1 + cell_width / 2
                    cy = y_min_padded + line_h / 2
                    
                    # Check if this cell crop is blank (contains no dots / dark pixels)
                    is_blank_cell = True
                    if cell_strip.size > 0:
                        # If any pixel in the cell strip is significantly darker than paper background (less than 200)
                        if np.min(cell_strip) < 200:
                            is_blank_cell = False
                            
                    # Predict
                    if is_blank_cell:
                        char = " "
                        confidence = 1.0
                        top3 = [{"char": " ", "confidence": 1.0}]
                        low_conf = False
                    elif self.classifier.is_available():
                        from PIL import Image as PILImage
                        pil_crop = PILImage.fromarray(square_image.astype(np.uint8))
                        
                        # Dynamically invert if light background
                        mean_val = np.mean(square_image)
                        if mean_val > 127:
                            from PIL import ImageOps
                            pil_crop_input = ImageOps.invert(pil_crop)
                        else:
                            pil_crop_input = pil_crop
                            
                        prediction = self.classifier.predict_single(pil_crop_input)
                        char = prediction["char"]
                        confidence = prediction["confidence"]
                        top3 = prediction.get("top3", [])
                        low_conf = prediction.get("low_confidence", False)
                    else:
                        char = "?"
                        confidence = 0.0
                        top3 = []
                        low_conf = True
                        
                    # Map char back to pattern
                    inv_map = {v: k for k, v in self.decoder.grade1.items() if len(k) == 6}
                    pattern = inv_map.get(char.lower(), (0, 0, 0, 0, 0, 0))
                    
                    cell_dict = {
                        "pattern": pattern,
                        "confidence": confidence,
                        "x": round(cx, 1),
                        "y": round(cy, 1),
                        "bbox": bbox,
                        "dot_count": sum(pattern),
                        "image": square_image,
                        "classifier_char": char,
                        "classifier_confidence": confidence,
                        "classifier_top3": top3,
                        "low_confidence": low_conf,
                    }
                    
                    grid_cells.append(cell_dict)
                    grid_decoded_chars.append({
                        "char": char,
                        "confidence": confidence,
                        "cell_index": global_cell_idx,
                        "source": "classifier" if self.classifier.is_available() else "pattern",
                    })
                    global_cell_idx += 1
                
                # Add a newline between lines if there is another line
                if line_idx < len(lines_cells) - 1:
                    grid_cells.append({
                        "pattern": (0, 0, 0, 0, 0, 0),
                        "confidence": 1.0,
                        "x": x_max + 10.0,
                        "y": y_min_padded + line_h / 2,
                        "bbox": [x_max, float(y_min_padded), x_max + 20.0, float(y_max_padded)],
                        "dot_count": 0,
                        "image": None,
                        "classifier_char": "\n",
                        "classifier_confidence": 1.0,
                        "classifier_top3": [],
                        "low_confidence": False,
                    })
                    grid_decoded_chars.append({
                        "char": "\n",
                        "confidence": 1.0,
                        "cell_index": global_cell_idx,
                        "source": "blank",
                    })
                    global_cell_idx += 1
                    
            return grid_cells, grid_decoded_chars

        # ── Pathway B: Single-Line Horizontal Grid Segmentation ──────────
        # If it's a wide single-line image (aspect ratio >= 1.5) and not multiline
        if aspect_ratio >= 1.5 and not is_multiline:
            logger.info("[WORD-DECODE] Wide image detected (aspect_ratio=%.2f). Applying high-reliability horizontal grid segmentation.", aspect_ratio)
            # Find horizontal bounds of dots
            dark_pixels = np.where(processed_img < 200)
            if len(dark_pixels[1]) > 0:
                x_min = max(0, int(np.min(dark_pixels[1])) - 15)
                x_max = min(img_w, int(np.max(dark_pixels[1])) + 15)
            else:
                x_min = 0
                x_max = img_w
            
            word_width = x_max - x_min
            
            # Estimate N based on aspect ratio
            active_ratio = word_width / img_h
            expected_n = max(1, round(active_ratio / 0.55))
            logger.info("[WORD-DECODE] Grid bounds: [%d, %d], word_width=%d, active_ratio=%.2f -> expected cells N=%d", x_min, x_max, word_width, active_ratio, expected_n)
            
            # Divide horizontally
            cell_width = word_width / expected_n
            grid_cells = []
            grid_decoded_chars = []
            
            for i in range(expected_n):
                cx1 = x_min + i * cell_width
                cx2 = x_min + (i + 1) * cell_width
                
                # Crop and pad to square
                cell_strip = processed_img[:, int(cx1):int(cx2)]
                h_c, w_c = cell_strip.shape[:2]
                max_dim = max(h_c, w_c)
                square_image = np.ones((max_dim, max_dim), dtype=np.uint8) * 255
                y_offset = (max_dim - h_c) // 2
                x_offset = (max_dim - w_c) // 2
                square_image[y_offset:y_offset+h_c, x_offset:x_offset+w_c] = cell_strip
                
                # We also need a bbox and center coordinates
                bbox = [round(cx1, 1), 0.0, round(cx2, 1), float(img_h)]
                cx = cx1 + cell_width / 2
                cy = img_h / 2
                
                # Check if this cell crop is blank (contains no dots / dark pixels)
                is_blank_cell = True
                if cell_strip.size > 0:
                    # If any pixel in the cell strip is significantly darker than paper background (less than 200)
                    if np.min(cell_strip) < 200:
                        is_blank_cell = False
                        
                # Predict
                if is_blank_cell:
                    char = " "
                    confidence = 1.0
                    top3 = [{"char": " ", "confidence": 1.0}]
                    low_conf = False
                elif self.classifier.is_available():
                    from PIL import Image as PILImage
                    pil_crop = PILImage.fromarray(square_image.astype(np.uint8))
                    
                    # Dynamically invert if light background
                    mean_val = np.mean(square_image)
                    if mean_val > 127:
                        from PIL import ImageOps
                        pil_crop_input = ImageOps.invert(pil_crop)
                    else:
                        pil_crop_input = pil_crop
                        
                    prediction = self.classifier.predict_single(pil_crop_input)
                    char = prediction["char"]
                    confidence = prediction["confidence"]
                    top3 = prediction.get("top3", [])
                    low_conf = prediction.get("low_confidence", False)
                else:
                    char = "?"
                    confidence = 0.0
                    top3 = []
                    low_conf = True
                
                # Map char back to pattern
                inv_map = {v: k for k, v in self.decoder.grade1.items() if len(k) == 6}
                pattern = inv_map.get(char.lower(), (0, 0, 0, 0, 0, 0))
                
                cell_dict = {
                    "pattern": pattern,
                    "confidence": confidence,
                    "x": round(cx, 1),
                    "y": round(cy, 1),
                    "bbox": bbox,
                    "dot_count": sum(pattern),
                    "image": square_image,
                    "classifier_char": char,
                    "classifier_confidence": confidence,
                    "classifier_top3": top3,
                    "low_confidence": low_conf,
                }
                
                grid_cells.append(cell_dict)
                grid_decoded_chars.append({
                    "char": char,
                    "confidence": confidence,
                    "cell_index": i,
                    "source": "classifier" if self.classifier.is_available() else "pattern",
                })
                
                logger.info("  Cell %d (Grid): predicted='%s' conf=%.3f", i, char, confidence)
                
            # Log combined result
            decoded_word = "".join(c["char"] for c in grid_decoded_chars)
            avg_conf = float(np.mean([c["confidence"] for c in grid_decoded_chars])) if grid_decoded_chars else 0.0
            logger.info("[WORD-DECODE] Raw decoded word (Grid): '%s'", decoded_word)
            logger.info("[WORD-DECODE] Average confidence (Grid): %.3f", avg_conf)
            
            return grid_cells, grid_decoded_chars

        logger.info(
            "[WORD-DECODE] Starting character-by-character decoding for %d cells",
            len(cells),
        )

        # ── Step 1: Separate / crop each cell from the full image ─────
        separated_cells = self.segmenter.separate_cells_horizontally(processed_img, cells)
        logger.info("[WORD-DECODE] Separated %d individual cells", len(separated_cells))

        decoded_characters: list[dict] = []

        for sep_cell in separated_cells:
            cell_idx   = sep_cell.get("index", 0)
            dot_count  = sep_cell.get("dot_count", 0)
            pattern    = sep_cell.get("pattern", (0, 0, 0, 0, 0, 0))

            # ── Blank / space cell — classifier has no space output ────
            if dot_count == 0 or sum(pattern) == 0:
                logger.info("  Cell %d: BLANK → ' ' (space)", cell_idx)
                decoded_characters.append({
                    "char":       " ",
                    "confidence": 1.0,
                    "cell_index": cell_idx,
                    "source":     "blank",
                })
                if cell_idx < len(cells):
                    cells[cell_idx]["classifier_char"]       = " "
                    cells[cell_idx]["classifier_confidence"] = 1.0
                    cells[cell_idx]["classifier_top3"]       = [{"char": " ", "confidence": 1.0}]
                    cells[cell_idx]["low_confidence"]        = False
                continue

            # ── Neural classifier path ────────────────────────────────
            if self.classifier.is_available():
                try:
                    cell_image = sep_cell["image"]

                    # crop_cell_to_pil already handles background inversion
                    pil_crop = crop_cell_to_pil(
                        cell_image,
                        (0, 0, cell_image.shape[1], cell_image.shape[0]),
                        padding=0,
                    )

                    if pil_crop is None:
                        raise ValueError("crop_cell_to_pil returned None")

                    # ── Step 2: Classify this single cell ─────────────
                    prediction = self.classifier.predict_single(pil_crop)
                    char       = prediction["char"]
                    confidence = prediction["confidence"]

                    logger.info(
                        "  Cell %d: predicted='%s' conf=%.3f  top3=%s",
                        cell_idx, char, confidence,
                        [(t["char"], round(t["confidence"], 3))
                         for t in prediction.get("top3", [])],
                    )

                    decoded_characters.append({
                        "char":       char,
                        "confidence": confidence,
                        "cell_index": cell_idx,
                        "source":     "classifier",
                    })

                    # Enrich original cell in-place
                    if cell_idx < len(cells):
                        cells[cell_idx]["classifier_char"]       = char
                        cells[cell_idx]["classifier_confidence"] = confidence
                        cells[cell_idx]["classifier_top3"]       = prediction.get("top3", [])
                        cells[cell_idx]["low_confidence"]        = prediction.get("low_confidence", False)

                    continue

                except Exception as exc:
                    logger.error(
                        "  Cell %d: classifier failed (%s) — falling back to pattern",
                        cell_idx, exc,
                    )

            # ── Pattern-lookup fallback (classifier unavailable or failed) ─
            pat_tuple = tuple(int(p) for p in pattern)
            char      = self.decoder.grade1.get(pat_tuple, "?")
            conf      = 1.0 if pat_tuple in self.decoder.grade1 else 0.4
            logger.info(
                "  Cell %d: pattern-fallback pattern=%s → '%s' (conf=%.2f)",
                cell_idx, pat_tuple, char, conf,
            )
            decoded_characters.append({
                "char":       char,
                "confidence": conf,
                "cell_index": cell_idx,
                "source":     "pattern",
            })

        # ── Step 3: Log combined result ──────────────────────────────
        decoded_word = "".join(c["char"] for c in decoded_characters)
        if decoded_characters:
            avg_conf = float(np.mean([c["confidence"] for c in decoded_characters]))
        else:
            avg_conf = 0.0

        logger.info("[WORD-DECODE] Raw decoded word : '%s'", decoded_word)
        logger.info("[WORD-DECODE] Average confidence: %.3f", avg_conf)

        return cells, decoded_characters

    # ------------------------------------------------------------------
    # LINE GROUPING (for live scan)
    # ------------------------------------------------------------------

    def _group_cells_into_lines(
        self, cells: list[dict], spacing_ds: float = 20.0
    ) -> list[list[dict]]:
        """
        Group cells into horizontal Braille lines by Y-coordinate proximity.

        Uses a ds-relative tolerance so it adapts to different Braille scales
        instead of a fixed 20px threshold.

        Args:
            cells: Ordered cell list from segment().
            spacing_ds: dot_spacing from estimate_spacing (for adaptive tolerance).

        Returns:
            List of lines, each line is a list of cells sorted by X.
        """
        if not cells:
            return []

        # Tolerance = 60% of dot_spacing — tight enough to separate rows
        # but loose enough to absorb real-world Y jitter within a row
        tol = max(15.0, spacing_ds * 0.65)

        lines: list[list[dict]] = []
        current_line: list[dict] = [cells[0]]
        current_y = cells[0]["y"]

        for cell in cells[1:]:
            if abs(cell["y"] - current_y) <= tol:
                current_line.append(cell)
            else:
                lines.append(sorted(current_line, key=lambda c: c["x"]))
                current_line = [cell]
                current_y = cell["y"]

        if current_line:
            lines.append(sorted(current_line, key=lambda c: c["x"]))

        return lines

    def _decode_line(self, line_cells: list[dict]) -> str:
        """Decode a single line of cells to a string using the classifier path."""
        preds = self._cells_to_predictions(line_cells)
        text, _ = self.decoder.decode_from_predictions(preds)
        return self._clean_decoded_text(text)

    @staticmethod
    def _clean_decoded_text(text: str) -> str:
        """
        Post-process a raw decoded character stream into readable text.

        - Collapse runs of multiple spaces into a single space.
        - Remove any stray '?' noise characters at word boundaries.
        - Collapse consecutively repeated word patterns (e.g., 'olly olly olly' -> 'olly').
        - Collapse N-word repeating chunks (e.g., 'oll j oll j oll' -> 'oll j').
        - Strip leading/trailing whitespace.
        """
        import re
        if not text or not text.strip():
            return text.strip() if text else ""

        # Collapse 2+ spaces to one
        text = re.sub(r" {2,}", " ", text)
        # Remove lone '?' that are isolated (noise from unrecognised dots)
        text = re.sub(r"(?<![\w])\?(?![\w])", "", text)
        # Clean up any double spaces left by '?' removal
        text = re.sub(r" {2,}", " ", text)
        text = text.strip()

        if not text:
            return ""

        # ── Collapse simple consecutive repeated words ─────────────────
        # e.g. 'olly olly olly' → 'olly'
        text = re.sub(r"\b(\w{3,})(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(\w+)(?:\s+\1\b){2,}", r"\1", text, flags=re.IGNORECASE)

        # ── Collapse N-word repeating chunk patterns ───────────────────
        # e.g. 'oll j oll j oll' (unit='oll j' repeated 2.5×) → 'oll j'
        # e.g. 'abc def ghi abc def ghi abc def ghi' → 'abc def ghi'
        words = text.split()
        for unit_size in range(1, 5):          # try unit sizes 1–4
            if len(words) < unit_size * 2:     # need at least 2 full units
                continue
            unit = [w.lower() for w in words[:unit_size]]
            full_repeats = 0
            i = 0
            while i + unit_size <= len(words):
                chunk = [w.lower() for w in words[i:i + unit_size]]
                if chunk == unit:
                    full_repeats += 1
                    i += unit_size
                else:
                    break
            # Allow partial tail match (e.g. last element of the unit)
            if full_repeats >= 2:
                remaining = [w.lower() for w in words[i:]]
                tail_matches = (remaining == unit[:len(remaining)]) if remaining else True
                if tail_matches:
                    collapsed = " ".join(words[:unit_size])
                    logger.info(
                        "_clean_decoded_text: collapsed %d×%d-word chunk '%s' → '%s'",
                        full_repeats, unit_size, text, collapsed,
                    )
                    text = collapsed
                    break

        return text.strip()

    @staticmethod
    def _detect_repetition(text: str) -> bool:
        """
        Check if the decoded text has suspicious repeated word patterns.

        Detects:
          • Simple consecutive repetition:   'olly olly olly'
          • N-word chunk repetition:         'oll j oll j oll'  (unit='oll j')
          • Repeated character substrings:   'abcabcabc'
        """
        if not text or len(text) < 6:
            return False

        words = text.split()

        # ── Simple consecutive word repetition ────────────────────────
        if len(words) >= 3:
            for i in range(len(words) - 2):
                if (
                    words[i].lower() == words[i + 1].lower() == words[i + 2].lower()
                    and len(words[i]) >= 3
                ):
                    logger.warning(
                        "pipeline: suspicious word repetition: '%s'", words[i]
                    )
                    return True

        # ── N-word chunk repetition: 'X Y X Y X' ─────────────────────
        # Try every unit size from 1 to 4 words.
        # A unit is considered repeating if it appears ≥ 2 full times at the
        # start of the word list, with an optional partial tail match.
        for unit_size in range(1, 5):
            if len(words) < unit_size * 2:   # need at least 2 full copies
                continue
            unit = [w.lower() for w in words[:unit_size]]
            # Unit must contain at least one word of length ≥ 2 (not pure noise)
            if not any(len(w) >= 2 for w in unit):
                continue
            full_repeats = 0
            i = 0
            while i + unit_size <= len(words):
                if [w.lower() for w in words[i:i + unit_size]] == unit:
                    full_repeats += 1
                    i += unit_size
                else:
                    break
            if full_repeats >= 2:
                # Allow optional partial tail
                remaining = [w.lower() for w in words[i:]]
                if not remaining or remaining == unit[:len(remaining)]:
                    logger.warning(
                        "pipeline: %d-word chunk repeated %dx: %s",
                        unit_size, full_repeats, " ".join(words[:unit_size]),
                    )
                    return True

        # ── Repeated character substrings ─────────────────────────────
        for chunk_len in range(3, 10):
            if len(text) < chunk_len * 3:
                continue
            for i in range(len(text) - chunk_len * 3 + 1):
                chunk = text[i:i + chunk_len]
                if not chunk.strip() or not any(c.isalnum() for c in chunk):
                    continue
                if (
                    text[i + chunk_len:i + chunk_len * 2] == chunk
                    and text[i + chunk_len * 2:i + chunk_len * 3] == chunk
                ):
                    logger.warning(
                        "pipeline: substring chunk repetition: '%s'", chunk
                    )
                    return True

        return False

    # ------------------------------------------------------------------
    # HEATMAP BUILDER
    # ------------------------------------------------------------------

    def _build_heatmap(self, cells: list[dict], decoded_chars: list[str]) -> list[dict]:
        """
        Build per-cell heatmap data for frontend overlay rendering.

        Each entry contains the cell's bounding box, its decoded character,
        and its classifier confidence so the React Native frontend can draw
        colour-coded boxes over the camera feed.

        Args:
            cells: Enriched cell dicts.
            decoded_chars: List of decoded characters (one per cell).

        Returns:
            List of heatmap entry dicts.
        """
        heatmap: list[dict] = []
        for i, cell in enumerate(cells):
            bbox = cell.get("bbox", [])
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox
            char = decoded_chars[i] if i < len(decoded_chars) else "?"
            conf = cell.get("classifier_confidence", cell.get("confidence", 0.0))
            heatmap.append({
                "x": round(float(x1), 1),
                "y": round(float(y1), 1),
                "w": round(float(x2 - x1), 1),
                "h": round(float(y2 - y1), 1),
                "confidence": round(float(conf), 4),
                "char": char,
            })
        return heatmap

    # ------------------------------------------------------------------
    # FULL IMAGE PROCESSING
    # ------------------------------------------------------------------

    async def process_image(
        self, image_bytes: bytes, options: Optional[dict] = None
    ) -> dict:
        """
        Full pipeline for a single Braille image (camera capture or upload).

        Args:
            image_bytes: Raw image bytes (JPEG/PNG).
            options: Processing options dict:
                correct (bool): Run AI error correction (default True).
                translate_to (str): Target language code, e.g. 'hi'.
                speak (bool): Generate TTS audio bytes (default False).
                save_annotated (bool): Embed annotated image in response (default False).

        Returns:
            Complete result dict with text, confidences, audio, heatmap, metadata.
        """
        opts = options or {}
        do_correct = opts.get("correct", True)
        translate_to = opts.get("translate_to")
        do_speak = opts.get("speak", False)
        save_annotated = opts.get("save_annotated", False)

        t_start = time.monotonic()

        # ── Pre-check: Single-Cell Upload Bypass ───────────────────
        from PIL import Image as PILImage
        from io import BytesIO
        try:
            pil_img = PILImage.open(BytesIO(image_bytes)).convert("RGB")
            w, h = pil_img.size
            aspect_ratio = w / h if h > 0 else 1.0
            
            # A single-cell crop is usually small (e.g. < 250x250) and squareish/vertical
            is_single_cell = (w < 250 and h < 250) and (0.4 <= aspect_ratio <= 1.2)
            logger.info("process_image: Uploaded image size %dx%d, aspect_ratio=%.2f, is_single_cell=%s", w, h, aspect_ratio, is_single_cell)
        except Exception as e:
            logger.warning("process_image: failed to load PIL image for single-cell check: %s", e)
            is_single_cell = False

        if is_single_cell and self.classifier.is_available():
            logger.info("process_image: Single-cell image detected. Running direct classification bypass path.")
            
            # Check background brightness and invert if light (EfficientNet expects white dots on dark background)
            gray_img = pil_img.convert("L")
            mean_val = np.mean(np.array(gray_img))
            if mean_val > 127:
                from PIL import ImageOps
                processed_pil = ImageOps.invert(pil_img)
                logger.info("process_image single-cell bypass: Inverting light background (mean=%.1f) to dark background.", mean_val)
            else:
                processed_pil = pil_img
                logger.info("process_image single-cell bypass: Keeping dark background (mean=%.1f) as is.", mean_val)

            # Classify directly using model transforms
            pred = self.classifier.predict_single(processed_pil)
            pred_char = pred["char"]
            pred_conf = pred["confidence"]
            
            logger.info("process_image single-cell bypass prediction: char='%s', conf=%.4f", pred_char, pred_conf)

            # Build heatmap entry
            heatmap = [{
                "x": 0.0,
                "y": 0.0,
                "w": float(w),
                "h": float(h),
                "confidence": float(pred_conf),
                "char": pred_char,
            }]

            # Map decoded character back to standard 6-dot pattern for response cells representation
            inv_map = {v: k for k, v in self.decoder.grade1.items() if len(k) == 6}
            pattern = list(inv_map.get(pred_char.lower(), (0, 0, 0, 0, 0, 0)))
            cell_dict = {
                "pattern": pattern,
                "confidence": float(pred_conf),
                "x": float(w) / 2,
                "y": float(h) / 2,
                "bbox": [0.0, 0.0, float(w), float(h)],
                "dot_count": int(sum(pattern)),
            }

            # Handle annotated image export if requested
            annotated_b64 = None
            if save_annotated:
                try:
                    vis_img = np.array(pil_img)
                    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(vis_img, (0, 0), (w - 1, h - 1), (46, 204, 113), 4)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(vis_img, f"Pred: {pred_char} ({pred_conf*100:.1f}%)", (10, 30), font, 0.7, (46, 204, 113), 2, cv2.LINE_AA)
                    _, buf = cv2.imencode(".jpg", vis_img, [cv2.IMWRITE_JPEG_QUALITY, ANNOTATED_JPEG_QUALITY])
                    annotated_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                except Exception as exc:
                    logger.warning("process_image single-cell bypass annotated visualization failed: %s", exc)

            # Generate TTS if requested
            audio_bytes = None
            if do_speak and pred_char.strip():
                try:
                    lang = translate_to or "en"
                    audio_bytes = await self.tts.generate_speech_bytes(pred_char, lang=lang)
                except Exception as exc:
                    logger.warning("process_image single-cell bypass TTS generation failed: %s", exc)

            processing_ms = (time.monotonic() - t_start) * 1000

            result = {
                "success": True,
                "raw_text": pred_char,
                "corrected_text": pred_char,
                "translated_text": None,
                "cells": [cell_dict],
                "confidences": [pred_conf],
                "avg_confidence": round(pred_conf, 4),
                "cell_count": 1,
                "dot_count": int(sum(pattern)),
                "guidance": "Single cell decoded successfully! ✅",
                "side_detected": "front",
                "quality": "excellent",
                "detection_quality": "excellent",
                "correction_method": "none",
                "correction_changes": [],
                "was_corrected": False,
                "annotated_image_base64": annotated_b64,
                "audio_bytes": audio_bytes,
                "processing_time_ms": round(processing_ms, 1),
                "classifier_used": True,
                "heatmap": heatmap,
                "error": None,
            }
            self.last_result = result
            return result

        # ── Step 1: Preprocess ──────────────────────────────────────
        try:
            pre = self.preprocessor.full_pipeline(image_bytes)
        except Exception as exc:
            logger.error("process_image: preprocess failed: %s", exc)
            return self._error_result(str(exc))

        processed_img = pre["processed"]
        guidance = pre["guidance"]
        side = pre["side"]
        quality = pre["quality"]

        # ── Step 2: Detect ─────────────────────────────────────────
        detection = self.detector.detect(processed_img)
        dots = detection["dots"]
        det_quality = detection["quality"]

        # ── Dot density sanity gate ───────────────────────────────────────
        # If the dot count is unrealistically high relative to image area,
        # the blob detector has picked up noise/texture. Cap before segmenting.
        img_h, img_w = processed_img.shape[:2]
        img_area_kpx2 = (img_h * img_w) / 1000.0
        density = len(dots) / max(img_area_kpx2, 1.0)
        if density > MAX_DOT_DENSITY_PER_KPX2 or len(dots) > DOT_DENSITY_HARD_CAP:
            logger.warning(
                "process_image: dot density %.3f/kpx² exceeds threshold (dots=%d) — "
                "clamping to top-%d by confidence. Image may be noisy or blurry.",
                density, len(dots), DOT_DENSITY_HARD_CAP,
            )
            dots = sorted(dots, key=lambda d: d["confidence"], reverse=True)[:DOT_DENSITY_HARD_CAP]
            guidance = "Image too noisy — move to better lighting 💡"
            det_quality = "poor"

        # ── Step 3: Segment & Decode with Retry on Repetition ───────
        ds_scale = 1.0
        cells = self.segmenter.segment(dots, image_width=img_w, ds_scale=ds_scale)

        # [DEBUG] Log segmented cells before classification
        logger.info("[DEBUG] Segmented cells: %d cells", len(cells))
        for i, cell in enumerate(cells):
            logger.info(
                "  Cell %d: x=%.1f, y=%.1f, pattern=%s, dot_count=%d",
                i,
                cell.get("x", 0.0),
                cell.get("y", 0.0),
                cell.get("pattern"),
                cell.get("dot_count", 0),
            )

        # ── Step 4: CHARACTER-BY-CHARACTER DECODING ───────────────
        # Implements the handwritten-notes approach:
        #   detect word → separate cells → decode each cell individually
        #   → combine all decoded characters → final word
        #
        # Replaces the old batch-classify approach: each cell is now
        # classified in isolation with predict_single(), which is exactly
        # how the EfficientNet-B3 was trained (on single-cell images).
        cells, decoded_characters = self._decode_word_by_characters(cells, processed_img)

        classifier_used = bool(
            decoded_characters
            and any(c.get("source") == "classifier" for c in decoded_characters)
        )

        # Build predictions list compatible with BrailleDecoder
        predictions: list[dict] = []
        for dc in decoded_characters:
            cidx    = dc.get("cell_index", 0)
            pat     = tuple(int(p) for p in cells[cidx].get("pattern", (0,) * 6)) \
                      if cidx < len(cells) else (0,) * 6
            predictions.append({
                "char":       dc["char"],
                "confidence": dc["confidence"],
                "pattern":    pat,
            })

        logger.info(
            "[WORD-DECODE] Final predictions → decoder: %s",
            [(p["char"], round(p["confidence"], 3)) for p in predictions],
        )

        # ── Step 5: Decode predictions → raw text + per-char confidences
        if predictions:
            raw_text, confidences = self.decoder.decode_from_predictions(predictions)
        else:
            decode_stats = self.decoder.decode_with_stats(cells)
            raw_text     = decode_stats["decoded_text"]
            confidences  = decode_stats["confidences"]

        # Post-process: collapse multi-spaces and strip noise → proper sentence
        cleaned_text = self._clean_decoded_text(raw_text)

        # Check for text repetitions (like 'olly olly olly') and retry with scaled-up ds if found
        if self._detect_repetition(cleaned_text):
            logger.warning(
                "process_image: repetition detected in '%s'! Retrying segmentation with ds * 1.5",
                cleaned_text,
            )
            ds_scale = 1.5
            retry_cells = self.segmenter.segment(dots, image_width=img_w, ds_scale=ds_scale)

            # Re-run character-by-character decoding on retry cells
            retry_cells, retry_decoded_chars = self._decode_word_by_characters(retry_cells, processed_img)
            retry_classifier_used = bool(
                retry_decoded_chars
                and any(c.get("source") == "classifier" for c in retry_decoded_chars)
            )

            retry_predictions: list[dict] = []
            for dc in retry_decoded_chars:
                cidx = dc.get("cell_index", 0)
                pat  = tuple(int(p) for p in retry_cells[cidx].get("pattern", (0,) * 6)) \
                       if cidx < len(retry_cells) else (0,) * 6
                retry_predictions.append({
                    "char":       dc["char"],
                    "confidence": dc["confidence"],
                    "pattern":    pat,
                })

            if retry_predictions:
                retry_raw_text, retry_confidences = self.decoder.decode_from_predictions(retry_predictions)
            else:
                retry_decode_stats = self.decoder.decode_with_stats(retry_cells)
                retry_raw_text     = retry_decode_stats["decoded_text"]
                retry_confidences  = retry_decode_stats["confidences"]

            retry_cleaned_text = self._clean_decoded_text(retry_raw_text)

            # Accept retry result if it produces valid non-empty text
            if retry_cleaned_text.strip():
                cells            = retry_cells
                classifier_used  = retry_classifier_used
                predictions      = retry_predictions
                raw_text         = retry_raw_text
                confidences      = retry_confidences
                cleaned_text     = retry_cleaned_text
                logger.info("process_image: retry successful! New text: '%s'", cleaned_text)
            else:
                logger.info("process_image: retry returned empty text, keeping original")

        raw_text = cleaned_text

        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        # ── Step 6: Correct ────────────────────────────────────────
        correction_result: dict = {
            "corrected": raw_text,
            "method": "none",
            "was_corrected": False,
            "changes": [],
            "confidence": 1.0,
        }
        if do_correct and raw_text.strip():
            ctx = self.memory.get_correction_context()
            correction_result = await asyncio.to_thread(self.corrector.correct, raw_text, context=ctx)

        corrected_text = correction_result["corrected"]

        # ── Step 7: Translate ──────────────────────────────────────
        translated_text: Optional[str] = None
        if translate_to and corrected_text.strip():
            trans = await asyncio.to_thread(self.translator.translate, corrected_text, translate_to)
            translated_text = trans.get("translated") if trans.get("success") else None

        # ── Step 8: Update context memory ─────────────────────────
        if corrected_text.strip():
            self.memory.add_sentence(corrected_text)

        # ── Step 9: Build decoded char list for heatmap ───────────
        decoded_chars: list[str] = []
        for pred in predictions:
            ch = pred.get("char", "?")
            if ch not in ("[CAP]", "[NUM]"):
                decoded_chars.append(ch)
            else:
                decoded_chars.append(ch)  # keep indicators in heatmap
        # Pad/truncate to match cells length
        while len(decoded_chars) < len(cells):
            decoded_chars.append("?")

        heatmap = self._build_heatmap(cells, decoded_chars)

        # ── Step 10: Annotated image ───────────────────────────────
        annotated_b64: Optional[str] = None
        if save_annotated:
            try:
                vis = self.detector.visualize(processed_img, dots)
                vis = self.segmenter.visualize_cells(vis, cells, decoded=decoded_chars)
                _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, ANNOTATED_JPEG_QUALITY])
                annotated_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            except Exception as exc:
                logger.warning("process_image: annotated image failed: %s", exc)

        # ── Step 11: TTS ───────────────────────────────────────────
        audio_bytes: Optional[bytes] = None
        if do_speak and corrected_text.strip():
            try:
                lang = translate_to or "en"
                audio_bytes = await self.tts.generate_speech_bytes(corrected_text, lang=lang)
            except Exception as exc:
                logger.warning("process_image: TTS failed: %s", exc)

        processing_ms = (time.monotonic() - t_start) * 1000

        result = {
            "success": True,
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "translated_text": translated_text,
            "cells": [self._cell_to_dict(c) for c in cells],
            "confidences": confidences,
            "avg_confidence": round(avg_confidence, 4),
            "cell_count": len(cells),
            "dot_count": len(dots),
            "guidance": guidance,
            "side_detected": side,
            "quality": quality,
            "detection_quality": det_quality,
            "correction_method": correction_result["method"],
            "correction_changes": correction_result["changes"],
            "was_corrected": correction_result["was_corrected"],
            "annotated_image_base64": annotated_b64,
            "audio_bytes": audio_bytes,
            "processing_time_ms": round(processing_ms, 1),
            "classifier_used": classifier_used,
            "heatmap": heatmap,
            "error": None,
        }

        self.last_result = result
        logger.info(
            "process_image: '%s...' conf=%.2f cells=%d time=%.0fms classifier=%s",
            corrected_text[:30],
            avg_confidence,
            len(cells),
            processing_ms,
            classifier_used,
        )
        return result

    # ------------------------------------------------------------------
    # LIVE FRAME (optimised fast path)
    # ------------------------------------------------------------------

    async def process_live_frame(self, frame_bytes: bytes) -> dict:
        """
        Optimised fast path for real-time camera frames.

        Skips AI error correction and TTS generation to stay under 200ms.

        Adaptive line budget:
            - Always decodes LIVE_LINES_MIN (2) lines first.
            - If wall time after first 2 lines is < LIVE_BUDGET_EXTEND_MS,
              extends to LIVE_LINES_MAX (4) lines automatically.
            - Returns decoded text per line in lines[] for frontend overlay.

        Args:
            frame_bytes: Raw JPEG frame bytes from mobile camera.

        Returns:
            Minimal result dict: text, lines, line_count, dots, guidance, quality, timing.
        """
        t_start = time.monotonic()

        try:
            pre = self.preprocessor.full_pipeline(frame_bytes)
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "guidance": "Error processing frame",
                "lines": [],
                "line_count": 0,
                "heatmap": [],
                "raw_text": "",
                "corrected_text": "",
                "dot_count": 0,
                "cell_count": 0,
                "avg_confidence": 0.0,
                "side_detected": "unknown",
                "detection_quality": "poor",
                "classifier_used": False,
                "processing_time_ms": 0.0,
            }

        detection = self.detector.detect(pre["processed"])
        dots = detection["dots"]

        # ── Dot density sanity gate (live path) ─────────────────────────
        proc_img = pre["processed"]
        img_h, img_w = proc_img.shape[:2]
        img_area_kpx2 = (img_h * img_w) / 1000.0
        density = len(dots) / max(img_area_kpx2, 1.0)
        if density > MAX_DOT_DENSITY_PER_KPX2 or len(dots) > DOT_DENSITY_HARD_CAP:
            logger.warning(
                "process_live_frame: dot density %.3f/kpx² (dots=%d) — clamping to %d",
                density, len(dots), DOT_DENSITY_HARD_CAP,
            )
            dots = sorted(dots, key=lambda d: d["confidence"], reverse=True)[:DOT_DENSITY_HARD_CAP]

        cells = self.segmenter.segment(dots, image_width=img_w)

        # ── Classify cells in a single forward pass ─────────────────
        cells, classifier_used = self._classify_cells(cells, pre["processed"])

        # ── Group into Braille lines (ds-adaptive tolerance) ─────────
        # Pull dot_spacing from the segmenter's last estimate if available
        spacing_ds = 20.0
        if cells:
            # Estimate from cell X-spread / count heuristic
            spacing_ds = self.segmenter.estimate_spacing(dots)["dot_spacing"] if dots else 20.0

        all_lines = self._group_cells_into_lines(cells, spacing_ds=spacing_ds)

        # ── Adaptive budget: start with LIVE_LINES_MIN, extend if fast ─
        t_after_setup = (time.monotonic() - t_start) * 1000
        n_lines_target = LIVE_LINES_MIN
        if t_after_setup < LIVE_BUDGET_EXTEND_MS:
            n_lines_target = LIVE_LINES_MAX

        target_lines = all_lines[:n_lines_target]

        line_texts: list[str] = []
        decoded_chars_for_target: list[str] = []

        for line_cells in target_lines:
            line_text = self._decode_line(line_cells)
            line_texts.append(line_text)
            for pred in self._cells_to_predictions(line_cells):
                decoded_chars_for_target.append(pred.get("char", "?"))

        # Build full_text from all lines (target + any remaining)
        all_line_texts = list(line_texts)
        for extra_line in all_lines[n_lines_target:]:
            extra_text = self._decode_line(extra_line)
            all_line_texts.append(extra_text)

        # Join lines with a space and clean up the full sentence
        full_text = " ".join(t for t in all_line_texts if t.strip())
        full_text = self._clean_decoded_text(full_text)

        # Build heatmap using decoded chars for target lines only
        # (remaining lines decoded above but not individually classified for speed)
        decoded_chars_padded = list(decoded_chars_for_target)
        while len(decoded_chars_padded) < len(cells):
            decoded_chars_padded.append("?")

        heatmap = self._build_heatmap(cells, decoded_chars_padded)

        # Compute avg confidence from classified cells
        preds_all = self._cells_to_predictions(cells)
        avg_conf = (
            sum(p.get("confidence", 0.0) for p in preds_all) / max(len(preds_all), 1)
        ) if preds_all else 0.0

        processing_ms = (time.monotonic() - t_start) * 1000

        if processing_ms > LIVE_FRAME_TIMEOUT_MS:
            logger.warning(
                "process_live_frame: %.0fms exceeded target %dms (lines=%d)",
                processing_ms,
                LIVE_FRAME_TIMEOUT_MS,
                n_lines_target,
            )
        else:
            logger.debug(
                "process_live_frame: %.0fms lines=%d/%d cells=%d",
                processing_ms,
                n_lines_target,
                len(all_lines),
                len(cells),
            )

        return {
            "success": True,
            "raw_text": full_text.strip(),
            "corrected_text": full_text.strip(),
            "lines": line_texts,           # decoded text for the target lines (2 or 4)
            "line_count": len(all_lines),  # total Braille lines detected in frame
            "dots": dots,
            "cell_count": len(cells),
            "dot_count": len(dots),
            "avg_confidence": round(avg_conf, 4),
            "guidance": pre["guidance"],
            "side_detected": pre["side"],
            "detection_quality": detection["quality"],
            "classifier_used": classifier_used,
            "heatmap": heatmap,
            "processing_time_ms": round(processing_ms, 1),
            "error": None,
        }


    # ------------------------------------------------------------------
    # PDF PROCESSING
    # ------------------------------------------------------------------

    async def process_pdf(
        self, pdf_bytes: bytes, options: Optional[dict] = None
    ) -> dict:
        """
        Process a multi-page PDF of Braille content.

        Extracts each page as an image, processes it through the full
        pipeline, stitches all decoded text, and optionally exports MP3.

        Args:
            pdf_bytes: Raw PDF file bytes.
            options: Same options as process_image, plus:
                generate_audio (bool): Export full MP3 audiobook.
                audio_output_path (str): Path for MP3 export.

        Returns:
            Dict with per-page results, full_text, and optional audio_path.
        """
        opts = options or {}
        generate_audio = opts.get("generate_audio", False)
        audio_output_path = opts.get("audio_output_path", "./output_audiobook.mp3")

        pages_results: list[dict] = []
        all_text_parts: list[str] = []

        try:
            import fitz  # type: ignore  # PyMuPDF

            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            logger.info("process_pdf: %d pages", pdf_doc.page_count)

            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("jpeg")

                page_opts = {**opts, "speak": False, "save_annotated": False}
                page_result = await self.process_image(img_bytes, page_opts)
                page_result["page_number"] = page_num + 1
                pages_results.append(page_result)

                if page_result.get("corrected_text"):
                    all_text_parts.append(page_result["corrected_text"])

            pdf_doc.close()

        except ImportError:
            logger.warning("process_pdf: PyMuPDF (fitz) not installed — trying pdf2image")
            try:
                from pdf2image import convert_from_bytes  # type: ignore
                images = convert_from_bytes(pdf_bytes, dpi=200)

                for page_num, pil_img in enumerate(images):
                    import io
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG")
                    img_bytes = buf.getvalue()

                    page_opts = {**opts, "speak": False, "save_annotated": False}
                    page_result = await self.process_image(img_bytes, page_opts)
                    page_result["page_number"] = page_num + 1
                    pages_results.append(page_result)

                    if page_result.get("corrected_text"):
                        all_text_parts.append(page_result["corrected_text"])

            except Exception as exc:
                logger.error("process_pdf: pdf extraction failed: %s", exc)
                return {
                    "success": False,
                    "error": f"PDF processing failed: {exc}",
                    "pages": [],
                    "full_text": "",
                    "audio_path": None,
                }

        except Exception as exc:
            logger.error("process_pdf: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "pages": [],
                "full_text": "",
                "audio_path": None,
            }

        full_text = "\n\n".join(all_text_parts)
        audio_path: Optional[str] = None

        if generate_audio and full_text.strip():
            try:
                lang = opts.get("translate_to", "en")
                audio_path = await self.tts.save_to_file(
                    full_text, audio_output_path, lang=lang
                )
                logger.info("process_pdf: audiobook saved to '%s'", audio_path)
            except Exception as exc:
                logger.warning("process_pdf: audio export failed: %s", exc)

        return {
            "success": True,
            "pages": pages_results,
            "page_count": len(pages_results),
            "full_text": full_text,
            "audio_path": audio_path,
            "error": None,
        }

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _cell_to_dict(self, cell: dict) -> dict:
        """Convert a cell dict to a JSON-serialisable form."""
        return {
            "pattern": list(cell.get("pattern", [])),
            "confidence": cell.get("confidence", 0.0),
            "x": cell.get("x", 0.0),
            "y": cell.get("y", 0.0),
            "bbox": list(cell.get("bbox", [])),
            "dot_count": cell.get("dot_count", 0),
        }

    def _error_result(self, error_msg: str) -> dict:
        """Build a standardised error result dict."""
        return {
            "success": False,
            "error": error_msg,
            "raw_text": "",
            "corrected_text": "",
            "translated_text": None,
            "cells": [],
            "confidences": [],
            "avg_confidence": 0.0,
            "cell_count": 0,
            "dot_count": 0,
            "guidance": "Processing error — please retry",
            "side_detected": "unknown",
            "quality": {},
            "detection_quality": "poor",
            "correction_method": "none",
            "correction_changes": [],
            "was_corrected": False,
            "annotated_image_base64": None,
            "audio_bytes": None,
            "processing_time_ms": 0.0,
            "classifier_used": False,
            "heatmap": [],
        }


# ─────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("\n" + "=" * 50)
    print("  BrailleAIPipeline Smoke Test")
    print("=" * 50)

    async def run() -> None:
        pipeline = BrailleAIPipeline()
        print("  [OK] Pipeline initialised")
        print(f"  [OK] Classifier available: {pipeline.classifier.is_available()}")

        # Create synthetic image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 230
        for r in range(3):
            for c in range(5):
                cx, cy = 80 + c * 80, 80 + r * 60
                cv2.circle(img, (cx, cy), 10, (80, 80, 80), -1)

        _, buf = cv2.imencode(".jpg", img)
        image_bytes = buf.tobytes()

        # Test live frame
        live_result = await pipeline.process_live_frame(image_bytes)
        print(f"  [OK] process_live_frame: success={live_result['success']} "
              f"lines={live_result['line_count']} time={live_result['processing_time_ms']}ms")
        print(f"    lines={live_result['lines']}")
        print(f"    heatmap_entries={len(live_result['heatmap'])}")

        # Test full image
        full_result = await pipeline.process_image(
            image_bytes,
            options={"correct": False, "speak": False, "save_annotated": True},
        )
        print(f"  [OK] process_image: success={full_result['success']} "
              f"cells={full_result['cell_count']} time={full_result['processing_time_ms']}ms")
        print(f"    guidance='{full_result['guidance']}'")
        print(f"    classifier_used={full_result['classifier_used']}")
        print(f"    heatmap_entries={len(full_result['heatmap'])}")
        print(f"    annotated={'yes' if full_result.get('annotated_image_base64') else 'no'}")

        print("\n[SUCCESS] Smoke test complete.\n")

    asyncio.run(run())
