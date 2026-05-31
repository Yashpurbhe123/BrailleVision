"""
═══════════════════════════════════
📄 FILE 14/42: backend/routers/scan.py
═══════════════════════════════════

BrailleVision AI — Scan API Router
Endpoints for image, live-frame, and PDF Braille scanning.
"""

from __future__ import annotations

import os
import logging
from io import BytesIO
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from pydantic import BaseModel  # type: ignore

from ai.pipeline import BrailleAIPipeline  # type: ignore
from database import db  # type: ignore

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 10 * 1024 * 1024   # 10 MB

router = APIRouter(prefix="/scan", tags=["scan"])

# Singleton pipeline — created once on first import
_pipeline: Optional[BrailleAIPipeline] = None


def get_pipeline() -> BrailleAIPipeline:
    """Return the singleton BrailleAIPipeline, creating it on first call."""
    global _pipeline
    if _pipeline is None:
        _pipeline = BrailleAIPipeline()
    return _pipeline


# ─────────────────────────────────────────────────────────────
# PYDANTIC RESPONSE MODELS
# ─────────────────────────────────────────────────────────────


class CellResult(BaseModel):
    """Single decoded Braille cell."""
    pattern: list[int]
    confidence: float
    x: float
    y: float
    bbox: list[float]
    dot_count: int


class HeatmapEntry(BaseModel):
    """Per-cell bounding box with confidence for frontend overlay rendering."""
    x: float
    y: float
    w: float
    h: float
    confidence: float
    char: str


class ScanResponse(BaseModel):
    """Full scan result for image upload endpoint."""
    success: bool
    raw_text: str
    corrected_text: str
    translated_text: Optional[str] = None
    cells: list[CellResult] = []
    confidences: list[float] = []
    avg_confidence: float
    cell_count: int
    dot_count: int
    guidance: str
    side_detected: str
    detection_quality: str
    correction_method: str
    correction_changes: list[dict] = []
    was_corrected: bool
    annotated_image_base64: Optional[str] = None
    processing_time_ms: float
    processing_ms: Optional[float] = None
    audio_base64: Optional[str] = None
    classifier_used: bool = False
    heatmap: list[HeatmapEntry] = []
    error: Optional[str] = None


class LiveScanResponse(BaseModel):
    """Minimal real-time frame scan result (optimised for speed)."""
    success: bool
    raw_text: str
    corrected_text: str
    dot_count: int
    cell_count: int
    avg_confidence: float
    guidance: str
    side_detected: str
    detection_quality: str
    processing_time_ms: float
    lines: list[str] = []          # decoded text per Braille line (up to LIVE_LINES_PER_FRAME)
    line_count: int = 0            # total number of Braille lines detected
    classifier_used: bool = False
    heatmap: list[HeatmapEntry] = []
    error: Optional[str] = None


class PageResult(BaseModel):
    """Single PDF page result."""
    page_number: int
    success: bool
    corrected_text: str
    avg_confidence: float
    cell_count: int


class PdfScanResponse(BaseModel):
    """Full PDF scan result."""
    success: bool
    page_count: int
    pages: list[PageResult] = []
    full_text: str
    audio_path: Optional[str] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────


@router.post("/image", response_model=ScanResponse, summary="Scan a Braille image")
async def scan_image(
    file: UploadFile = File(..., description="JPEG/PNG Braille image"),
    correct: bool = Form(True, description="Apply AI error correction"),
    translate_to: Optional[str] = Form(None, description="Target language code, e.g. 'hi'"),
    save_history: bool = Form(True, description="Save result to scan history"),
    save_annotated: bool = Form(False, description="Return annotated image as base64"),
) -> ScanResponse:
    """
    Process a Braille image and return full decoded text.

    Runs the complete pipeline: preprocess → detect → segment → decode
    → correct → translate → annotate.
    """
    content = await file.read()
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({len(content)} bytes). Max {MAX_IMAGE_BYTES} bytes.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="Empty file received.")

    # Save a debug copy of the uploaded image
    try:
        debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch")
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, "debug_upload.png")
        with open(debug_path, "wb") as f:
            f.write(content)
        logger.info("Saved debug upload image to: %s", debug_path)
    except Exception as e:
        logger.warning("Failed to save debug upload: %s", e)

    pipeline = get_pipeline()
    options = {
        "correct": correct,
        "translate_to": translate_to,
        "speak": True,
        "save_annotated": save_annotated,
    }

    result = await pipeline.process_image(content, options)

    if save_history and result.get("success"):
        try:
            await db.save_scan({
                "raw_text": result["raw_text"],
                "corrected_text": result["corrected_text"],
                "translated_text": result.get("translated_text"),
                "target_language": translate_to,
                "avg_confidence": result["avg_confidence"],
                "cell_count": result["cell_count"],
                "source_type": "image",
                "correction_method": result.get("correction_method"),
                "side_detected": result.get("side_detected"),
                "processing_time_ms": result["processing_time_ms"],
            })
        except Exception as exc:
            logger.warning("scan_image: history save failed: %s", exc)

    audio_base64 = None
    if result.get("audio_bytes"):
        import base64
        audio_base64 = base64.b64encode(result["audio_bytes"]).decode("utf-8")

    cells = [CellResult(**c) for c in result.get("cells", [])]
    heatmap_raw = result.get("heatmap", [])
    heatmap = [HeatmapEntry(**h) for h in heatmap_raw]
    return ScanResponse(
        success=result["success"],
        raw_text=result["raw_text"],
        corrected_text=result["corrected_text"],
        translated_text=result.get("translated_text"),
        cells=cells,
        confidences=result.get("confidences", []),
        avg_confidence=result["avg_confidence"],
        cell_count=result["cell_count"],
        dot_count=result.get("dot_count", 0),
        guidance=result["guidance"],
        side_detected=result["side_detected"],
        detection_quality=result.get("detection_quality", "unknown"),
        correction_method=result.get("correction_method", "none"),
        correction_changes=result.get("correction_changes", []),
        was_corrected=result.get("was_corrected", False),
        annotated_image_base64=result.get("annotated_image_base64"),
        processing_time_ms=result["processing_time_ms"],
        processing_ms=result["processing_time_ms"],
        audio_base64=audio_base64,
        classifier_used=result.get("classifier_used", False),
        heatmap=heatmap,
        error=result.get("error"),
    )


@router.post("/live", response_model=LiveScanResponse, summary="Process live camera frame")
async def scan_live(
    frame: UploadFile = File(..., description="JPEG camera frame"),
) -> LiveScanResponse:
    """
    Fast-path endpoint for real-time camera frame processing.

    Skips error correction for speed. Target latency < 200ms.
    """
    content = await frame.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty frame received.")

    # Save a debug copy of the uploaded image
    try:
        debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch")
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, "debug_upload.png")
        with open(debug_path, "wb") as f:
            f.write(content)
        logger.info("Saved debug live frame image to: %s", debug_path)
    except Exception as e:
        logger.warning("Failed to save debug live frame: %s", e)

    pipeline = get_pipeline()
    result = await pipeline.process_live_frame(content)

    heatmap_raw = result.get("heatmap", [])
    return LiveScanResponse(
        success=result["success"],
        raw_text=result.get("raw_text", ""),
        corrected_text=result.get("corrected_text", ""),
        dot_count=result.get("dot_count", 0),
        cell_count=result.get("cell_count", 0),
        avg_confidence=result.get("avg_confidence", 0.0),
        guidance=result.get("guidance", ""),
        side_detected=result.get("side_detected", "unknown"),
        detection_quality=result.get("detection_quality", "unknown"),
        processing_time_ms=result.get("processing_time_ms", 0.0),
        lines=result.get("lines", []),
        line_count=result.get("line_count", 0),
        classifier_used=result.get("classifier_used", False),
        heatmap=[HeatmapEntry(**h) for h in heatmap_raw],
        error=result.get("error"),
    )


@router.post("/pdf", response_model=PdfScanResponse, summary="Decode a Braille PDF")
async def scan_pdf(
    file: UploadFile = File(..., description="PDF file containing Braille content"),
    generate_audio: bool = Form(False, description="Export decoded text as MP3 audiobook"),
    translate_to: Optional[str] = Form(None, description="Target language code"),
    correct: bool = Form(True, description="Apply AI error correction per page"),
) -> PdfScanResponse:
    """
    Process a multi-page Braille PDF and optionally generate an MP3 audiobook.

    Each page is processed independently; results are stitched into full_text.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty PDF file received.")

    pipeline = get_pipeline()
    options = {
        "correct": correct,
        "translate_to": translate_to,
        "generate_audio": generate_audio,
        "audio_output_path": "./data/audiobook_export.mp3",
    }

    result = await pipeline.process_pdf(content, options)

    page_results = []
    for p in result.get("pages", []):
        page_results.append(
            PageResult(
                page_number=p.get("page_number", 0),
                success=p.get("success", False),
                corrected_text=p.get("corrected_text", ""),
                avg_confidence=p.get("avg_confidence", 0.0),
                cell_count=p.get("cell_count", 0),
            )
        )

    return PdfScanResponse(
        success=result["success"],
        page_count=result.get("page_count", 0),
        pages=page_results,
        full_text=result.get("full_text", ""),
        audio_path=result.get("audio_path"),
        error=result.get("error"),
    )
