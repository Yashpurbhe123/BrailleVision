"""
BrailleVision AI — Hybrid Braille Dot Detector
Fuses OpenCV SimpleBlobDetector + YOLOv8 detections via
confidence-weighted NMS for maximum dot recall.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
# PATH RESOLUTION
# ─────────────────────────────────────────────────────────────

# Layout:
#   BrailleVision/
#     backend/core/detector.py   ← this file
#     models/yolov8n.pt          ← target model
_BACKEND_DIR  = Path(__file__).resolve().parent.parent   # → backend/
_PROJECT_ROOT = _BACKEND_DIR.parent                       # → BrailleVision/
_MODELS_DIR   = _PROJECT_ROOT / "models"

DEFAULT_YOLO_PATH = _MODELS_DIR / "yolov8n.pt"

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# Blob area range — tuned for typical camera-captured Braille at 1MP+
# Real Braille dots are ~1.5mm; at 3px/mm → radius≈4.5px → area≈63px²
# Floor=25 avoids paper texture specks; cap=2500 avoids large noise blobs.
BLOB_MIN_AREA = 25      # Rejects sub-pixel paper texture specks
BLOB_MAX_AREA = 2500    # v3: tightened from 4000; real dots stay well below
BLOB_MIN_CIRCULARITY = 0.25   # v3: lowered from 0.40 — embossed dots cast
                               #     elongated shadows that fail 0.40
BLOB_MIN_CONVEXITY = 0.50    # Balanced convexity threshold
BLOB_MIN_INERTIA = 0.20      # Allow slight ellipse from shadow
BLOB_MIN_DIST = 6.0           # Minimum distance between dots

# Pre-detection Gaussian blur kernel (5×5) to suppress paper texture noise
BLOB_PREBLUR_KSIZE = 5

# Post-detection sigma-based outlier removal threshold
# Dots whose size deviates by more than this many σ from the median are noise
BLOB_SIGMA_OUTLIER_THRESHOLD = 2.5

YOLO_INPUT_SIZE = 640
YOLO_CONF_THRESHOLD = 0.30

NMS_RADIUS = 15          # Balanced NMS radius

# Post-detection filters
# If blobs > BLOB_HARD_CAP after size-filtering, keep only the top-N by confidence
BLOB_HARD_CAP = 500
# IQR multiplier for size outlier removal (1.5 is standard box-plot fence)
BLOB_SIZE_IQR_MULTIPLIER = 2.0

YOLO_WEIGHT_HIGH = 0.70   # yolo weight when avg_conf > 0.5
BLOB_WEIGHT_HIGH = 0.30

YOLO_WEIGHT_LOW = 0.40    # yolo weight when avg_conf <= 0.5
BLOB_WEIGHT_LOW = 0.60

QUALITY_MIN_GOOD = 6
QUALITY_MIN_LOW = 3


# ─────────────────────────────────────────────────────────────
# DETECTOR CLASS
# ─────────────────────────────────────────────────────────────


class HybridBrailleDetector:
    """
    Hybrid dot detector combining OpenCV SimpleBlobDetector with YOLOv8.

    Detection strategy:
        - SimpleBlobDetector: fast, parameter-tuned for Braille dot geometry
        - YOLOv8 nano: deep-learning fallback for challenging conditions
        - Fusion: confidence-weighted NMS merges both result sets
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialise the hybrid detector.

        Args:
            model_path: Path to fine-tuned YOLOv8 .pt file.
                        Falls back to yolov8n.pt if not found.
        """
        self.blob_detector = self._create_blob_detector()
        self.yolo: Optional[object] = None
        self.yolo_available = False

        # Resolve model path: prefer explicit arg, then env var, then models/yolov8n.pt
        resolved_path = (
            model_path
            or os.getenv("MODEL_PATH", "")
            or str(DEFAULT_YOLO_PATH)
        )
        self._load_yolo_model(resolved_path)
        logger.info(
            "HybridBrailleDetector ready. yolo_available=%s", self.yolo_available
        )

    # ------------------------------------------------------------------
    # BLOB DETECTOR SETUP
    # ------------------------------------------------------------------

    def _create_blob_detector(self) -> cv2.SimpleBlobDetector:
        """
        Create a tuned SimpleBlobDetector for Braille dot geometry.

        Returns:
            Configured cv2.SimpleBlobDetector instance.
        """
        params = cv2.SimpleBlobDetector_Params()

        # Look for dark blobs (Braille dots appear as dark shadows on light paper)
        params.filterByColor = True
        params.blobColor = 0   # 0 = dark blobs

        params.filterByArea = True
        params.minArea = BLOB_MIN_AREA
        params.maxArea = BLOB_MAX_AREA

        params.filterByCircularity = True
        params.minCircularity = BLOB_MIN_CIRCULARITY

        params.filterByConvexity = True
        params.minConvexity = BLOB_MIN_CONVEXITY

        params.filterByInertia = True
        params.minInertiaRatio = BLOB_MIN_INERTIA

        params.minDistBetweenBlobs = BLOB_MIN_DIST

        detector = cv2.SimpleBlobDetector_create(params)
        logger.debug("SimpleBlobDetector created with area=[%d,%d]", BLOB_MIN_AREA, BLOB_MAX_AREA)
        return detector

    # ------------------------------------------------------------------
    # BLOB DETECTION
    # ------------------------------------------------------------------

    def detect_blobs(self, img: np.ndarray) -> list[dict]:
        """
        Run SimpleBlobDetector on a grayscale image.

        v3 Pipeline:
            1. GaussianBlur (5×5) pre-pass to suppress paper texture noise.
            2. Adaptive threshold → clean binary image.
            3. SimpleBlobDetector on binary.
            4. IQR-based size outlier removal.
            5. Sigma-based (2.5σ) secondary outlier removal.
            6. Hard cap on max blobs.

        Args:
            img: Grayscale ndarray (already preprocessed).

        Returns:
            List of dot dicts: {x, y, size, confidence, source}.
        """
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── v3 Step 1: Gaussian pre-blur ────────────────────────────
        # Smooths paper texture / embossing micro-noise before thresholding.
        # A 5×5 kernel is small enough to preserve dot shape while killing
        # high-frequency specks that cause false blob hits.
        gray = cv2.GaussianBlur(
            gray,
            (BLOB_PREBLUR_KSIZE, BLOB_PREBLUR_KSIZE),
            0,  # sigma auto-derived from kernel size
        )

        # ── Step 2: Adaptive threshold ───────────────────────────────
        # blockSize=51: spans several dot spacings → local mean = paper BG.
        # C=8: robust to shadow & texture; only dots that are substantially
        # darker than local neighbourhood survive.
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=51,
            C=8,
        )

        keypoints = self.blob_detector.detect(binary)

        if not keypoints:
            # Fallback: try with inverted image in case dots are bright
            keypoints = self.blob_detector.detect(cv2.bitwise_not(binary))
            if not keypoints:
                logger.debug("detect_blobs: no blobs found")
                return []

        sizes = np.array([kp.size for kp in keypoints], dtype=np.float32)
        median_size = float(np.median(sizes))

        # ── Stage 1: confidence-weighted dot list ─────────────────────
        dots: list[dict] = []
        for kp in keypoints:
            size_dev = abs(kp.size - median_size) / max(median_size, 1e-6)
            confidence = float(max(0.3, 1.0 - size_dev))
            dots.append(
                {
                    "x": float(kp.pt[0]),
                    "y": float(kp.pt[1]),
                    "size": float(kp.size),
                    "confidence": round(confidence, 3),
                    "source": "blob",
                }
            )

        # ── Stage 2: IQR-based size outlier removal ─────────────────
        if len(dots) > 4:
            dot_sizes = np.array([d["size"] for d in dots], dtype=np.float32)
            q1, q3 = float(np.percentile(dot_sizes, 25)), float(np.percentile(dot_sizes, 75))
            iqr = q3 - q1
            lo = q1 - BLOB_SIZE_IQR_MULTIPLIER * iqr
            hi = q3 + BLOB_SIZE_IQR_MULTIPLIER * iqr
            before = len(dots)
            dots = [d for d in dots if lo <= d["size"] <= hi]
            if before != len(dots):
                logger.debug(
                    "detect_blobs: IQR filter removed %d outlier blobs (size [%.1f,%.1f])",
                    before - len(dots), lo, hi,
                )

        # ── Stage 3: Sigma-based outlier removal (v3 new) ────────────
        # Removes dots whose size is > BLOB_SIGMA_OUTLIER_THRESHOLD standard
        # deviations from the median dot size.  This is a second, tighter pass
        # that catches elongated noise blobs that survive the IQR fence.
        if len(dots) > 4:
            dot_sizes = np.array([d["size"] for d in dots], dtype=np.float32)
            median_sz = float(np.median(dot_sizes))
            std_sz = float(np.std(dot_sizes))
            if std_sz > 0:
                before = len(dots)
                dots = [
                    d for d in dots
                    if abs(d["size"] - median_sz) <= BLOB_SIGMA_OUTLIER_THRESHOLD * std_sz
                ]
                removed = before - len(dots)
                if removed:
                    logger.debug(
                        "detect_blobs: sigma filter removed %d outliers "
                        "(median=%.1f std=%.1f threshold=%.1fσ)",
                        removed, median_sz, std_sz, BLOB_SIGMA_OUTLIER_THRESHOLD,
                    )

        # ── Stage 4: hard cap ────────────────────────────────────────
        if len(dots) > BLOB_HARD_CAP:
            dots.sort(key=lambda d: d["confidence"], reverse=True)
            logger.warning(
                "detect_blobs: %d blobs exceeds cap %d — keeping top-%d by confidence",
                len(dots), BLOB_HARD_CAP, BLOB_HARD_CAP,
            )
            dots = dots[:BLOB_HARD_CAP]

        logger.debug("detect_blobs: found %d blobs (after filtering)", len(dots))
        return dots

    # ------------------------------------------------------------------
    # YOLO MODEL LOADING
    # ------------------------------------------------------------------

    def _load_yolo_model(self, model_path: str) -> None:
        """
        Load YOLOv8 model. Falls back to yolov8n.pt if custom model not found.

        Args:
            model_path: Path to .pt or .onnx model file.
        """
        try:
            # PyTorch 2.6+ changed the default value of weights_only to True.
            # YOLOv8 models contain custom ultralytics classes that fail strict unpickling.
            # We globally monkeypatch torch.load to set weights_only=False for the trusted local model.
            import torch
            orig_load = torch.load

            def patched_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return orig_load(*args, **kwargs)

            torch.load = patched_load
            logger.info("YOLOv8: Patched torch.load (weights_only=False override enabled)")
        except Exception as e:
            logger.debug("Failed to patch torch.load: %s", e)

        try:
            from ultralytics import YOLO  # type: ignore

            if model_path and os.path.exists(model_path):
                self.yolo = YOLO(model_path)
                logger.info("YOLOv8: loaded model from '%s'", model_path)
            elif DEFAULT_YOLO_PATH.exists():
                self.yolo = YOLO(str(DEFAULT_YOLO_PATH))
                logger.info(
                    "YOLOv8: loaded models/yolov8n.pt from '%s'", DEFAULT_YOLO_PATH
                )
            else:
                # Last-resort: let ultralytics download yolov8n.pt
                self.yolo = YOLO("yolov8n.pt")
                logger.warning(
                    "YOLOv8: models/yolov8n.pt not found at '%s' — "
                    "falling back to ultralytics auto-download",
                    DEFAULT_YOLO_PATH,
                )

            # Warmup pass
            dummy = np.zeros((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), dtype=np.uint8)
            self.yolo(dummy, conf=YOLO_CONF_THRESHOLD, verbose=False)
            self.yolo_available = True
            logger.info("YOLOv8: warmup complete")

        except Exception as exc:
            logger.warning("YOLOv8 unavailable: %s. Using blob-only mode.", exc)
            self.yolo = None
            self.yolo_available = False

    # ------------------------------------------------------------------
    # YOLO DETECTION
    # ------------------------------------------------------------------

    def detect_yolo(self, img: np.ndarray) -> list[dict]:
        """
        Run YOLOv8 inference and return detected dot centre-points.

        Resizes image to 640×640 for inference then maps detections
        back to original image coordinates.

        Args:
            img: BGR or grayscale ndarray.

        Returns:
            List of dot dicts: {x, y, w, h, confidence, source}.
        """
        if not self.yolo_available or self.yolo is None:
            return []

        try:
            orig_h, orig_w = img.shape[:2]
            resized = cv2.resize(img, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))

            # Convert grayscale to BGR for YOLO
            if len(resized.shape) == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

            results = self.yolo(resized, conf=YOLO_CONF_THRESHOLD, verbose=False)

            scale_x = orig_w / YOLO_INPUT_SIZE
            scale_y = orig_h / YOLO_INPUT_SIZE

            dots: list[dict] = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = xyxy
                    cx = ((x1 + x2) / 2) * scale_x
                    cy = ((y1 + y2) / 2) * scale_y
                    w = (x2 - x1) * scale_x
                    h = (y2 - y1) * scale_y
                    dots.append(
                        {
                            "x": float(cx),
                            "y": float(cy),
                            "w": float(w),
                            "h": float(h),
                            "size": float((w + h) / 2),
                            "confidence": round(conf, 3),
                            "source": "yolo",
                        }
                    )

            logger.debug("detect_yolo: %d detections", len(dots))
            return dots

        except Exception as exc:
            logger.error("detect_yolo failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # NMS
    # ------------------------------------------------------------------

    def _hamming_nms(
        self, dots: list[dict], radius: int = NMS_RADIUS
    ) -> list[dict]:
        """
        Spatial Non-Maximum Suppression: merge nearby dots.

        For each pair of dots within `radius` pixels, keep the one
        with higher confidence; average positions weighted by confidence.

        Args:
            dots: Combined list of dot dicts.
            radius: Pixel radius for merging (default 12).

        Returns:
            Deduplicated dot list.
        """
        if not dots:
            return []

        # Sort by confidence descending
        sorted_dots = sorted(dots, key=lambda d: d["confidence"], reverse=True)
        suppressed = [False] * len(sorted_dots)
        merged: list[dict] = []

        for i, dot_i in enumerate(sorted_dots):
            if suppressed[i]:
                continue

            cluster = [dot_i]
            for j, dot_j in enumerate(sorted_dots):
                if i == j or suppressed[j]:
                    continue
                dx = dot_i["x"] - dot_j["x"]
                dy = dot_i["y"] - dot_j["y"]
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist <= radius:
                    cluster.append(dot_j)
                    suppressed[j] = True

            # Weighted average position
            total_conf = sum(d["confidence"] for d in cluster)
            mx = sum(d["x"] * d["confidence"] for d in cluster) / max(total_conf, 1e-6)
            my = sum(d["y"] * d["confidence"] for d in cluster) / max(total_conf, 1e-6)

            best = cluster[0]
            merged.append(
                {
                    "x": round(mx, 1),
                    "y": round(my, 1),
                    "size": best.get("size", 10.0),
                    "confidence": round(best["confidence"], 3),
                    "source": best["source"],
                }
            )

        logger.debug("_hamming_nms: %d → %d dots", len(dots), len(merged))
        return merged

    # ------------------------------------------------------------------
    # FUSION
    # ------------------------------------------------------------------

    def fuse_detections(
        self,
        blob_dots: list[dict],
        yolo_dots: list[dict],
        yolo_avg_conf: float,
    ) -> list[dict]:
        """
        Combine blob and YOLO detections with confidence weighting.

        If YOLO is performing well (avg_conf > 0.5) it gets higher weight;
        otherwise blob detector takes precedence.

        Args:
            blob_dots: Dots from SimpleBlobDetector.
            yolo_dots: Dots from YOLOv8.
            yolo_avg_conf: Average YOLO detection confidence.

        Returns:
            Fused, NMS-filtered dot list.
        """
        if yolo_avg_conf > 0.5:
            yolo_w, blob_w = YOLO_WEIGHT_HIGH, BLOB_WEIGHT_HIGH
            method = "yolo_dominant"
        else:
            yolo_w, blob_w = YOLO_WEIGHT_LOW, BLOB_WEIGHT_LOW
            method = "blob_dominant"

        scaled_yolo = [
            {**d, "confidence": round(d["confidence"] * yolo_w, 3)} for d in yolo_dots
        ]
        scaled_blob = [
            {**d, "confidence": round(d["confidence"] * blob_w, 3)} for d in blob_dots
        ]

        combined = scaled_yolo + scaled_blob
        fused = self._hamming_nms(combined)

        logger.info(
            "fuse_detections: blob=%d yolo=%d → fused=%d method=%s",
            len(blob_dots),
            len(yolo_dots),
            len(fused),
            method,
        )
        return fused, method

    # ------------------------------------------------------------------
    # QUALITY
    # ------------------------------------------------------------------

    def get_detection_quality(
        self, dots: list[dict], img_shape: tuple[int, ...]
    ) -> str:
        """
        Assess detection quality from dot count and spatial distribution.

        Args:
            dots: List of detected dot dicts.
            img_shape: Shape of the source image.

        Returns:
            Quality label: 'good', 'low', or 'poor'.
        """
        count = len(dots)
        if count < QUALITY_MIN_LOW:
            return "poor"
        if count < QUALITY_MIN_GOOD:
            return "low"

        # Check spatial spread — at least 2 rows and 2 columns expected
        xs = [d["x"] for d in dots]
        ys = [d["y"] for d in dots]
        x_spread = max(xs) - min(xs) if xs else 0
        y_spread = max(ys) - min(ys) if ys else 0

        if x_spread < 20 or y_spread < 10:
            return "low"  # all dots clustered at one point

        return "good"

    # ------------------------------------------------------------------
    # MAIN DETECT
    # ------------------------------------------------------------------

    def detect(self, img: np.ndarray) -> dict:
        """
        Run full hybrid detection pipeline on an image.

        Args:
            img: Preprocessed grayscale or BGR ndarray.

        Returns:
            Dict with dots list, counts, quality, and fusion metadata.
        """
        blob_dots = self.detect_blobs(img)
        yolo_dots = self.detect_yolo(img)

        yolo_avg_conf = (
            sum(d["confidence"] for d in yolo_dots) / len(yolo_dots)
            if yolo_dots
            else 0.0
        )

        if yolo_dots or blob_dots:
            fused, fusion_method = self.fuse_detections(blob_dots, yolo_dots, yolo_avg_conf)
        else:
            fused = []
            fusion_method = "none"

        quality = self.get_detection_quality(fused, img.shape)

        result = {
            "dots": fused,
            "count": len(fused),
            "quality": quality,
            "yolo_count": len(yolo_dots),
            "blob_count": len(blob_dots),
            "yolo_avg_conf": round(yolo_avg_conf, 3),
            "fusion_method": fusion_method,
        }
        logger.info(
            "detect: total=%d quality=%s fusion=%s",
            len(fused),
            quality,
            fusion_method,
        )
        return result

    # ------------------------------------------------------------------
    # VISUALISE
    # ------------------------------------------------------------------

    def visualize(
        self, img: np.ndarray, dots: list[dict]
    ) -> np.ndarray:
        """
        Draw detected dots on a copy of the image.

        Colour-coded by confidence:
            Green  → conf ≥ 0.8
            Yellow → conf 0.5–0.79
            Red    → conf < 0.5

        Args:
            img: BGR or grayscale ndarray.
            dots: List of dot dicts from detect().

        Returns:
            Annotated BGR ndarray.
        """
        vis = img.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        for dot in dots:
            x, y = int(dot["x"]), int(dot["y"])
            conf = dot.get("confidence", 0.5)
            radius = max(4, int(dot.get("size", 10) / 2))

            if conf >= 0.8:
                color = (0, 255, 60)      # bright green
            elif conf >= 0.5:
                color = (0, 220, 255)     # yellow-cyan
            else:
                color = (0, 60, 255)      # red

            cv2.circle(vis, (x, y), radius, color, 2)
            cv2.circle(vis, (x, y), 2, color, -1)

        return vis


# ─────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("\n" + "=" * 50)
    print("  HybridBrailleDetector Smoke Test")
    print("=" * 50)

    detector = HybridBrailleDetector()

    # Synthetic image: white background with small dark circles
    img = np.ones((480, 640), dtype=np.uint8) * 230
    positions = [(100 + c * 60, 100 + r * 80) for r in range(4) for c in range(6)]
    for x, y in positions:
        cv2.circle(img, (x, y), 10, 30, -1)

    print(f"  Synthetic image: {img.shape}  dots_drawn={len(positions)}")

    blob_dots = detector.detect_blobs(img)
    print(f"  blob detections: {len(blob_dots)}")

    result = detector.detect(img)
    print(f"  total fused: {result['count']}  quality={result['quality']}")
    print(f"  fusion_method={result['fusion_method']}")

    vis = detector.visualize(img, result["dots"])
    print(f"  visualize: shape={vis.shape}  ✓")

    print("\n✅ Smoke test complete.\n")
