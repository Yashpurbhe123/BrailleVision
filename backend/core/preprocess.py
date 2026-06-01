"""
BrailleVision AI — Image Preprocessor
Full pipeline: load → grayscale → CLAHE → shadow removal →
threshold → perspective correction → side detection → mirror.
"""

from __future__ import annotations

import logging
from typing import Union

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

BRIGHTNESS_LOW_THRESHOLD = 60
BRIGHTNESS_HIGH_THRESHOLD = 200
BLUR_THRESHOLD = 100.0
SIDE_VARIANCE_THRESHOLD = 500.0
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_SIZE = (8, 8)
SHADOW_KERNEL_SIZE = 21   # Must be larger than a Braille dot to model background correctly
SHADOW_DILATE_ITER = 5    # 5 iterations is enough; 15 merges dots into blobs
PERSPECTIVE_APPROX_EPSILON = 0.02


# ─────────────────────────────────────────────────────────────
# PREPROCESSOR CLASS
# ─────────────────────────────────────────────────────────────


class ImagePreprocessor:
    """
    Provides a full preprocessing pipeline for physical Braille images.

    Pipeline steps:
        1. load_image          — accepts path / bytes / ndarray
        2. to_grayscale        — converts BGR → gray
        3. apply_clahe         — adaptive histogram equalisation
        4. remove_shadows      — background normalisation
        5. apply_threshold     — binarisation
        6. auto_correct_perspective — warp paper to rectangle
        7. detect_side         — front (indentation) vs back (raised)
        8. mirror_if_back      — flip back-view images horizontally
    """

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    def load_image(
        self, source: Union[str, bytes, np.ndarray]
    ) -> np.ndarray:
        """
        Load an image from a file path, raw bytes, or an existing ndarray.

        Args:
            source: File path string, raw JPEG/PNG bytes, or BGR ndarray.

        Returns:
            BGR ndarray copy.

        Raises:
            ValueError: If source cannot be decoded into a valid image.
        """
        if isinstance(source, np.ndarray):
            if source.size == 0:
                raise ValueError("Received empty ndarray.")
            logger.debug("load_image: ndarray input %s", source.shape)
            return source.copy()

        if isinstance(source, bytes):
            arr = np.frombuffer(source, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(
                    "Failed to decode image bytes. Ensure bytes are valid JPEG/PNG."
                )
            logger.debug("load_image: bytes input → %s", img.shape)
            return img

        if isinstance(source, str):
            img = cv2.imread(source, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to load image from path: {source!r}")
            logger.debug("load_image: path '%s' → %s", source, img.shape)
            return img

        raise ValueError(
            f"Unsupported source type {type(source)}. "
            "Expected str path, bytes, or numpy ndarray."
        )

    # ------------------------------------------------------------------
    # GRAYSCALE
    # ------------------------------------------------------------------

    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """
        Convert a BGR image to grayscale.

        Args:
            img: BGR or already-grayscale ndarray.

        Returns:
            Single-channel grayscale ndarray.
        """
        if len(img.shape) == 2:
            logger.debug("to_grayscale: already grayscale")
            return img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        logger.debug("to_grayscale: converted %s → %s", img.shape, gray.shape)
        return gray

    # ------------------------------------------------------------------
    # CLAHE
    # ------------------------------------------------------------------

    def apply_clahe(
        self,
        img: np.ndarray,
        clip_limit: float = CLAHE_CLIP_LIMIT,
        tile_size: tuple[int, int] = CLAHE_TILE_SIZE,
    ) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalisation (CLAHE).

        Enhances local contrast — critical for Braille in uneven lighting.

        Args:
            img: Grayscale ndarray.
            clip_limit: Threshold for contrast limiting (default 3.0).
            tile_size: Grid size for CLAHE (default 8×8).

        Returns:
            CLAHE-enhanced grayscale ndarray.
        """
        if len(img.shape) != 2:
            raise ValueError("apply_clahe requires a grayscale (2D) image.")
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        result = clahe.apply(img)
        logger.debug("apply_clahe: clip=%.1f tile=%s", clip_limit, tile_size)
        return result

    # ------------------------------------------------------------------
    # SHADOW REMOVAL
    # ------------------------------------------------------------------

    def remove_shadows(self, img: np.ndarray) -> np.ndarray:
        """
        Remove cast shadows using morphological black-hat background subtraction.

        Dilates the image to get a background estimate, then subtracts
        the original from the background to normalise illumination.

        Args:
            img: Grayscale ndarray.

        Returns:
            Shadow-normalised grayscale ndarray (uint8).
        """
        kernel = np.ones((SHADOW_KERNEL_SIZE, SHADOW_KERNEL_SIZE), np.uint8)
        background = cv2.dilate(img, kernel, iterations=SHADOW_DILATE_ITER)

        # Subtract original from background to cancel illumination variations
        diff = cv2.subtract(background, img)
        
        # Invert so dots are dark on a perfectly clean white background
        normalised = cv2.bitwise_not(diff)

        logger.debug("remove_shadows: completed via subtraction")
        return normalised

    # ------------------------------------------------------------------
    # THRESHOLD
    # ------------------------------------------------------------------

    def apply_threshold(
        self, img: np.ndarray, method: str = "otsu"
    ) -> np.ndarray:
        """
        Binarise a grayscale image.

        Args:
            img: Grayscale ndarray.
            method: One of 'otsu', 'adaptive', or 'binary'.

        Returns:
            Binary (0/255) ndarray.

        Raises:
            ValueError: If method is not recognised.
        """
        if method == "otsu":
            _, binary = cv2.threshold(
                img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            logger.debug("apply_threshold: otsu")
            return binary

        if method == "adaptive":
            binary = cv2.adaptiveThreshold(
                img,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2,
            )
            logger.debug("apply_threshold: adaptive")
            return binary

        if method == "binary":
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            logger.debug("apply_threshold: binary(127)")
            return binary

        raise ValueError(
            f"Unknown threshold method '{method}'. Choose 'otsu', 'adaptive', or 'binary'."
        )

    # ------------------------------------------------------------------
    # PERSPECTIVE CORRECTION
    # ------------------------------------------------------------------

    def auto_correct_perspective(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """
        Detect the largest rectangular contour and warp to a flat view.

        Useful when the camera is angled relative to the Braille page.

        Args:
            img: Grayscale or BGR ndarray.

        Returns:
            Tuple of (corrected_image, was_corrected_bool).
        """
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            logger.debug("auto_correct_perspective: no contours found")
            return img, False

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        h_img, w_img = gray.shape[:2]
        min_area = 0.15 * h_img * w_img

        for contour in contours[:5]:  # check top 5 by area
            if cv2.contourArea(contour) < min_area:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(
                contour, PERSPECTIVE_APPROX_EPSILON * peri, True
            )

            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                pts = self._order_points(pts)

                (tl, tr, br, bl) = pts
                width = int(max(
                    np.linalg.norm(br - bl),
                    np.linalg.norm(tr - tl),
                ))
                height = int(max(
                    np.linalg.norm(tr - br),
                    np.linalg.norm(tl - bl),
                ))

                if width < 50 or height < 50:
                    continue

                dst = np.array(
                    [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                    dtype=np.float32,
                )

                M = cv2.getPerspectiveTransform(pts, dst)
                corrected = cv2.warpPerspective(img, M, (width, height))
                logger.debug(
                    "auto_correct_perspective: warped %s → (%d,%d)",
                    img.shape,
                    width,
                    height,
                )
                return corrected, True

        logger.debug("auto_correct_perspective: no 4-corner contour found")
        return img, False

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order 4 corner points as [top-left, top-right, bottom-right, bottom-left]."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left
        rect[2] = pts[np.argmax(s)]   # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    # ------------------------------------------------------------------
    # SIDE DETECTION
    # ------------------------------------------------------------------

    def detect_side(
        self, img: np.ndarray
    ) -> tuple[str, float]:
        """
        Determine whether the Braille paper is viewed from the front
        (indentations) or back (raised bumps).

        Back-view images have higher edge sharpness (Laplacian variance)
        because raised bumps cast sharper shadows than front indentations.

        Args:
            img: Grayscale ndarray.

        Returns:
            Tuple of (side_string, confidence_float).
            side_string is 'back' or 'front'.
        """
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)

        # Compute local variance in 32×32 windows
        h, w = gray.shape
        window = 32
        variances: list[float] = []
        for y in range(0, h - window, window):
            for x in range(0, w - window, window):
                patch = lap[y : y + window, x : x + window]
                variances.append(float(np.var(patch)))

        mean_var = float(np.mean(variances)) if variances else 0.0
        confidence = min(1.0, mean_var / 1000.0)

        if mean_var > SIDE_VARIANCE_THRESHOLD:
            side = "back"
            logger.debug("detect_side: 'back' (var=%.1f)", mean_var)
            return "back", confidence

        side = "front"
        logger.debug("detect_side: 'front' (var=%.1f)", mean_var)
        return "front", 1.0 - confidence

    # ------------------------------------------------------------------
    # MIRROR
    # ------------------------------------------------------------------

    def mirror_if_back(self, img: np.ndarray, side: str) -> np.ndarray:
        """
        Horizontally flip the image if it was captured from the back.

        Braille read from the back is laterally inverted — mirroring
        restores it to the correct reading orientation.

        Args:
            img: ndarray (grayscale or BGR).
            side: 'back' or 'front'.

        Returns:
            Flipped image if back, unchanged otherwise.
        """
        if side == "back":
            mirrored = cv2.flip(img, 1)
            logger.debug("mirror_if_back: flipped (back view)")
            return mirrored
        logger.debug("mirror_if_back: no flip (front view)")
        return img

    # ------------------------------------------------------------------
    # QUALITY ASSESSMENT
    # ------------------------------------------------------------------

    def assess_quality(self, img: np.ndarray) -> dict:
        """
        Analyse image quality and return guidance for the camera operator.

        Checks brightness and blur to give actionable feedback.

        Args:
            img: Grayscale ndarray.

        Returns:
            Dict with brightness, blur_score, lighting, blur labels,
            and a guidance message string.
        """
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        lighting = (
            "low" if brightness < BRIGHTNESS_LOW_THRESHOLD
            else "bright" if brightness > BRIGHTNESS_HIGH_THRESHOLD
            else "good"
        )
        blur_label = "blurry" if blur_score < BLUR_THRESHOLD else "sharp"

        if lighting == "low":
            guidance = "Move to brighter area or add lighting 💡"
        elif lighting == "bright":
            guidance = "Reduce glare or lighting ☀️"
        elif blur_label == "blurry":
            guidance = "Hold camera steady 🤚"
        else:
            guidance = "Good positioning — scanning ✅"

        result = {
            "brightness": round(brightness, 1),
            "blur_score": round(blur_score, 1),
            "lighting": lighting,
            "blur": blur_label,
            "guidance": guidance,
            "ok": lighting == "good" and blur_label == "sharp",
        }
        logger.debug("assess_quality: %s", result)
        return result

    # ------------------------------------------------------------------
    # FULL PIPELINE
    # ------------------------------------------------------------------

    def full_pipeline(
        self, source: Union[str, bytes, np.ndarray]
    ) -> dict:
        """
        Run the complete preprocessing pipeline on an image.

        Steps:
            1. load_image
            2. assess_quality (on original)
            3. to_grayscale
            4. apply_clahe
            5. remove_shadows
            6. apply_threshold (otsu)
            7. auto_correct_perspective
            8. detect_side
            9. mirror_if_back

        Args:
            source: File path, bytes, or ndarray.

        Returns:
            Dict with all intermediate and final images plus metadata.
        """
        logger.info("full_pipeline: starting")

        # 1 – load
        original = self.load_image(source)

        # 2 – quality (on original colour image)
        gray_for_quality = self.to_grayscale(original)
        quality = self.assess_quality(gray_for_quality)

        # 3 – grayscale
        gray = gray_for_quality

        # 4 – shadow removal FIRST (normalises uneven lighting on raw grayscale)
        no_shadow = self.remove_shadows(gray)

        # 5 – CLAHE SECOND (enhance local contrast on normalised background)
        clahe_img = self.apply_clahe(no_shadow)

        # 6 – mild Gaussian denoising to smooth noise before detection
        denoised = cv2.GaussianBlur(clahe_img, (3, 3), 0)

        # 7 – perspective correction on the grayscale image
        #      (more edge information than binary — better contour detection)
        corrected_gray, was_perspective_corrected = self.auto_correct_perspective(denoised)

        # 8 – side detection (on CLAHE image for best accuracy)
        side, side_confidence = self.detect_side(clahe_img)

        # 9 – mirror if back-view
        processed = self.mirror_if_back(corrected_gray, side)

        # 10 – threshold (kept for display / debug only, NOT sent to detector)
        thresholded = self.apply_threshold(processed, method="otsu")

        logger.info(
            "full_pipeline: done. side=%s conf=%.2f perspective=%s quality=%s",
            side,
            side_confidence,
            was_perspective_corrected,
            quality["guidance"],
        )

        return {
            "processed": processed,          # Grayscale — sent to detector
            "thresholded": thresholded,       # Binary — for display/debug only
            "original": original,
            "grayscale": gray,
            "clahe": clahe_img,
            "no_shadow": no_shadow,
            "side": side,
            "side_confidence": round(side_confidence, 3),
            "was_perspective_corrected": was_perspective_corrected,
            "quality": quality,
            "guidance": quality["guidance"],
        }


# ─────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    prep = ImagePreprocessor()

    print("\n" + "=" * 50)
    print("  ImagePreprocessor Smoke Test")
    print("=" * 50)

    # Create a synthetic test image (white background, grey circles simulating Braille)
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 230
    for i in range(5):
        for j in range(8):
            cv2.circle(test_img, (80 + j * 70, 80 + i * 80), 12, (80, 80, 80), -1)

    print("  Created synthetic 640×480 test image with dot grid")

    # Test load from ndarray
    loaded = prep.load_image(test_img)
    print(f"  load_image (ndarray): shape={loaded.shape}  ✓")

    # Test grayscale
    gray = prep.to_grayscale(loaded)
    print(f"  to_grayscale: shape={gray.shape}  ✓")

    # Test CLAHE
    cl = prep.apply_clahe(gray)
    print(f"  apply_clahe: shape={cl.shape}  ✓")

    # Test shadow removal
    ns = prep.remove_shadows(gray)
    print(f"  remove_shadows: shape={ns.shape}  ✓")

    # Test threshold methods
    for m in ("otsu", "adaptive", "binary"):
        th = prep.apply_threshold(gray, method=m)
        print(f"  apply_threshold('{m}'): shape={th.shape}  ✓")

    # Test perspective correction
    corr, fixed = prep.auto_correct_perspective(test_img)
    print(f"  auto_correct_perspective: was_corrected={fixed}  ✓")

    # Test side detection
    side, conf = prep.detect_side(gray)
    print(f"  detect_side: side='{side}'  confidence={conf:.2f}  ✓")

    # Test mirror
    mirrored = prep.mirror_if_back(test_img, "back")
    print(f"  mirror_if_back('back'): shape={mirrored.shape}  ✓")

    # Test quality
    q = prep.assess_quality(gray)
    print(f"  assess_quality: {q}  ✓")

    # Full pipeline
    result = prep.full_pipeline(test_img)
    print(f"  full_pipeline: keys={list(result.keys())}  ✓")
    print(f"  guidance='{result['guidance']}'  side='{result['side']}'")

    # Test load from bytes
    _, buf = cv2.imencode(".jpg", test_img)
    loaded_bytes = prep.load_image(buf.tobytes())
    print(f"  load_image (bytes): shape={loaded_bytes.shape}  ✓")

    print("\n✅ Smoke test complete.\n")
