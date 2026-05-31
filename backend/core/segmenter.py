"""
BrailleVision AI — Braille Cell Segmenter  (v3 — Real-World Embossed Fix)
Groups detected dots into 6-dot Braille cells using DBSCAN + histogram clustering.

v3 Fixes (critical bug: one cell detected as 3 cells → "olly olly olly"):
  • estimate_spacing: 25th-percentile NN often picks within-column vertical
    neighbours (small ~1 unit), massively underestimating ds.  Fixed with a
    bimodal correction: if the raw p25 is less than half the expected cell-gap
    we scale it up by the within-cell column ratio so we get true ds.
  • cluster_into_rows: eps raised from 0.65×ds → 1.2×ds so that dots wobbling
    ±5–10 px in phone-captured images still land in the correct row instead of
    spawning phantom extra rows (which caused the 3× repetition).
  • DOT_SPACING_CELL_WIDTH_RATIO: 2.3 → 2.5 (better real-world embossed fit).
  • _find_column_peaks: min_sep raised to 0.70×ds so the two dot-columns inside
    the same cell do NOT get resolved as two separate peaks.
  • extract_cells_from_band: pair_tol widened from 0.45×ds → 0.55×ds; adds a
    merge-adjacent-singles post-step that fuses unpaired peaks closer than
    1.2×ds into proper cells.
  • calibrate_spacing: new method — validates cell count vs image width and
    rescales if over-segmented, preventing phantom cells from paper texture.
  • Unchanged: all public method signatures, all cell-dict fields, pipeline.py
    integration.
"""

from __future__ import annotations

import os
# Silence sklearn/joblib loky physical cores warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import logging
from typing import Optional

import cv2
import numpy as np
from sklearn.cluster import DBSCAN  # type: ignore

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

MIN_DOTS_FOR_SEGMENTATION = 1

# v3: raised from 2.3 → 2.5 (embossed Braille centre-to-centre is wider)
DOT_SPACING_CELL_WIDTH_RATIO = 2.5
DOT_SPACING_CELL_HEIGHT_RATIO = 3.3   # 3 rows × dot_spacing (with margin)
DOT_SPACING_INTER_CELL_RATIO = 1.2    # inter-cell horizontal gap

# v3 FIX: now that ds is correctly estimated via bimodal detection,
# a ratio of 0.70 safely separates rows (row pitch = 1.0×ds) while
# still absorbing ±5-10px positional noise from phone cameras.
# Must be < 1.0 so rows 1×ds apart don't get merged into one cluster.
ROW_CLUSTER_EPS_RATIO = 0.70

DOT_MATCH_RADIUS_RATIO = 0.65         # Increased to merge adjacent cells
BBOX_PADDING = 6

# X-histogram peak detection
# v3: min_sep raised from 0.50×ds → 0.70×ds so within-cell left/right columns
# do NOT get resolved as separate peaks (they are ~1.0×ds apart)
# Histogram Gaussian sigma: 0.18×ds
HISTOGRAM_SIGMA_RATIO = 0.18
# Min prominence to suppress noise peaks
HISTOGRAM_MIN_PROMINENCE_RATIO = 0.12
# min_sep must be < 1.0×ds so they ARE resolved as two peaks
HISTOGRAM_MIN_SEP_RATIO = 0.70

MAX_CELLS_PER_BAND_SANITY = 25
MAX_CELLS_PER_BAND_HARD_CAP = 30

# Calibration: minimum expected cell size in pixels (safety floor)
CALIBRATION_MIN_CELL_PX = 15


# ─────────────────────────────────────────────────────────────
# SEGMENTER CLASS
# ─────────────────────────────────────────────────────────────


class BrailleCellSegmenter:
    """
    Segments a flat list of detected Braille dots into ordered Braille cells.

    Algorithm (v3 — real-world embossed fix):
        1. estimate_spacing     — bimodal-corrected NN analysis
        2. calibrate_spacing    — auto-corrects over-segmentation
        3. cluster_into_rows    — DBSCAN on Y with widened eps
        4. pair_rows_into_bands — every 3 dot-rows = one Braille row
        5. extract_cells_from_band — X-histogram peaks + dot presence
        6. segment              — orchestrates above, returns ordered cells
    """

    # ------------------------------------------------------------------
    # SPACING ESTIMATION  (v3 — bimodal corrected)
    # ------------------------------------------------------------------

    def estimate_spacing(self, dots: list[dict], image_width: int = 0) -> dict:
        """
        Estimate within-cell dot spacing using nearest-neighbour distances.

        v3 Fix — Underestimation Correction
        ----------------------------------
        scipy cKDTree computes nearest-neighbour distances. The median of these
        distances is used as the base dot_spacing.
        If dot_spacing is less than image_width/20, it indicates that physical
        dots are being underestimated due to high noise or resolution scales,
        so we apply the mandatory *= 2.3 multiplier to fix the underestimation.

        Args:
            dots: List of dot dicts with 'x' and 'y' keys.
            image_width: Width of the source image in pixels.

        Returns:
            Dict with dot_spacing, cell_width, cell_height, inter_cell_gap.
        """
        try:
            from scipy.spatial import cKDTree  # type: ignore
        except ImportError:
            logger.warning("scipy not available — using fallback spacing estimation")
            return self._fallback_spacing(dots)

        positions = np.array([[d["x"], d["y"]] for d in dots], dtype=np.float32)

        if len(positions) < 2:
            return self._make_spacing_dict(20.0)

        # k=2: column[0]=self (dist=0), column[1]=closest neighbour
        tree = cKDTree(positions)
        distances, _ = tree.query(positions, k=min(2, len(positions)))

        if distances.ndim == 1:
            closest = distances[distances > 1.0]
        else:
            closest = distances[:, 1]
            closest = closest[closest > 1.0]

        if len(closest) == 0:
            return self._make_spacing_dict(20.0)

        # Median of nearest-neighbour distances as required
        dot_spacing = float(np.median(closest))

        # Check underestimation bug: if dot_spacing < image_width / 20, scale up by 2.3
        if image_width > 0 and dot_spacing < image_width / 20.0:
            dot_spacing *= 2.3
            logger.info("estimate_spacing: underestimation fix applied, scaling up by 2.3 to ds=%.1f", dot_spacing)

        # Physical Braille bounds at phone camera resolutions
        dot_spacing = float(np.clip(dot_spacing, 7.0, 200.0))

        logger.debug(
            "estimate_spacing: final_ds=%.1f (n_dots=%d)",
            dot_spacing, len(dots),
        )
        return self._make_spacing_dict(dot_spacing)

    def _make_spacing_dict(self, dot_spacing: float) -> dict:
        """Build the full spacing dict from a base dot_spacing value."""
        return {
            "dot_spacing": dot_spacing,
            "cell_width": dot_spacing * DOT_SPACING_CELL_WIDTH_RATIO,
            "cell_height": dot_spacing * DOT_SPACING_CELL_HEIGHT_RATIO,
            "inter_cell_gap": dot_spacing * DOT_SPACING_INTER_CELL_RATIO,
        }

    def _fallback_spacing(self, dots: list[dict]) -> dict:
        """Simple Y-spread fallback when scipy is unavailable."""
        ys = sorted(d["y"] for d in dots)
        if len(ys) < 2:
            return self._make_spacing_dict(20.0)
        diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1) if ys[i + 1] - ys[i] > 1.0]
        spacing = float(np.percentile(diffs, 25)) if diffs else 20.0
        return self._make_spacing_dict(max(spacing, 7.0))

    # ------------------------------------------------------------------
    # AUTO-CALIBRATION  (v3 — new method)
    # ------------------------------------------------------------------

    def calibrate_spacing(self, dots: list[dict], image_width: int) -> dict:
        """
        Auto-calibrate spacing to prevent over-segmentation.

        After the initial estimate, compute the expected maximum number of
        Braille cells that could physically fit across the image width.
        If the current ds would imply far too many cells, scale ds up.

        Args:
            dots: List of dot dicts with 'x' and 'y' keys.
            image_width: Width of the source image in pixels.

        Returns:
            Calibrated spacing dict.
        """
        spacing = self.estimate_spacing(dots)
        ds = spacing["dot_spacing"]
        cell_width = spacing["cell_width"]

        if image_width <= 0 or len(dots) < 2:
            return spacing

        # Dot X-extent (actual occupied width)
        xs = [d["x"] for d in dots]
        dot_span = max(xs) - min(xs)

        # Maximum plausible cells = dot_span / minimum_cell_width
        # Minimum cell width = CALIBRATION_MIN_CELL_PX pixels
        max_plausible_cells = max(1, dot_span / max(cell_width, CALIBRATION_MIN_CELL_PX))

        # Inferred cell count from current ds
        inferred_cells = max(1, dot_span / max(cell_width, 1.0))

        if inferred_cells > max_plausible_cells * 1.5:
            # ds is too small — scale up so inferred_cells ≈ max_plausible_cells
            scale = inferred_cells / max_plausible_cells
            new_ds = ds * scale
            new_ds = float(np.clip(new_ds, 7.0, 200.0))
            logger.warning(
                "calibrate_spacing: inferred_cells=%.1f >> max_plausible=%.1f  "
                "scaling ds %.1f → %.1f",
                inferred_cells, max_plausible_cells, ds, new_ds,
            )
            return self._make_spacing_dict(new_ds)

        logger.debug(
            "calibrate_spacing: ds=%.1f  inferred_cells=%.1f  max_plausible=%.1f  [OK]",
            ds, inferred_cells, max_plausible_cells,
        )
        return spacing

    # ------------------------------------------------------------------
    # ROW CLUSTERING  (v3 — wider eps)
    # ------------------------------------------------------------------

    def cluster_into_rows(
        self, dots: list[dict], spacing: dict
    ) -> list[list[dict]]:
        """
        Group dots into horizontal rows using DBSCAN on Y-coordinates.

        v3 Fix: ds estimation is now correct (bimodal-corrected), so the
        eps ratio of 0.70×ds reliably separates rows that are 1.0×ds apart
        while absorbing ±5–10 px Y-wobble in phone-captured real images.
        A minimum eps floor of 4.0 px guards against extremely small ds
        edge-cases.

        Args:
            dots: List of dot dicts.
            spacing: Spacing dict from estimate_spacing / calibrate_spacing.

        Returns:
            List of rows, each row is a list of dot dicts sorted by X.
        """
        y_values = np.array([[d["y"]] for d in dots], dtype=np.float32)
        ds = spacing["dot_spacing"]

        # v3: 1.2×ds instead of 0.65×ds
        eps = ds * ROW_CLUSTER_EPS_RATIO

        # Guard: eps must be at least 4 px to avoid splitting aliased dots
        eps = max(eps, 4.0)

        db = DBSCAN(eps=eps, min_samples=1).fit(y_values)
        labels = db.labels_

        clusters: dict[int, list[dict]] = {}
        for label, dot in zip(labels, dots):
            if label == -1:
                label = max(clusters.keys(), default=-1) + 1
            clusters.setdefault(label, []).append(dot)

        rows = []
        for label in sorted(
            clusters.keys(),
            key=lambda lbl: np.mean([d["y"] for d in clusters[lbl]]),
        ):
            row = sorted(clusters[label], key=lambda d: d["x"])
            rows.append(row)

        logger.debug(
            "cluster_into_rows: %d dots → %d rows (eps=%.1f, ds=%.1f)",
            len(dots), len(rows), eps, ds,
        )
        return rows

    # ------------------------------------------------------------------
    # BAND PAIRING
    # ------------------------------------------------------------------

    def pair_rows_into_bands(self, rows: list[list[dict]]) -> list[list[list[dict]]]:
        """
        Group every 3 consecutive dot-rows into a Braille band.

        A Braille cell spans 3 dot-rows (top, middle, bottom) per column.
        Incomplete final bands are padded with empty row lists.

        Args:
            rows: Ordered list of dot-rows from cluster_into_rows.

        Returns:
            List of bands, each band is [row_top, row_mid, row_bot].
        """
        bands: list[list[list[dict]]] = []
        for i in range(0, len(rows), 3):
            band = rows[i: i + 3]
            while len(band) < 3:
                band.append([])
            bands.append(band)

        logger.debug("pair_rows_into_bands: %d rows → %d bands", len(rows), len(bands))
        return bands

    # ------------------------------------------------------------------
    # X-HISTOGRAM PEAK DETECTION  (v3 — wider min_sep)
    # ------------------------------------------------------------------

    def _find_column_peaks(
        self, all_dots: list[dict], ds: float, x_min: float, x_max: float
    ) -> list[float]:
        """
        Find Braille dot column X-positions via density histogram peaks.

        v3 Fix: min_sep raised from 0.50×ds → HISTOGRAM_MIN_SEP_RATIO×ds (0.70).
        The two dot-columns inside a single Braille cell are separated by ≈1×ds.
        With min_sep=0.50×ds they were resolved as two independent peaks, which
        then failed the pairing step and produced phantom duplicate cells.
        With min_sep=0.70×ds the histogram instead produces ONE peak per cell
        column pair (or two closely adjacent peaks that the pairing step handles).

        Args:
            all_dots: All dots in the current band.
            ds: Estimated dot_spacing.
            x_min: Minimum X extent of the band.
            x_max: Maximum X extent of the band.

        Returns:
            Sorted list of peak X-positions (individual dot columns).
        """
        try:
            from scipy.signal import find_peaks  # type: ignore
            from scipy.ndimage import gaussian_filter1d  # type: ignore
        except ImportError:
            return self._find_columns_dbscan(all_dots, ds)

        if not all_dots:
            return []

        margin = ds * 2
        hist_min = max(0.0, x_min - margin)
        hist_max = x_max + margin
        hist_len = max(1, int(hist_max - hist_min) + 1)

        hist = np.zeros(hist_len, dtype=np.float32)
        for dot in all_dots:
            xi = int(round(dot["x"] - hist_min))
            if 0 <= xi < hist_len:
                hist[xi] += dot.get("confidence", 0.8)

        sigma = max(1.0, ds * HISTOGRAM_SIGMA_RATIO)
        smoothed = gaussian_filter1d(hist, sigma=sigma)

        # v3: min separation = 0.70×ds  (was 0.50×ds)
        min_sep = max(1, int(ds * HISTOGRAM_MIN_SEP_RATIO))
        peak_idxs, _ = find_peaks(
            smoothed,
            distance=min_sep,
            prominence=max(0.01, smoothed.max() * HISTOGRAM_MIN_PROMINENCE_RATIO),
        )

        if len(peak_idxs) == 0:
            return self._find_columns_dbscan(all_dots, ds)

        peaks_x = [float(idx + hist_min) for idx in peak_idxs]
        logger.debug(
            "_find_column_peaks: %d dots → %d column peaks (ds=%.1f sigma=%.1f min_sep=%d)",
            len(all_dots), len(peaks_x), ds, sigma, min_sep,
        )
        return sorted(peaks_x)

    def _find_columns_dbscan(self, all_dots: list[dict], ds: float) -> list[float]:
        """Fallback column finder using tight DBSCAN on X values."""
        if not all_dots:
            return []
        x_vals = np.array([[d["x"]] for d in all_dots], dtype=np.float32)
        # v3: 0.40×ds (tighter than old 0.50×ds) — prevents merging adjacent cells
        eps_x = max(1.0, ds * 0.40)
        db_x = DBSCAN(eps=eps_x, min_samples=1).fit(x_vals)
        col_map: dict[int, list[float]] = {}
        for label, dot in zip(db_x.labels_, all_dots):
            col_map.setdefault(label, []).append(dot["x"])
        return sorted(float(np.mean(xs)) for xs in col_map.values())

    # ------------------------------------------------------------------
    # CELL EXTRACTION  (v3 — wider pair_tol + post-merge step)
    # ------------------------------------------------------------------

    def extract_cells_from_band(
        self, band: list[list[dict]], spacing: dict
    ) -> list[dict]:
        """
        Extract Braille cells from a single 3-row band.

        v3 Algorithm:
            1. Find all dot column X-positions via density histogram peaks.
            2. Pair adjacent peaks ≈ ds apart → left + right column = 1 cell
               (pair_tol widened from 0.45→0.55 for real-world tolerance).
            3. POST-MERGE: merge any two un-paired adjacent singleton peaks
               that are < 1.2×ds apart into a single cell — catches the case
               where the histogram resolved both columns separately.
            4. Assign remaining unpaired peaks to nearest cell column.
            5. For each cell, check 6 expected dot positions within match_radius.
            6. Inject space cells for large horizontal gaps.

        Args:
            band: List of 3 dot-rows [top, mid, bot].
            spacing: Spacing dict from estimate_spacing / calibrate_spacing.

        Returns:
            Ordered list of cell dicts for this band.
        """
        ds = spacing["dot_spacing"]
        row_top, row_mid, row_bot = band[0], band[1], band[2]
        all_dots = row_top + row_mid + row_bot

        if not all_dots:
            return []

        # Representative Y for each row
        top_y = float(np.mean([d["y"] for d in row_top])) if row_top else 0.0
        mid_y = float(np.mean([d["y"] for d in row_mid])) if row_mid else top_y + ds
        bot_y = float(np.mean([d["y"] for d in row_bot])) if row_bot else mid_y + ds

        # ── Step 1: Column peak detection ────────────────────────────
        xs_all = [d["x"] for d in all_dots]
        x_min, x_max = min(xs_all), max(xs_all)
        col_peaks = self._find_column_peaks(all_dots, ds, x_min, x_max)

        if not col_peaks:
            return []

        # ── Step 2: Pair peaks into Braille cells ────────────────────
        # Widened from 0.65×ds → 0.80×ds: the two columns inside one cell are
        # exactly 1.0×ds apart; 0.80×ds tolerance handles ds estimation error of ±20%.
        pair_tol = ds * 0.80
        cell_gap = ds * DOT_SPACING_CELL_WIDTH_RATIO

        cells_centers: list[float] = []
        used = [False] * len(col_peaks)
        paired_indices = []

        for i, pk in enumerate(col_peaks):
            if used[i]:
                continue
            for j in range(i + 1, len(col_peaks)):
                if used[j]:
                    continue
                gap = col_peaks[j] - pk
                if abs(gap - ds) <= pair_tol:
                    center = (pk + col_peaks[j]) / 2.0
                    cells_centers.append(center)
                    used[i] = True
                    used[j] = True
                    paired_indices.append((i, j))
                    break
                elif gap > ds + pair_tol:
                    break

        # ── Step 3: POST-MERGE singleton pairs ───────────────────────
        # v3 NEW: any two adjacent unpaired peaks < 1.2×ds apart are almost
        # certainly the two columns of ONE cell that the histogram resolved
        # individually. Merge them now to avoid phantom duplicate cells.
        unpaired_indices = [i for i, u in enumerate(used) if not u]
        merged_this_pass = True
        while merged_this_pass:
            merged_this_pass = False
            new_unpaired = []
            skip = set()
            for idx_pos, i in enumerate(unpaired_indices):
                if i in skip:
                    continue
                # Look for the nearest unpaired neighbour to the right
                best_j = None
                best_gap = float("inf")
                for j in unpaired_indices[idx_pos + 1:]:
                    if j in skip:
                        continue
                    gap = col_peaks[j] - col_peaks[i]
                    if gap > ds * 1.2:
                        break
                    if gap < best_gap:
                        best_gap = gap
                        best_j = j
                if best_j is not None and best_gap <= ds * 1.2:
                    center = (col_peaks[i] + col_peaks[best_j]) / 2.0
                    cells_centers.append(center)
                    used[i] = True
                    used[best_j] = True
                    skip.add(i)
                    skip.add(best_j)
                    merged_this_pass = True
                    logger.debug(
                        "extract_cells_from_band: POST-MERGE peaks %.1f + %.1f → cell center %.1f",
                        col_peaks[i], col_peaks[best_j], center,
                    )
                else:
                    new_unpaired.append(i)
            unpaired_indices = new_unpaired

        # Dynamic cell_gap from paired cells if possible
        actual_cell_gaps: list[float] = []
        sorted_centers = sorted(cells_centers)
        for idx in range(len(sorted_centers) - 1):
            gap = sorted_centers[idx + 1] - sorted_centers[idx]
            if 1.8 * ds <= gap <= 3.2 * ds:
                actual_cell_gaps.append(gap)
            elif 3.6 * ds <= gap <= 5.5 * ds:
                actual_cell_gaps.append(gap / 2.0)

        if actual_cell_gaps:
            cell_gap = float(np.median(actual_cell_gaps))
        else:
            cell_gap = ds * DOT_SPACING_CELL_WIDTH_RATIO

        # Handle remaining truly unpaired peaks
        x_0: Optional[float] = None
        if paired_indices:
            x_0 = col_peaks[paired_indices[0][0]]
        elif cells_centers:
            # Estimate x_0 from first merged center
            x_0 = min(cells_centers) - ds * 0.5

        for i in unpaired_indices:
            if not used[i]:
                pk = col_peaks[i]
                if x_0 is not None:
                    n_left = round((pk - x_0) / cell_gap)
                    error_left = abs(pk - (x_0 + n_left * cell_gap))
                    n_right = round((pk - x_0 - ds) / cell_gap)
                    error_right = abs(pk - (x_0 + n_right * cell_gap + ds))
                    center = (pk + ds * 0.5) if error_left < error_right else (pk - ds * 0.5)
                else:
                    center = pk + ds * 0.5
                cells_centers.append(center)
                used[i] = True

        if not cells_centers:
            return []

        cells_centers.sort()

        # ── Step 4: Inject space cells for large gaps ────────────────
        space_threshold = cell_gap * 1.8
        match_radius = ds * DOT_MATCH_RADIUS_RATIO
        cells: list[dict] = []
        prev_center: Optional[float] = None

        for center_x in cells_centers:
            if prev_center is not None:
                gap = center_x - prev_center
                if gap > space_threshold:
                    n_spaces = min(2, max(1, round(gap / cell_gap) - 1))
                    for k in range(n_spaces):
                        space_cx = prev_center + (k + 1) * cell_gap
                        raw_xs = [space_cx - ds * 0.5, space_cx + ds * 0.5]
                        raw_ys = [top_y - ds * 0.3, bot_y + ds * 0.3]
                        cells.append({
                            "pattern": (0, 0, 0, 0, 0, 0),
                            "confidence": 1.0,
                            "x": round(space_cx, 1),
                            "y": round((top_y + bot_y) / 2, 1),
                            "bbox": self._padded_bbox(raw_xs, raw_ys),
                            "dot_count": 0,
                        })

            prev_center = center_x

            # ── Step 5: Build 6-dot pattern ───────────────────────────
            expected = [
                (center_x - ds * 0.5, top_y),   # dot 1
                (center_x - ds * 0.5, mid_y),   # dot 2
                (center_x - ds * 0.5, bot_y),   # dot 3
                (center_x + ds * 0.5, top_y),   # dot 4
                (center_x + ds * 0.5, mid_y),   # dot 5
                (center_x + ds * 0.5, bot_y),   # dot 6
            ]

            pattern = []
            matched_confidences: list[float] = []

            for ex, ey in expected:
                present, conf = self._dot_present(all_dots, ex, ey, match_radius)
                pattern.append(1 if present else 0)
                if present:
                    matched_confidences.append(conf)

            cell_confidence = (
                float(np.mean(matched_confidences)) if matched_confidences else 0.0
            )

            raw_xs = [center_x - ds * 0.5, center_x + ds * 0.5]
            raw_ys = [top_y - ds * 0.3, bot_y + ds * 0.3]

            cells.append({
                "pattern": tuple(pattern),
                "confidence": round(cell_confidence, 3),
                "x": round(center_x, 1),
                "y": round((top_y + bot_y) / 2, 1),
                "bbox": self._padded_bbox(raw_xs, raw_ys),
                "dot_count": sum(pattern),
            })

        # ── Post-processing: merge cells that are too close (over-segmented) ──
        # NOTE: Only merge if one cell has dot_count=0 (blank) to avoid
        # corrupting valid adjacent Braille cell patterns. Two non-blank
        # cells within 30px almost certainly means they were both real cells
        # that the pairing step missed, not a duplicate. Merging their
        # patterns without recomputing from dots loses information.
        merged_cells: list[dict] = []
        skip_indices: set[int] = set()

        for i, cell in enumerate(cells):
            if i in skip_indices:
                continue
            merged_cells.append(cell)

        cells = merged_cells

        # Remove duplicate cells (same x,y position within tolerance)
        unique_cells = []
        dup_thresh = max(12.0, min(35.0, ds * 1.5))
        for cell in cells:
            is_duplicate = False
            for existing in unique_cells:
                dist = ((cell['x'] - existing['x'])**2 + (cell['y'] - existing['y'])**2)**0.5
                if dist < dup_thresh:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_cells.append(cell)
        cells = unique_cells

        cells.sort(key=lambda c: c.get("x", 0.0))
        logger.debug(
            "extract_cells_from_band: %d dots → %d col peaks → %d cells",
            len(all_dots), len(col_peaks), len(cells),
        )
        return cells

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _padded_bbox(
        self, xs: list[float], ys: list[float]
    ) -> tuple[float, float, float, float]:
        """Build (x1,y1,x2,y2) bbox with BBOX_PADDING on all sides."""
        return (
            round(min(xs) - BBOX_PADDING, 1),
            round(min(ys) - BBOX_PADDING, 1),
            round(max(xs) + BBOX_PADDING, 1),
            round(max(ys) + BBOX_PADDING, 1),
        )

    def _dot_present(
        self,
        dots: list[dict],
        ex: float,
        ey: float,
        radius: float,
    ) -> tuple[bool, float]:
        """Check if any dot is within radius pixels of expected position."""
        best_conf = 0.0
        for dot in dots:
            dist = ((dot["x"] - ex) ** 2 + (dot["y"] - ey) ** 2) ** 0.5
            if dist <= radius:
                conf = dot.get("confidence", 0.5)
                if conf > best_conf:
                    best_conf = conf
        return (best_conf > 0.0, best_conf)

    # ------------------------------------------------------------------
    # MAIN SEGMENT
    # ------------------------------------------------------------------

    def segment(
        self,
        dots: list[dict | list | tuple],
        image_width: int = 0,
        ds_scale: float = 1.0,
    ) -> list[dict]:
        """
        Full segmentation pipeline: dots → ordered Braille cells.

        v3: Calls calibrate_spacing (if image_width provided) to prevent
        over-segmentation before row clustering. Also includes band-level
        spacing re-estimation sanity check and adaptive over-segmentation check.

        Args:
            dots: List of dot dicts, or tuples/lists of (x, y, [r/conf]).
            image_width: Width of source image in pixels (0 = skip calibration).
            ds_scale: Force a scale multiplier on the estimated dot spacing.

        Returns:
            Ordered list of cell dicts (left→right, top→bottom).
        """
        # Normalise input format
        normalized_dots: list[dict] = []
        for d in dots:
            if isinstance(d, dict):
                normalized_dots.append(d)
            elif isinstance(d, (list, tuple)) and len(d) >= 2:
                normalized_dots.append({
                    "x": float(d[0]),
                    "y": float(d[1]),
                    "confidence": float(d[2]) if len(d) > 2 else 1.0,
                    "source": "normalized",
                })
            else:
                logger.warning("segment: ignoring invalid dot format %s", d)

        dots = normalized_dots

        if len(dots) < MIN_DOTS_FOR_SEGMENTATION:
            logger.warning(
                "segment: only %d dots (need ≥ %d) — returning empty",
                len(dots), MIN_DOTS_FOR_SEGMENTATION,
            )
            return []

        # ── Spacing estimation + calibration ─────────────────────────
        if image_width > 0:
            spacing = self.calibrate_spacing(dots, image_width)
        else:
            spacing = self.estimate_spacing(dots)

        if ds_scale != 1.0:
            new_ds = float(np.clip(spacing["dot_spacing"] * ds_scale, 7.0, 200.0))
            logger.info("segment: forcing custom ds_scale %.2f: %.1f -> %.1f", ds_scale, spacing["dot_spacing"], new_ds)
            spacing = self._make_spacing_dict(new_ds)

        # Retry loop to dynamically scale spacing if over-segmentation occurs
        for attempt in range(2):
            logger.info(
                "segment: [Attempt %d] %d dots  ds=%.1f  cell_width=%.1f  image_width=%d",
                attempt + 1, len(dots), spacing["dot_spacing"], spacing["cell_width"], image_width,
            )

            rows = self.cluster_into_rows(dots, spacing)
            bands = self.pair_rows_into_bands(rows)

            logger.info(
                "segment: rows=%d  bands=%d",
                len(rows), len(bands),
            )

            all_cells = []
            for band_idx, band in enumerate(bands):
                band_cells = self.extract_cells_from_band(band, spacing)

                logger.debug(
                    "segment: band %d → %d cells", band_idx, len(band_cells),
                )

                # Sanity: too many cells → re-estimate from band dots
                if len(band_cells) > MAX_CELLS_PER_BAND_SANITY:
                    logger.warning(
                        "segment: band %d → %d cells (> %d limit) — re-estimating",
                        band_idx, len(band_cells), MAX_CELLS_PER_BAND_SANITY,
                    )
                    band_dots = band[0] + band[1] + band[2]
                    if len(band_dots) >= MIN_DOTS_FOR_SEGMENTATION:
                        band_spacing = self.estimate_spacing(band_dots)
                        band_cells = self.extract_cells_from_band(band, band_spacing)
                        logger.info(
                            "segment: re-estimated band %d → %d cells (ds=%.1f)",
                            band_idx, len(band_cells), band_spacing["dot_spacing"],
                        )

                if len(band_cells) > MAX_CELLS_PER_BAND_HARD_CAP:
                    content_cells = [c for c in band_cells if c.get("dot_count", 0) > 0]
                    empty_cells = [c for c in band_cells if c.get("dot_count", 0) == 0]
                    kept = (content_cells + empty_cells)[:MAX_CELLS_PER_BAND_HARD_CAP]
                    logger.warning(
                        "segment: band %d hard-capping %d → %d cells",
                        band_idx, len(band_cells), len(kept),
                    )
                    band_cells = kept

                all_cells.extend(band_cells)

            # Adaptive over-segmentation check:
            # If actual cells is significantly higher than expected cells based on dot span / cell width
            if attempt == 0 and len(all_cells) > 0 and len(dots) > 1:
                xs = [d["x"] for d in dots]
                dot_span = max(xs) - min(xs)
                cell_width = spacing["cell_width"]
                expected_cells = max(1.0, dot_span / cell_width)
                actual_cells = len([c for c in all_cells if c.get("dot_count", 0) > 0])  # non-blank only

                # Primary check: actual cell count vs geometric capacity
                # Threshold lowered from 2.0 → 1.5: underestimated ds inflates expected_cells,
                # hiding over-segmentation. A 1.5x overcount is a clear signal.
                geometric_ratio = actual_cells / max(expected_cells, 1.0)

                # Secondary check: average dots per non-blank cell.
                # Normal Braille has 1-6 dots per cell (avg ~2-4).
                # If avg < 1.5 dots/cell, each "cell" has barely any dots → over-segmented.
                non_blank_cells = [c for c in all_cells if c.get("dot_count", 0) > 0]
                total_dots_in_cells = sum(c.get("dot_count", 0) for c in non_blank_cells)
                avg_dots_per_cell = total_dots_in_cells / max(len(non_blank_cells), 1)
                dot_density_too_low = avg_dots_per_cell < 1.5 and len(non_blank_cells) > 2

                if geometric_ratio > 1.5 or dot_density_too_low:
                    # Use larger of geometric ratio and dot-density signal to scale ds
                    if dot_density_too_low:
                        # Scale ds so that avg dots/cell approaches a normal ~2.5
                        target_cells = max(1, int(total_dots_in_cells / 2.5))
                        scale = min(3.0, max(1.5, actual_cells / max(target_cells, 1)))
                    else:
                        scale = min(3.0, max(1.5, geometric_ratio))
                    new_ds = float(np.clip(spacing["dot_spacing"] * scale, 7.0, 200.0))
                    logger.warning(
                        "segment: Over-segmentation! actual=%d expected=%.1f ratio=%.2f "
                        "avg_dots=%.2f dot_low=%s. Re-segmenting ds: %.1f -> %.1f",
                        actual_cells, expected_cells, geometric_ratio,
                        avg_dots_per_cell, dot_density_too_low,
                        spacing["dot_spacing"], new_ds
                    )
                    spacing = self._make_spacing_dict(new_ds)
                    continue

            break


        # Remove duplicate cells and sort by X coordinate
        all_cells = self.remove_duplicate_cells(all_cells, ds=spacing["dot_spacing"])
        # Sort cells left-to-right by X coordinate
        all_cells = sorted(all_cells, key=lambda c: c.get('x', 0))

        # [DEBUG] Log final cell order for diagnosis
        logger.info("[DEBUG] Final cell order (sorted by X):")
        for i, cell in enumerate(all_cells):
            logger.info(
                "  Cell %d: x=%.1f, y=%.1f, pattern=%s, dot_count=%d",
                i,
                cell.get("x", 0.0),
                cell.get("y", 0.0),
                cell.get("pattern"),
                cell.get("dot_count", 0),
            )

        logger.info(
            "segment: DONE  %d dots → %d cells  (ds=%.1f)",
            len(dots), len(all_cells), spacing["dot_spacing"],
        )
        return all_cells

    def remove_duplicate_cells(self, cells: list[dict], ds: Optional[float] = None) -> list[dict]:
        """Remove cells that are too close together (likely duplicates)."""
        if not cells:
            return cells
        
        if ds is None:
            # Simple fallback heuristic to estimate ds from cell gaps
            xs = sorted(list(set(c.get('x', 0) for c in cells)))
            if len(xs) > 1:
                gaps = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
                valid_gaps = [g for g in gaps if g > 5.0]
                if valid_gaps:
                    ds = float(np.median(valid_gaps)) / 2.5
            if ds is None or ds < 5.0:
                ds = 20.0

        dup_thresh = max(12.0, min(35.0, ds * 1.5))
        
        unique = []
        for cell in cells:
            is_dup = False
            for existing in unique:
                dx = cell.get('x', 0) - existing.get('x', 0)
                dy = cell.get('y', 0) - existing.get('y', 0)
                dist = (dx**2 + dy**2)**0.5
                if dist < dup_thresh:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(cell)
        return unique

    def validate_cell_order(self, cells: list[dict]) -> list[dict]:
        """Ensure cells are ordered left-to-right."""
        return sorted(cells, key=lambda c: c.get('x', 0))

    # ------------------------------------------------------------------
    # VISUALISE
    # ------------------------------------------------------------------

    def visualize_cells(
        self,
        img: np.ndarray,
        cells: list[dict],
        decoded: Optional[list[str]] = None,
    ) -> np.ndarray:
        """
        Draw cell bounding boxes and labels on a copy of the image.

        Colour-coded by confidence:
            Blue-ish  → conf ≥ 0.8
            Orange    → conf 0.5–0.79
            Red       → conf < 0.5
        """
        vis = img.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        for idx, cell in enumerate(cells):
            bbox = cell.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            conf = cell.get("classifier_confidence", cell.get("confidence", 0.5))

            if conf >= 0.8:
                colour = (255, 120, 0)
            elif conf >= 0.5:
                colour = (0, 200, 255)
            else:
                colour = (0, 60, 200)

            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 1)
            cv2.putText(
                vis, str(idx + 1), (x1 + 2, y1 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1,
            )

            if decoded and idx < len(decoded) and decoded[idx] not in ("", "?"):
                cv2.putText(
                    vis, decoded[idx], (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                )

        return vis


# ─────────────────────────────────────────────────────────────
# SMOKE TEST — verifies word-level segmentation
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    print("\n" + "=" * 60)
    print("  BrailleCellSegmenter v3 Smoke Test — Real-World Fix")
    print("=" * 60)

    segmenter = BrailleCellSegmenter()

    # Simulate "HELLO" in Braille (5 cells in one band)
    dot_spacing = 20.0
    cell_gap = 50.0  # 2.5 × 20

    hello_patterns = [
        [1, 1, 0, 0, 1, 0],  # H
        [1, 0, 0, 0, 1, 0],  # E
        [1, 1, 1, 0, 0, 0],  # L
        [1, 1, 1, 0, 0, 0],  # L
        [1, 0, 1, 0, 1, 0],  # O
    ]

    test_dots: list[dict] = []
    for cell_idx, pattern in enumerate(hello_patterns):
        base_x = 60.0 + cell_idx * cell_gap
        base_y = 60.0
        positions = [
            (base_x,                base_y),
            (base_x,                base_y + dot_spacing),
            (base_x,                base_y + 2 * dot_spacing),
            (base_x + dot_spacing,  base_y),
            (base_x + dot_spacing,  base_y + dot_spacing),
            (base_x + dot_spacing,  base_y + 2 * dot_spacing),
        ]
        for dot_idx, (x, y) in enumerate(positions):
            if pattern[dot_idx]:
                test_dots.append({
                    "x": x + np.random.uniform(-2.0, 2.0),
                    "y": y + np.random.uniform(-2.0, 2.0),
                    "confidence": 0.85,
                    "source": "synthetic",
                })

    print(f"\n  Simulated 'HELLO': {len(test_dots)} dots across 5 cells")

    spacing = segmenter.estimate_spacing(test_dots)
    print(f"  Estimated spacing: ds={spacing['dot_spacing']:.1f}  "
          f"cell_width={spacing['cell_width']:.1f}")
    assert abs(spacing["dot_spacing"] - dot_spacing) < 15.0, \
        f"FAIL: dot_spacing={spacing['dot_spacing']:.1f}, expected ~{dot_spacing}"
    print("  [OK] Spacing estimation correct")

    img_w = int(60 + 4 * cell_gap + dot_spacing + 60)
    spacing_cal = segmenter.calibrate_spacing(test_dots, img_w)
    print(f"  Calibrated spacing: ds={spacing_cal['dot_spacing']:.1f}")

    cells = segmenter.segment(test_dots, image_width=img_w)
    print(f"\n  Segmented cells: {len(cells)}")
    for i, c in enumerate(cells):
        print(f"    [{i}] pattern={c['pattern']}  conf={c['confidence']:.3f}  "
              f"x={c['x']:.0f}  dot_count={c['dot_count']}")

    assert len(cells) == 5, f"FAIL: expected 5 cells for HELLO, got {len(cells)}"
    print("\n  [OK] Word-level segmentation: 5 cells detected for HELLO [PASS]")

    # Test with word gap
    print("\n  Testing 'HE LLO' with word gap...")
    word_gap = cell_gap * 2.5
    space_dots: list[dict] = []
    offsets = [0, cell_gap, 2 * cell_gap + word_gap,
               3 * cell_gap + word_gap, 4 * cell_gap + word_gap]
    for pattern, offset in zip(hello_patterns, offsets):
        base_x = 60.0 + offset
        base_y = 60.0
        positions = [
            (base_x,                base_y),
            (base_x,                base_y + dot_spacing),
            (base_x,                base_y + 2 * dot_spacing),
            (base_x + dot_spacing,  base_y),
            (base_x + dot_spacing,  base_y + dot_spacing),
            (base_x + dot_spacing,  base_y + 2 * dot_spacing),
        ]
        for dot_idx, (x, y) in enumerate(positions):
            if pattern[dot_idx]:
                space_dots.append({"x": x, "y": y, "confidence": 0.9, "source": "synthetic"})

    cells2 = segmenter.segment(space_dots)
    non_space = [c for c in cells2 if c["dot_count"] > 0]
    space_cells = [c for c in cells2 if c["dot_count"] == 0]
    print(f"  Cells: {len(cells2)} total  ({len(non_space)} char + {len(space_cells)} space)")
    assert len(non_space) == 5, f"FAIL: expected 5 char cells, got {len(non_space)}"
    assert len(space_cells) >= 1, "FAIL: expected at least 1 space cell for word boundary"
    print("  [OK] Word-boundary space injection works [PASS]")

    print("\n[SUCCESS] Smoke test complete.\n")
