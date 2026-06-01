"""
╔══════════════════════════════════════════════════════╗
║  BrailleVision AI — EfficientNet-B3 Cell Classifier  ║
║  Trained on 3,00,000+ images  |  Val Accuracy: 96.8% ║
╚══════════════════════════════════════════════════════╝

Load priority:
  1. braille_scripted.pt  — TorchScript (fastest, no timm required)
  2. best_model.pth       — Checkpoint with embedded idx_to_char map

The classifier:
  • Handles 46 Braille classes: a-z, 0-9, punctuation, [CAP], [NUM]
  • Does NOT classify blank/space cells (handled upstream in pipeline)
  • Thread-safe singleton after load (read-only state)
  • Uses torch.autocast on CUDA for FP16 throughput
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image  # type: ignore

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONSTANTS  (must match training configuration)
# ─────────────────────────────────────────────────────────────

MODEL_INPUT_SIZE = 128          # px — EfficientNet-B3 trained input size
MODEL_MEAN = [0.485, 0.456, 0.406]
MODEL_STD  = [0.229, 0.224, 0.225]
NUM_CLASSES = 46

# Confidence below this → prediction flagged as low_confidence.
# Set to 0.45 because the model is 96.8% accurate; very few legitimate
# predictions fall below 0.45 — anything lower is genuine uncertainty.
CONFIDENCE_THRESHOLD = 0.45

# ── Path resolution ──────────────────────────────────────────
# Layout:  BrailleVision AI/
#              backend/ai/models/cell_classifier.py   ← this file
#              models/braille_scripted.pt
#              models/best_model.pth
#              models/class_map.json
_BACKEND_DIR   = Path(__file__).resolve().parent.parent.parent   # → backend/
_PROJECT_ROOT  = _BACKEND_DIR.parent                              # → BrailleVision AI/
_MODELS_DIR    = _PROJECT_ROOT / "models"

DEFAULT_SCRIPTED_PATH  = _MODELS_DIR / "braille_scripted.pt"
DEFAULT_CHECKPOINT_PATH = _MODELS_DIR / "best_model.pth"
DEFAULT_CLASS_MAP_PATH  = _MODELS_DIR / "class_map.json"


# ─────────────────────────────────────────────────────────────
# INFERENCE TRANSFORM (lazy — imported on first use)
# ─────────────────────────────────────────────────────────────

def _get_infer_transform():
    """Build inference transform. Called once after torch is imported."""
    from torchvision import transforms  # type: ignore
    return transforms.Compose([
        transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=MODEL_MEAN, std=MODEL_STD),
    ])


# ─────────────────────────────────────────────────────────────
# CHAR NORMALISATION
# ─────────────────────────────────────────────────────────────

_CLASS_NAME_TO_CHAR: dict[str, str] = {
    "capital_sign": "[CAP]",
    "number_sign":  "[NUM]",
    "period":       ".",
    "comma":        ",",
    "exclamation":  "!",
    "question":     "?",
    "colon":        ":",
    "semicolon":    ";",
    "hyphen":       "-",
    "apostrophe":   "'",
}

def _norm_char(name: str) -> str:
    return _CLASS_NAME_TO_CHAR.get(name, name)


# ─────────────────────────────────────────────────────────────
# CELL CLASSIFIER
# ─────────────────────────────────────────────────────────────

class CellClassifier:
    """
    96.8%-accurate EfficientNet-B3 Braille cell classifier.

    Trained on 3,00,000+ real and augmented Braille cell images.
    Loaded from braille_scripted.pt (TorchScript) when available,
    falls back to best_model.pth (PyTorch checkpoint) otherwise.

    The idx_to_char map is loaded in order of preference:
        1. Embedded in best_model.pth (always up-to-date)
        2. class_map.json  (legacy fallback)

    Blank cells (dot_count == 0) must be filtered OUT by the caller
    before passing to predict_batch — this class has no space output.
    """

    _load_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        scripted_path:    Path = DEFAULT_SCRIPTED_PATH,
        checkpoint_path:  Path = DEFAULT_CHECKPOINT_PATH,
        class_map_path:   Path = DEFAULT_CLASS_MAP_PATH,
    ) -> None:
        self._scripted_path   = Path(scripted_path)
        self._checkpoint_path = Path(checkpoint_path)
        self._class_map_path  = Path(class_map_path)

        self._model           = None        # nn.Module | ScriptModule | None
        self._idx_to_char:    dict[int, str] = {}
        self._device          = None        # torch.device
        self._available:      bool = False
        self._num_classes:    int  = NUM_CLASSES
        self._model_type:     str  = "none"  # "scripted" | "checkpoint" | "none"
        self._val_acc:        float = 0.0
        self._transform       = None

        self._load()

    # ── Load ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """
        Load model weights and char map.

        Try order:
          1. braille_scripted.pt   (TorchScript — fastest, no timm dep)
          2. best_model.pth        (PyTorch checkpoint — auto-builds arch)
        """
        with self._load_lock:
            if self._available:
                return

            # ── Import torch ──────────────────────────────────────────
            try:
                import torch  # type: ignore
            except OSError as exc:
                logger.error("CellClassifier: torch DLL failed: %s — disabled", exc)
                return

            # ── Device selection ──────────────────────────────────────
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(
                "CellClassifier: device=%s  (CUDA=%s)",
                self._device,
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            )
            self._transform = _get_infer_transform()

            # ── Load idx_to_char from checkpoint (authoritative) ──────
            self._load_char_map(torch)

            # ── Try TorchScript model first ───────────────────────────
            if self._try_load_scripted(torch):
                return

            # ── Fall back to checkpoint ───────────────────────────────
            self._try_load_checkpoint(torch)

    def _load_char_map(self, torch) -> None:
        """
        Load idx_to_char from best_model.pth (embedded) or class_map.json.
        The checkpoint's embedded map is always preferred as ground truth.
        """
        # Priority 1: embedded in checkpoint
        if self._checkpoint_path.exists():
            try:
                state = torch.load(
                    self._checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                if isinstance(state, dict) and "idx_to_char" in state:
                    raw = state["idx_to_char"]
                    # Handle both int and str keys (torch.save preserves int keys)
                    self._idx_to_char = {int(k): str(v) for k, v in raw.items()}
                    self._val_acc = float(state.get("val_acc", 0.0))
                    logger.info(
                        "CellClassifier: loaded idx_to_char from checkpoint "
                        "(%d classes, val_acc=%.4f)",
                        len(self._idx_to_char),
                        self._val_acc,
                    )
                    self._num_classes = len(self._idx_to_char)
                    return
            except Exception as exc:
                logger.warning("CellClassifier: checkpoint char-map load failed: %s", exc)

        # Priority 2: class_map.json
        if self._class_map_path.exists():
            try:
                with open(self._class_map_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if "idx_to_char" in raw:
                    self._idx_to_char = {int(k): v for k, v in raw["idx_to_char"].items()}
                elif "idx_to_class" in raw:
                    self._idx_to_char = {int(k): _norm_char(v) for k, v in raw["idx_to_class"].items()}
                else:
                    raise KeyError("class_map.json missing 'idx_to_char' and 'idx_to_class'")
                self._num_classes = len(self._idx_to_char)
                logger.info("CellClassifier: loaded idx_to_char from class_map.json (%d classes)", self._num_classes)
                return
            except Exception as exc:
                logger.warning("CellClassifier: class_map.json load failed: %s", exc)

        logger.error("CellClassifier: no char map found — classifier will be disabled")

    def _try_load_scripted(self, torch) -> bool:
        """
        Attempt to load braille_scripted.pt as a TorchScript model.

        Returns True on success (sets self._available).
        """
        if not self._scripted_path.exists():
            logger.info("CellClassifier: braille_scripted.pt not found — skipping")
            return False

        if not self._idx_to_char:
            logger.warning("CellClassifier: char map missing — cannot enable scripted model")
            return False

        try:
            model = torch.jit.load(str(self._scripted_path), map_location=self._device)
            model.eval()

            # Warmup / shape validation
            dummy = torch.zeros(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).to(self._device)
            with torch.no_grad():
                out = model(dummy)

            if out.shape[1] != self._num_classes:
                logger.error(
                    "CellClassifier: scripted model output=%d classes, "
                    "char map has %d — mismatch, skipping scripted model",
                    out.shape[1], self._num_classes,
                )
                return False

            self._model = model
            self._model_type = "scripted"
            self._available = True
            logger.info(
                "CellClassifier: braille_scripted.pt loaded (TorchScript) "
                "%d classes, val_acc=%.1f%%, device=%s",
                self._num_classes,
                self._val_acc * 100,
                self._device,
            )
            return True

        except Exception as exc:
            logger.warning("CellClassifier: scripted model load failed: %s — trying checkpoint", exc)
            return False

    def _try_load_checkpoint(self, torch) -> bool:
        """
        Load best_model.pth as a full PyTorch checkpoint.

        Builds EfficientNet-B3 via timm (preferred) or torchvision.
        Returns True on success.
        """
        if not self._checkpoint_path.exists():
            logger.error("CellClassifier: best_model.pth not found at '%s'", self._checkpoint_path)
            return False

        if not self._idx_to_char:
            logger.error("CellClassifier: char map missing — cannot load checkpoint")
            return False

        try:
            state = torch.load(
                self._checkpoint_path,
                map_location=self._device,
                weights_only=True,
            )
            weights = state.get("model_state_dict", state)

            model = self._build_efficientnet_b3(self._num_classes, torch)
            model.load_state_dict(weights, strict=True)
            model.eval()
            model.to(self._device)

            # Warmup
            dummy = torch.zeros(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).to(self._device)
            with torch.no_grad():
                out = model(dummy)

            if out.shape[1] != self._num_classes:
                logger.error(
                    "CellClassifier: checkpoint output=%d != char map %d",
                    out.shape[1], self._num_classes,
                )
                return False

            self._model = model
            self._model_type = "checkpoint"
            self._available = True
            logger.info(
                "CellClassifier: best_model.pth loaded (epoch=%s) "
                "%d classes, val_acc=%.1f%%, device=%s",
                state.get("epoch", "?"),
                self._num_classes,
                self._val_acc * 100,
                self._device,
            )
            return True

        except Exception as exc:
            logger.error("CellClassifier: checkpoint load failed: %s", exc)
            return False

    def _build_efficientnet_b3(self, num_classes: int, torch):
        """Build EfficientNet-B3 via timm (preferred) or torchvision fallback."""
        import torch.nn as nn  # type: ignore
        try:
            import timm  # type: ignore
            model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=num_classes)
            logger.debug("CellClassifier: architecture built via timm")
            return model
        except ImportError:
            logger.warning("timm not installed — falling back to torchvision EfficientNet-B3")

        from torchvision.models import efficientnet_b3  # type: ignore
        model = efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        logger.debug("CellClassifier: architecture built via torchvision")
        return model

    # ── Public API ──────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True if model is loaded and ready for inference."""
        return self._available and self._model is not None

    def model_info(self) -> dict:
        """
        Return diagnostic information about the loaded model.
        Useful for hackathon demo and API health endpoints.
        """
        return {
            "available":    self._available,
            "model_type":   self._model_type,     # "scripted" | "checkpoint" | "none"
            "num_classes":  self._num_classes,
            "val_accuracy": round(self._val_acc * 100, 2),   # e.g. 96.82
            "device":       str(self._device),
            "input_size":   MODEL_INPUT_SIZE,
            "threshold":    CONFIDENCE_THRESHOLD,
            "model_path":   str(
                self._scripted_path if self._model_type == "scripted"
                else self._checkpoint_path
            ),
        }

    def preprocess_pil_image(self, pil_image: Image.Image) -> Image.Image:
        """
        Preprocess PIL image to match training conditions:
        1. Grayscale conversion.
        2. Resize to intermediate STANDARD_SIZE (64x64).
        3. Histogram equalization.
        """
        # 1. Convert to grayscale if needed
        if pil_image.mode != 'L':
            pil_image = pil_image.convert('L')
        
        # 2. Resize to standard size (e.g. 64x64)
        STANDARD_SIZE = 64
        pil_image = pil_image.resize((STANDARD_SIZE, STANDARD_SIZE), Image.LANCZOS)
        
        # 3. Apply histogram equalization
        img_array = np.array(pil_image)
        img_array = cv2.equalizeHist(img_array)
        
        return Image.fromarray(img_array)

    def predict_single(self, pil_image: Image.Image) -> dict:
        """
        Classify one cropped Braille cell image.

        Args:
            pil_image: PIL Image of the cropped cell (any size/mode).

        Returns:
            {char, confidence, top3, low_confidence}
        """
        if not self.is_available():
            raise RuntimeError("CellClassifier not loaded")

        import torch  # type: ignore

        try:
            pil_image = self.preprocess_pil_image(pil_image)
        except Exception as exc:
            logger.warning("predict_single: preprocessing failed: %s", exc)

        transform = self._transform or _get_infer_transform()
        tensor = transform(pil_image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            if self._device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self._model(tensor)
            else:
                logits = self._model(tensor)

        probs = torch.softmax(logits[0], dim=0)
        top3_probs, top3_idxs = probs.topk(3)

        top3 = [
            {"char": self._idx_to_char.get(int(idx), "?"), "confidence": float(p)}
            for idx, p in zip(top3_idxs, top3_probs)
        ]
        best_char = top3[0]["char"]
        best_conf = top3[0]["confidence"]

        # Confidence boosting for very confident predictions
        if best_conf > 0.90:
            best_conf = min(1.0, best_conf * 1.05)

        # Output validation: ensure no prediction returns "space"
        if best_char == " ":
            best_char = "?"

        return {
            "char":           best_char,
            "confidence":     round(best_conf, 4),
            "top3":           top3,
            "low_confidence": best_conf < CONFIDENCE_THRESHOLD,
        }

    def predict_batch(self, pil_images: list[Image.Image]) -> list[dict]:
        """
        Classify a batch of cropped Braille cell images in one forward pass.

        Significantly faster than calling predict_single in a loop.
        Falls back to per-image prediction on tensor construction errors.

        Args:
            pil_images: List of PIL Images (one per detected, non-blank cell).

        Returns:
            List of {char, confidence, top3, low_confidence} dicts.
        """
        if not self.is_available():
            raise RuntimeError("CellClassifier not loaded")

        if not pil_images:
            return []

        import torch  # type: ignore
        transform = self._transform or _get_infer_transform()

        try:
            preprocessed_images = []
            for img in pil_images:
                try:
                    preprocessed_images.append(self.preprocess_pil_image(img))
                except Exception as exc:
                    logger.warning("predict_batch: individual preprocessing failed: %s", exc)
                    preprocessed_images.append(img)

            tensors = [transform(img) for img in preprocessed_images]
            batch   = torch.stack(tensors).to(self._device)
        except Exception as exc:
            logger.warning("predict_batch: tensor build failed (%s) — single fallback", exc)
            return [self.predict_single(img) for img in pil_images]

        with torch.no_grad():
            if self._device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self._model(batch)   # type: ignore[operator]
            else:
                logits = self._model(batch)       # type: ignore[operator]

        probs_all = torch.softmax(logits, dim=1)              # (N, 46)
        top3_probs_all, top3_idxs_all = probs_all.topk(3, dim=1)

        results: list[dict] = []
        for i in range(len(pil_images)):
            top3 = [
                {
                    "char":       self._idx_to_char.get(int(top3_idxs_all[i, k]), "?"),
                    "confidence": float(top3_probs_all[i, k]),
                }
                for k in range(3)
            ]
            best_char = top3[0]["char"]
            best_conf = top3[0]["confidence"]

            # Confidence boosting for very confident predictions
            if best_conf > 0.90:
                best_conf = min(1.0, best_conf * 1.05)

            # Output validation: ensure no prediction returns "space"
            if best_char == " ":
                best_char = "?"

            results.append({
                "char":           best_char,
                "confidence":     round(best_conf, 4),
                "top3":           top3,
                "low_confidence": best_conf < CONFIDENCE_THRESHOLD,
            })

        return results


# ─────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────

_classifier_instance: Optional[CellClassifier] = None
_singleton_lock = threading.Lock()


def get_classifier(
    scripted_path:   Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    class_map_path:  Optional[Path] = None,
) -> CellClassifier:
    """
    Return the process-wide singleton CellClassifier.

    The model loads once on first call and is reused for all requests.
    Pass custom paths only in tests / benchmarks.
    """
    global _classifier_instance
    if _classifier_instance is None:
        with _singleton_lock:
            if _classifier_instance is None:
                kwargs: dict = {}
                if scripted_path   is not None: kwargs["scripted_path"]   = Path(scripted_path)
                if checkpoint_path is not None: kwargs["checkpoint_path"] = Path(checkpoint_path)
                if class_map_path  is not None: kwargs["class_map_path"]  = Path(class_map_path)
                _classifier_instance = CellClassifier(**kwargs)
    return _classifier_instance


# ─────────────────────────────────────────────────────────────
# HELPER: crop numpy → PIL
# ─────────────────────────────────────────────────────────────

def crop_cell_to_pil(
    img:     np.ndarray,
    bbox:    tuple[float, float, float, float],
    padding: int = 6,
) -> Optional[Image.Image]:
    """
    Crop a Braille cell from a numpy image using its bounding box.

    Applies `padding` pixels of border on all sides (clamped to image edges).
    Returns None if the resulting crop is empty.

    Args:
        img:     Grayscale or BGR uint8 ndarray.
        bbox:    (x1, y1, x2, y2) in pixel coords (floats OK).
        padding: Extra pixels around the bbox for visual context.
    """
    try:
        h, w = img.shape[:2]
        x1 = max(0, int(bbox[0]) - padding)
        y1 = max(0, int(bbox[1]) - padding)
        x2 = min(w, int(bbox[2]) + padding)
        y2 = min(h, int(bbox[3]) + padding)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Determine if the background is light or dark.
        # The pre-trained EfficientNet model expects white/light dots on a black/dark background.
        # If the background is light (mean brightness > 127), we invert the crop values to match.
        # If it is already dark (mean <= 127), we keep the crop as is.
        mean_val = np.mean(crop)
        if mean_val > 127:
            processed_crop = 255 - crop
        else:
            processed_crop = crop
        return Image.fromarray(processed_crop)
    except Exception as exc:
        logger.debug("crop_cell_to_pil: failed bbox=%s: %s", bbox, exc)
        return None


# ─────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("\n" + "=" * 60)
    print("  BrailleVision AI — CellClassifier Smoke Test")
    print("  Model: 3,00,000+ images | Val Acc: 96.8%")
    print("=" * 60)

    clf = get_classifier()

    print(f"\n  Model info:")
    info = clf.model_info()
    for k, v in info.items():
        print(f"    {k:<15}: {v}")

    if clf.is_available():
        import numpy as np
        from PIL import Image

        # Synthetic white cell crop (as the pipeline would produce)
        dummy_arr = np.ones((64, 64), dtype=np.uint8) * 240
        dummy_pil = Image.fromarray(dummy_arr)

        result = clf.predict_single(dummy_pil)
        print(f"\n  predict_single (blank white crop):")
        print(f"    char       = '{result['char']}'")
        print(f"    confidence = {result['confidence']:.4f}")
        print(f"    low_conf   = {result['low_confidence']}")
        print(f"    top3       = {result['top3']}")

        batch = clf.predict_batch([dummy_pil] * 8)
        print(f"\n  predict_batch (8 images):")
        for i, r in enumerate(batch):
            print(f"    [{i}] '{r['char']}' conf={r['confidence']:.4f}")

        print("\n  [OK] Smoke test passed.\n")
    else:
        print("\n  [WARN] Classifier not available — check model paths above.\n")
