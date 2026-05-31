# 👁️👁️ BrailleVision AI — Assistive Physical Braille Reader & Translator

<div align="center">

![BrailleVision AI](https://img.shields.io/badge/BrailleVision-AI-blueviolet?style=for-the-badge&logo=eye&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React Native](https://img.shields.io/badge/React_Native-Expo_56-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Nano-FF4B4B?style=for-the-badge&logo=ultralytics&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A mobile-first, real-time AI platform that reads physical embossed Braille from a camera and converts it to speech in under 200ms — empowering blind and visually impaired users worldwide.**

</div>

---

## 📖 Table of Contents

1. [Project Overview](#-project-overview)
2. [System Architecture](#️-system-architecture)
3. [How It Works — End-to-End Pipeline](#-how-it-works--end-to-end-pipeline)
   - [Step 1: Image Preprocessing](#step-1-image-preprocessing-corepreprocesspy)
   - [Step 2: Hybrid Dot Detection](#step-2-hybrid-dot-detection-coredetectorpy)
   - [Step 3: Cell Segmentation](#step-3-cell-segmentation-coresegmenterpy)
   - [Step 4: Braille Decoding](#step-4-braille-decoding-coredecoderpy)
   - [Step 5: AI Error Correction](#step-5-ai-error-correction-corecorrectorpy)
   - [Step 6: Multi-Language Translation](#step-6-multi-language-translation-coretranslatorpy)
   - [Step 7: Neural Speech Synthesis](#step-7-neural-speech-synthesis-coretts_enginepy)
   - [Step 8: Context Memory](#step-8-context-memory-aicontext_memorypy)
4. [Key Features](#-key-features)
5. [Tech Stack](#-tech-stack)
6. [Codebase Layout](#-codebase-layout)
7. [Setup & Installation](#-setup--installation)
8. [Environment Configuration](#-environment-configuration)
9. [API Reference](#-api-reference)
10. [Mobile App Screens](#-mobile-app-screens)
11. [Training Pipeline](#-training-pipeline)
12. [Testing & Validation](#-testing--validation)
13. [Accessibility Compliance](#️-accessibility-compliance)
14. [Performance Targets](#-performance-targets)
15. [License](#-license)

---

## 🔭 Project Overview

**BrailleVision AI** is a full-stack assistive technology platform built for the BrailleVision Hackathon 2026. It solves one of the most challenging computer vision problems in accessibility: **reading physical, embossed Braille paper from a mobile camera in real-world, uncontrolled conditions**.

Unlike simple digital Braille converters, this system handles:
- **Real embossed paper** with uneven dot heights, shadows, and texture noise
- **Varied lighting** — dim rooms, harsh sunlight, mixed light sources
- **Camera shake** and motion blur
- **Skewed or tilted** paper at any angle
- **Front and back** of the Braille page (raised vs. indented dots)
- **Grade 1 and Grade 2** standard Braille (including full contraction sets)

The decoded Braille text is then optionally corrected by an LLM (Groq Llama-3.1), translated into 17 languages, and read aloud using Microsoft Edge Neural TTS — all within a single scan.

---

## 🏗️ System Architecture

```
                 📱 MOBILE CLIENT (React Native · Expo SDK 56)
         ┌───────────────────────────────────────────────────────┐
         │  expo-camera          Live camera frame capture       │
         │  expo-speech + AV     Native & neural TTS playback    │
         │  react-native-svg     SVG dot overlay on viewfinder   │
         │  Zustand store        Centralized state management     │
         │  Expo Router          File-system based navigation     │
         └──────────────────────────┬────────────────────────────┘
                                    │  HTTPS multipart/form-data
                                    │  (JPEG frames + JSON opts)
                                    ▼
                 🐍 BACKEND SERVER (FastAPI · Python 3.12)
         ┌───────────────────────────────────────────────────────┐
         │                                                       │
         │  ① ImagePreprocessor   CLAHE → Shadow Removal →      │
         │                        Perspective Correction →       │
         │                        Side Detection → Mirror        │
         │                                                       │
         │  ② HybridBrailleDetector                             │
         │      OpenCV SimpleBlobDetector (fast path)           │
         │    + YOLOv8 Nano inference (deep path)               │
         │    → Confidence-weighted NMS Fusion                  │
         │                                                       │
         │  ③ BrailleCellSegmenter  DBSCAN spatial clustering   │
         │                          → ordered 6-dot cell list   │
         │                                                       │
         │  ④ BrailleDecoder   Grade 1 + Grade 2 lookup tables  │
         │                     + Fuzzy Hamming matching         │
         │                                                       │
         │  ⑤ AIErrorCorrector  Groq Llama-3.1 (primary)       │
         │                      + pyspellchecker (fallback)     │
         │                                                       │
         │  ⑥ BrailleTranslator  Google Translate → 17 langs   │
         │                                                       │
         │  ⑦ BrailleTTSEngine   Microsoft edge-tts neural      │
         │                        voices → base64 MP3 bytes     │
         │                                                       │
         │  ⑧ ContextMemory  Sliding window sentence buffer     │
         │                                                       │
         └──────────────────────────┬────────────────────────────┘
                                    │  SQLAlchemy ORM (aiosqlite)
                                    ▼
                        📂 SQLite DATABASE
         ┌───────────────────────────────────────────────────────┐
         │  braillevision.db   ScanHistory ORM model            │
         │  ./data/            Audio .mp3 asset cache           │
         └───────────────────────────────────────────────────────┘
```

---

## 🔬 How It Works — End-to-End Pipeline

Every image — whether from a live camera frame or a full capture — flows through the `BrailleAIPipeline` orchestrator (`backend/ai/pipeline.py`). The pipeline has two modes:

| Mode | Trigger | Skips | Target Latency |
|:-----|:--------|:------|:---------------|
| **Live Frame** (`process_live_frame`) | Continuous camera loop | Error correction, TTS | < 200 ms |
| **Full Capture** (`process_image`) | User taps "Capture" | Nothing | ~500–1500 ms |

---

### Step 1: Image Preprocessing (`core/preprocess.py`)

The `ImagePreprocessor` normalizes raw camera images into a clean, detector-friendly grayscale map through a 9-step pipeline:

| # | Step | What it does | Why it matters |
|:--|:-----|:-------------|:---------------|
| 1 | **Load Image** | Accepts `bytes`, `ndarray`, or file path | Flexible input from HTTP multipart |
| 2 | **Quality Assessment** | Measures brightness + Laplacian blur score | Generates real-time camera guidance ("Hold steady", "Move to brighter area") |
| 3 | **Grayscale** | BGR → single-channel | Reduces compute; Braille is monochrome |
| 4 | **CLAHE** | Contrast Limited Adaptive Histogram Equalization (clip=3.0, tile=8×8) | Recovers dot contrast under uneven lighting |
| 5 | **Shadow Removal** | Morphological dilation + per-pixel normalization (`img / background × 255`) | Eliminates cast shadows from fingers, edges, and ambient lighting gradients |
| 6 | **Gaussian Denoising** | 3×3 kernel smoothing | Reduces sensor noise before blob detection |
| 7 | **Perspective Correction** | Canny edge → largest 4-corner contour → `warpPerspective` | Corrects angled shots so dot grids align horizontally |
| 8 | **Side Detection** | Laplacian variance analysis across 32×32 windows | Distinguishes front (indentations) from back (raised bumps) |
| 9 | **Mirror if Back** | `cv2.flip(img, 1)` if `side == "back"` | Restores correct reading orientation for back-view images |

**Output**: A normalized grayscale `ndarray` + quality metadata + camera guidance string.

---

### Step 2: Hybrid Dot Detection (`core/detector.py`)

The `HybridBrailleDetector` fuses two complementary algorithms to maximize dot recall:

#### 2a. OpenCV SimpleBlobDetector (Fast Path)
- Tuned parameters for real embossed Braille geometry:
  - **Area**: 8–2500 px² (accounts for scale variation)
  - **Circularity**: ≥ 0.30 (dots aren't perfect circles due to shadows)
  - **Convexity**: ≥ 0.50 (relaxed for real-world dots)
  - **Inertia**: ≥ 0.20 (allows slightly elliptical dots from side-lighting)
- Applies **adaptive Gaussian thresholding** before blob search to handle absolute brightness differences
- Falls back to inverted image if no blobs found (for back-lit or high-contrast setups)
- **Confidence** is estimated from how uniformly sized all detected dots are — uniform dot sizes → higher confidence

#### 2b. YOLOv8 Nano (Deep Path)
- Loads a fine-tuned `yolov8n.pt` (or pretrained as fallback)
- Resizes images to 640×640 for inference, then **maps detections back to original coordinates**
- Confidence threshold: 0.30
- Performs a warmup pass on startup to eliminate first-call latency

#### 2c. Confidence-Weighted NMS Fusion
Both detector outputs are merged with **spatial Non-Maximum Suppression (NMS)**:

```
If YOLOv8 avg_conf > 0.5:  YOLO weight = 70%,  Blob weight = 30%  [yolo_dominant]
Else:                       YOLO weight = 40%,  Blob weight = 60%  [blob_dominant]
```

Dots within 12px of each other are merged into a single centroid using **confidence-weighted position averaging**. This produces the final deduplicated dot list with quality labels: `good`, `low`, or `poor`.

---

### Step 3: Cell Segmentation (`core/segmenter.py`)

The `BrailleCellSegmenter` converts the flat dot list into ordered 6-dot Braille cells using **DBSCAN spatial clustering** — this is the key innovation that makes the system robust to camera scale, tilt, and any paper size:

1. **Cluster rows** of dots using DBSCAN on Y-coordinates
2. **Estimate dot spacing** from median horizontal and vertical gaps
3. **Group dots into 6-slot cells** by aligning each dot to the nearest expected `(col, row_in_cell)` position
4. **Sort cells** in natural reading order (left-to-right, top-to-bottom)
5. **Fill missing dots** with zeros if a dot position has no detection

This approach works without any fixed grid assumption, adapting dynamically to the camera distance and angle.

---

### Step 4: Braille Decoding (`core/decoder.py`)

The `BrailleDecoder` translates 6-element binary patterns `(dot1, dot2, dot3, dot4, dot5, dot6)` into text through a multi-stage lookup:

```
Dot layout:
  dot1  dot4
  dot2  dot5
  dot3  dot6
```

#### Decode Priority Chain

| Priority | Lookup | Covers |
|:---------|:-------|:-------|
| 1 | **Grade 1 exact match** | a–z, punctuation, indicators (`[CAP]`, `[NUM]`) |
| 2 | **Number Mode** | Digits 0–9 (same patterns as a–j, active after `[NUM]` indicator) |
| 3 | **Grade 2 whole-word contractions** | `and`, `for`, `of`, `with` (only when cell is standalone in a word) |
| 4 | **Grade 2 affixes** | `sh`, `th`, `wh`, `er`, `ou`, `en`, `in`, `st`, `ar` |
| 5 | **Fuzzy Hamming match** | Closest Grade 1 entry within 1–2 dot error distance |
| 6 | **Unknown** | Returns `?` with confidence 0.0 |

#### Capitalization Rules
- Cell `(0,0,0,0,0,1)` → `[CAP]` indicator → next letter is uppercase
- `capitalize_next` flag resets after each uppercase character

#### Confidence Scoring
| Match Type | Confidence |
|:-----------|:-----------|
| Exact Grade 1 / Grade 2 | 1.0 |
| Fuzzy (1 dot error) | 0.75 |
| Fuzzy (2 dot errors) | 0.50 |
| Unresolvable | 0.0 |

Average confidence across all cells generates quality labels: `excellent` (≥0.85), `good` (≥0.65), `fair` (≥0.45), `poor` (<0.45).

---

### Step 5: AI Error Correction (`core/corrector.py`)

The `AIErrorCorrector` fixes OCR-style errors in decoded text using a **two-tier cascade**:

#### Tier 1: Groq Llama-3.1-8b-instant (Primary)
- Uses a **Braille-aware system prompt** that explains common dot detection failure modes (missing letters, swapped patterns like `i` ↔ `e`, `h` ↔ `b`)
- Sends context from the sliding window memory for better semantic correction
- Results are **memoized by MD5 hash** to avoid duplicate API calls
- Temperature: 0.1 (deterministic corrections)
- Falls through to Tier 2 on API failure

#### Tier 2: pyspellchecker (Offline Fallback)
- Word-by-word correction preserving punctuation and original capitalization
- Works entirely offline with no external dependencies

#### Word-Level Diff
`get_diff()` computes a `difflib.SequenceMatcher` diff between original and corrected text and returns a structured list of changes for frontend display.

---

### Step 6: Multi-Language Translation (`core/translator.py`)

The `BrailleTranslator` supports **17 languages** via Google Translate (deep-translator):

| Language | Code | TTS Voice |
|:---------|:-----|:----------|
| English | `en` | `en-US-JennyNeural` |
| Hindi | `hi` | `hi-IN-SwaraNeural` |
| Tamil | `ta` | `ta-IN-PallaviNeural` |
| Marathi | `mr` | `mr-IN-AarohiNeural` |
| Telugu | `te` | `te-IN-MohanNeural` |
| Kannada | `kn` | `kn-IN-SapnaNeural` |
| Bengali | `bn` | `bn-IN-TanishaaNeural` |
| Gujarati | `gu` | `gu-IN-DhwaniNeural` |
| Punjabi | `pa` | `pa-IN-OjasvNeural` |
| Spanish | `es` | `es-ES-ElviraNeural` |
| French | `fr` | `fr-FR-DeniseNeural` |
| German | `de` | `de-DE-KatjaNeural` |
| Arabic | `ar` | `ar-SA-ZariyahNeural` |
| Japanese | `ja` | `ja-JP-NanamiNeural` |
| Chinese (Simplified) | `zh-CN` | `zh-CN-XiaoxiaoNeural` |
| Portuguese | `pt` | `pt-BR-FranciscaNeural` |
| Russian | `ru` | `ru-RU-SvetlanaNeural` |

All translations are **cached by MD5 hash** to avoid redundant network calls. English-to-English passes through instantly without any API call.

---

### Step 7: Neural Speech Synthesis (`core/tts_engine.py`)

The `BrailleTTSEngine` generates natural-sounding speech using **Microsoft Edge TTS** (neural voices):

- Generates audio as raw `bytes` for in-memory streaming
- Can also **save MP3 files** to `./data/` for persistence and caching
- Each language maps to its own neural voice (see table above)
- Falls back gracefully if `edge-tts` is unavailable

---

### Step 8: Context Memory (`ai/context_memory.py`)

The `ContextMemory` module maintains a **sliding window of recently decoded sentences**:

- Stores the last N sentences as context
- Provides a `get_correction_context()` string for the LLM corrector
- Improves correction accuracy by giving the LLM semantic continuity across multiple scans of the same Braille page

---

## 🌟 Key Features

### Core Vision
- **Hybrid Dot Detection**: YOLOv8 Nano + OpenCV SimpleBlobDetector fused via confidence-weighted NMS — maximizes recall under any lighting
- **Adaptive Cell Segmentation**: DBSCAN-based clustering adapts to any camera scale, tilt, or paper size — no fixed grid required
- **Dual-Grade Braille Decoding**: Full Grade 1 (alphabet, punctuation, numbers) and Grade 2 (contractions and affixes) with proper indicator handling
- **Fuzzy Hamming Correction**: Tolerates 1–2 dot detection errors before falling to "unknown" — gracefully handles partial occlusions

### AI & Language
- **Two-Tier Error Correction**: Groq Llama-3.1 (primary) → pyspellchecker (offline fallback) — works with or without internet
- **17-Language Translation**: Google Translate via deep-translator with result caching
- **Neural TTS in 17 Languages**: Microsoft Edge neural voices, each matched to its own language
- **Sliding Context Memory**: The LLM corrector receives recent scan history for better semantic continuity

### Camera & Guidance
- **Live Frame Mode**: Sub-200ms preprocessing → detection → segmentation → decode for real-time viewfinder overlays
- **Smart Camera Guidance**: Real-time voice cues — *"Hold camera steady"*, *"Move to brighter area"*, *"Good positioning — scanning ✅"*
- **Side Detection**: Automatically detects front (indentation) vs. back (raised) of the Braille page and corrects orientation
- **Perspective Correction**: Auto-warps angled shots to a flat top-down view using 4-corner homography

### Mobile App
- **Live Scanner**: Continuous camera feed with SVG dot overlay and real-time guidance banner
- **Photo Upload**: Pick any image from the gallery for full-pipeline processing
- **Scan History**: Paginated log of all past scans with text, confidence, language, and audio
- **Settings**: Language selection, TTS toggle, high-contrast mode, developer debug info
- **5-Step Voice Onboarding**: Fully narrated introduction for new users with screen-reader support
- **Zustand State Management**: Centralized, reactive store for scan cache, settings, and theme
- **Focused Camera Lifecycle**: Unmounts the viewfinder and releases the hardware camera/torch immediately upon tab blur to prevent lock conflicts with Image/Document Pickers
- **Atomic State Hooks**: Integrates `useAppStore.getState()` for stable and fresh Zustand operations inside callbacks, solving stale closure lag in live frame intervals

### Accessibility
- **Screen Reader Support**: Every element has `accessibilityLabel`, `accessibilityRole`, and `accessibilityState`
- **High-Contrast Theme**: Custom HSL color system with extreme high-contrast toggle
- **Full Voice Control**: App reads all feedback, guidance, and decoded text aloud
- **Jarvis-style Scan Animation**: Animated scanning line and HUD indicators for tactile feedback

---

## 🛠️ Tech Stack

### Backend
| Technology | Version | Role |
|:-----------|:--------|:-----|
| **Python** | 3.12+ | Runtime |
| **FastAPI** | 0.111.0 | Async HTTP API framework |
| **Uvicorn** | 0.29.0 | ASGI server |
| **OpenCV** | 4.9.0 | Image preprocessing + blob detection |
| **NumPy** | 1.26.4 | Numerical operations |
| **Ultralytics (YOLOv8)** | 8.2.0 | Deep learning dot detector |
| **PyTorch** | 2.3.0 | YOLOv8 inference backend |
| **scikit-learn** | 1.4.2 | DBSCAN clustering |
| **SciPy** | 1.13.0 | Spatial distance computations |
| **edge-tts** | 7.2.8 | Microsoft neural TTS generation |
| **deep-translator** | 1.11.4 | Google Translate wrapper |
| **openai** | 1.30.0 | Groq API client (OpenAI-compatible) |
| **pyspellchecker** | 0.8.1 | Offline spell correction fallback |
| **SQLAlchemy** | 2.0.30 | Async ORM for scan history |
| **aiosqlite** | 0.20.0 | Async SQLite driver |
| **albumentations** | 1.4.6 | Synthetic training augmentation |
| **python-dotenv** | 1.0.1 | Environment variable loading |
| **pytest + httpx** | 8.2.0 / 0.27.0 | Test suite |

### Mobile
| Technology | Version | Role |
|:-----------|:--------|:-----|
| **React Native** | 0.85.3 | Cross-platform mobile UI |
| **Expo SDK** | 56 | Development environment + managed workflow |
| **TypeScript** | 6.0.3 | Type safety |
| **Expo Router** | 56.2.5 | File-system based navigation |
| **expo-camera** | 56.0.7 | Live camera frame capture |
| **expo-av** | 16.0.8 | Audio playback for neural TTS |
| **expo-speech** | 56.0.3 | Native TTS fallback |
| **react-native-svg** | 15.15.4 | Dot overlay SVG graphics |
| **react-native-reanimated** | 4.3.1 | Smooth micro-animations |
| **zustand** | 5.0.13 | Lightweight state management |
| **lucide-react-native** | 1.16.0 | Icon library |

---

## 📂 Codebase Layout

The project comprises **42 fully implemented source files** across two apps:

```
BrailleVision AI/
│
├── backend/                          # Python FastAPI backend (42 files)
│   ├── main.py                       # FastAPI entrypoint — wires lifecycles, CORS, routers
│   ├── requirements.txt              # All pinned Python dependencies
│   ├── .env / .env.example           # Environment variables (API keys, DB URL, model path)
│   ├── braillevision.db              # Auto-created SQLite database
│   │
│   ├── ai/                           # Master pipeline & memory
│   │   ├── pipeline.py               # BrailleAIPipeline — orchestrates all 8 modules
│   │   └── context_memory.py         # Sliding sentence window for LLM context
│   │
│   ├── core/                         # Core CV, detection, and decoding engines
│   │   ├── preprocess.py             # ImagePreprocessor — 9-step image normalization
│   │   ├── detector.py               # HybridBrailleDetector — YOLOv8 + Blob NMS fusion
│   │   ├── segmenter.py              # BrailleCellSegmenter — DBSCAN cell clustering
│   │   ├── decoder.py                # BrailleDecoder — Grade 1 & Grade 2 + fuzzy Hamming
│   │   ├── corrector.py              # AIErrorCorrector — Groq LLM + pyspellchecker
│   │   ├── translator.py             # BrailleTranslator — 17-language Google Translate
│   │   └── tts_engine.py             # BrailleTTSEngine — neural edge-tts audio generation
│   │
│   ├── database/                     # SQLAlchemy async database layer
│   │   ├── db.py                     # Async engine + session factory + init_db()
│   │   └── models.py                 # ScanHistory ORM model
│   │
│   ├── routers/                      # FastAPI HTTP route handlers
│   │   ├── scan.py                   # POST /api/scan/frame and /api/scan/capture
│   │   ├── tts.py                    # POST /api/tts/speak
│   │   ├── translate.py              # POST /api/translate
│   │   └── history.py                # GET/DELETE /api/history
│   │
│   ├── training/                     # Dataset synthesis and model training
│   │   ├── generate_synthetic.py     # Generates 5000+ synthetic Braille images for YOLOv8
│   │   ├── train_yolo.py             # Fine-tune YOLOv8 and export to ONNX
│   │   └── benchmark.py              # Per-condition accuracy benchmark (shadow, blur, etc.)
│   │
│   ├── tests/                        # Pytest test suite
│   │   └── test_decoder.py           # 23 tests: Grade 1 & 2, capitalization, numbers, fuzzy
│   │
│   └── data/                         # Audio asset cache directory (auto-created)
│
└── mobile/                           # React Native Expo 56 mobile app
    ├── app.json                       # Expo app configuration
    ├── package.json                   # Node dependencies (Expo SDK 56)
    ├── tsconfig.json                  # TypeScript configuration
    │
    ├── app/                           # Expo Router file-system routes
    │   ├── index.tsx                  # Initial launcher — checks onboarding status
    │   ├── _layout.tsx                # Root Stack navigator + font loading
    │   ├── onboarding.tsx             # 5-step voice onboarding (fully narrated)
    │   └── (tabs)/                    # Main tab screens
    │       ├── _layout.tsx            # Tab navigator with accessibility labels
    │       ├── scanner.tsx            # 📷 Live Braille scanner with guidance HUD
    │       ├── upload.tsx             # 🖼️ Photo upload + full pipeline scan
    │       ├── history.tsx            # 📜 Paginated scan history with audio replay
    │       └── settings.tsx           # ⚙️ Language, TTS, contrast, debug options
    │
    ├── components/                    # Reusable accessible UI components
    │   ├── BrailleOverlay.tsx         # SVG dot bounding box overlay on camera view
    │   ├── ConfidenceDisplay.tsx      # Heatmap-colored confidence percentage indicator
    │   ├── GuidanceBanner.tsx         # Micro-animated voice feedback banner
    │   ├── ScanAnimation.tsx          # Jarvis-style futuristic scanning line animation
    │   └── TTSPlayer.tsx              # Full-featured audio player (play/pause/progress)
    │
    ├── store/
    │   └── useAppStore.ts             # Zustand store (scan cache, settings, language, theme)
    │
    ├── services/
    │   ├── api.ts                     # Backend HTTP client (multipart + JSON fetch)
    │   └── voice.ts                   # Neural TTS audio cache + expo-speech fallback
    │
    ├── hooks/
    │   └── useBrailleScanner.ts       # Frame throttle, capture handler, camera lifecycle
    │
    └── constants/
        └── theme.ts                   # HSL-based color palette + high-contrast design tokens
```

---

## ⚡ Setup & Installation

### Prerequisites

| Requirement | Minimum Version |
|:-----------|:----------------|
| Python | 3.12+ |
| Node.js | 20+ |
| npm | 10+ |
| Groq API Key | (Optional — free tier at [console.groq.com](https://console.groq.com)) |

---

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Copy environment config and fill in your values
copy .env.example .env      # Windows
cp .env.example .env        # macOS/Linux

# Start the backend server (auto-reloads on code changes)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> **First startup**: The backend automatically:
> 1. Creates `braillevision.db` (SQLite) with the ScanHistory schema
> 2. Downloads YOLOv8 nano weights (`yolov8n.pt`) if not already present
> 3. Runs a warm-up pass through YOLOv8 to eliminate first-request latency
> 4. Creates the `./data/` directory for audio file caching

The API will be live at `http://localhost:8000` and the interactive docs at `http://localhost:8000/docs`.

---

### 2. Mobile App Setup

```bash
# Navigate to mobile directory
cd mobile

# Install dependencies (legacy-peer-deps required for React 19 packages)
npm install --legacy-peer-deps

# Start the Expo dev server
npm run start
```

Then:
- Press **`a`** — launch on Android emulator
- Press **`i`** — launch on iOS simulator
- **Scan the QR code** — run on a physical device via Expo Go

> **Connect to backend**: Open `mobile/services/api.ts` and update the `BASE_URL` constant to point to your machine's local IP address (e.g., `http://192.168.1.100:8000`). Do **not** use `localhost` on a physical device.

---

## 🔧 Environment Configuration

Copy `.env.example` to `.env` in the `backend/` directory and configure:

```env
# Groq LLM API Key (optional — enables AI error correction)
# Get a free key at https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# LLM model selection (llama-3.1-8b-instant is free and fast)
GROQ_MODEL=llama-3.1-8b-instant

# SQLite database connection string
DATABASE_URL=sqlite+aiosqlite:///./braillevision.db

# Path to fine-tuned YOLOv8 model (optional, uses yolov8n.pt if not found)
MODEL_PATH=./ai/models/braille_yolov8n.pt

# CORS origins (use * for development, restrict in production)
CORS_ORIGINS=["*"]

# Debug mode
DEBUG=True

# Maximum image file size in megabytes
MAX_IMAGE_SIZE_MB=10

# Minimum confidence threshold for dot detection
CONFIDENCE_THRESHOLD=0.35
```

---

## 📡 API Reference

All endpoints are documented interactively at **`http://localhost:8000/docs`** (Swagger UI).

### Health

| Endpoint | Method | Description |
|:---------|:------:|:------------|
| `/` | `GET` | Welcome message + service status |
| `/health` | `GET` | Database connectivity health check |

---

### Scan

#### `POST /api/scan/frame`
**Live frame scanning** — optimized fast path for real-time camera overlays.

- **Request**: `multipart/form-data` with `file` (JPEG frame bytes)
- **Skips**: Error correction, TTS generation
- **Target**: < 200ms end-to-end

**Response:**
```json
{
  "success": true,
  "raw_text": "hello",
  "corrected_text": "hello",
  "dots": [{"x": 120.0, "y": 84.5, "size": 10.0, "confidence": 0.85, "source": "yolo"}],
  "cell_count": 5,
  "dot_count": 18,
  "avg_confidence": 0.87,
  "guidance": "Good positioning — scanning ✅",
  "side_detected": "front",
  "detection_quality": "good",
  "processing_time_ms": 143.2,
  "error": null
}
```

---

#### `POST /api/scan/capture`
**Full capture scan** — complete pipeline with correction, translation, TTS, and database storage.

- **Request**: `multipart/form-data` with `file` + optional JSON options:

```json
{
  "correct": true,
  "translate_to": "hi",
  "speak": true,
  "save_annotated": true
}
```

**Response:**
```json
{
  "success": true,
  "raw_text": "helo wrold",
  "corrected_text": "hello world",
  "translated_text": "नमस्ते दुनिया",
  "cells": [{"pattern": [1,0,0,0,0,0], "confidence": 1.0, "x": 80.0, "y": 60.0, "bbox": [...], "dot_count": 1}],
  "confidences": [1.0, 0.75, ...],
  "avg_confidence": 0.91,
  "cell_count": 10,
  "dot_count": 34,
  "guidance": "Good positioning — scanning ✅",
  "side_detected": "front",
  "quality": {"brightness": 148.2, "blur_score": 243.1, "lighting": "good", "blur": "sharp", "ok": true},
  "detection_quality": "good",
  "correction_method": "llm",
  "correction_changes": [{"original": "helo", "corrected": "hello", "position": 0, "type": "replace"}],
  "was_corrected": true,
  "annotated_image_base64": "...(base64 JPEG)...",
  "audio_bytes": "...(base64 MP3)...",
  "processing_time_ms": 892.4,
  "error": null
}
```

---

### Text-to-Speech

#### `POST /api/tts/speak`
Generate neural TTS audio from text.

**Request:**
```json
{
  "text": "Hello world",
  "lang": "en"
}
```

**Response:**
```json
{
  "success": true,
  "audio_base64": "//NExAAA...",
  "lang": "en",
  "voice": "en-US-JennyNeural"
}
```

---

### Translation

#### `POST /api/translate`
Translate decoded Braille text to any supported language.

**Request:**
```json
{
  "text": "Hello world",
  "target_lang": "hi"
}
```

**Response:**
```json
{
  "original": "Hello world",
  "translated": "नमस्ते दुनिया",
  "target_language": "hi",
  "language_name": "Hindi",
  "success": true
}
```

---

### History

| Endpoint | Method | Parameters | Description |
|:---------|:------:|:-----------|:------------|
| `/api/history` | `GET` | `?page=1&size=20` | Paginated scan history |
| `/api/history/stats` | `GET` | — | Word counts, accuracy averages, today's scan count |
| `/api/history/{id}` | `DELETE` | `id` (path) | Delete a specific scan record |

---

## 📱 Mobile App Screens

### 1. Scanner Tab (`app/(tabs)/scanner.tsx`)
- Continuous live camera feed via `expo-camera` with a **focused-only active lifecycle** (unmounts the camera automatically on tab blur to release hardware resources and turn off the flash)
- **SVG dot overlay** (`BrailleOverlay.tsx`) draws detected dot positions in real-time
- **Guidance banner** (`GuidanceBanner.tsx`) shows animated voice cues
- **Confidence display** (`ConfidenceDisplay.tsx`) heatmap indicator
- **Jarvis-style scan animation** (`ScanAnimation.tsx`) sweeps across the frame
- Tap **"Capture"** to trigger the full pipeline scan
- Decoded text and audio appear in an expandable result card

### 2. Upload Tab (`app/(tabs)/upload.tsx`)
- Select photos from the device gallery via `expo-image-picker` or `expo-document-picker`
- **Zero-conflict uploading flow**: Guaranteed safe document/image picking because active hardware camera streams from the Scanner tab are systematically unmounted and released
- Supports JPEG, PNG, and PDF formats
- Runs the full pipeline (preprocessing → detection → decode → correct → translate → TTS)
- Displays annotated image with detected dot positions overlaid

### 3. History Tab (`app/(tabs)/history.tsx`)
- Paginated list of all past scans fetched from `/api/history`
- Each card shows: decoded text, timestamp, confidence score, language, correction status
- **Audio replay** of previously generated TTS via `TTSPlayer.tsx`
- Pull-to-refresh and infinite scroll
- Per-record delete

### 4. Settings Tab (`app/(tabs)/settings.tsx`)
- **Language**: Select any of 17 supported translation target languages
- **TTS Toggle**: Enable/disable audio playback
- **High-Contrast Mode**: Switches to bright yellow/black accessibility palette
- **Debug Info**: Shows API URL, model status, last scan stats
- **Clear History**: Wipes all scan records from the database

### 5. Onboarding (`app/onboarding.tsx`)
- 5-step narrated introduction using `expo-speech`
- Each step explains a core feature with spoken description
- Persists completion status so it only shows once

---

## 🎯 Training Pipeline

### 1. Generate Synthetic Dataset (`backend/training/generate_synthetic.py`)

Creates **5000+ synthetic Braille page images** for YOLOv8 training:
- Randomly selects Braille letter patterns and renders them as dot grids
- Applies augmentations via **Albumentations**:
  - Random illumination angles and gradients
  - Cast shadow overlays
  - Paper texture noise
  - Perspective distortions
  - Gaussian blur
  - Salt-and-pepper noise
- Exports images + **YOLOv8-format YOLO `.txt` bounding box annotations**

```bash
cd backend
python training/generate_synthetic.py
# Output: backend/data/synthetic_dataset/ (images/ + labels/)
```

### 2. Fine-Tune YOLOv8 (`backend/training/train_yolo.py`)

Fine-tunes `yolov8n.pt` on the synthetic dataset and exports optimized models:

```bash
python training/train_yolo.py
# Output: backend/ai/models/braille_yolov8n.pt
#         backend/ai/models/braille_yolov8n.onnx
```

Training parameters are configurable (epochs, batch size, imgsz, device).

### 3. Accuracy Benchmark (`backend/training/benchmark.py`)

Tests detection robustness under simulated adverse conditions:

```bash
python training/benchmark.py
```

Generates per-condition accuracy reports for:
- **Shadow** overlays at varying intensities
- **Low contrast** (brightness reduction)
- **Gaussian blur** at multiple sigma values
- **Salt & pepper noise** at different densities
- **Perspective tilt** at ±15°, ±30°

### 4. Classification & Character Dataset

To achieve grade-2 English Braille compatibility and class-level accuracy, we self-trained an **EfficientNet-B3** backbone classifier on a highly curated, massive-scale collection:
* **Cleaned Dataset Version**: 2.0
* **Total Sample Count**: **304,528 Images** (46 classes)
  * Real Original physical scans: `21,656` samples (augmented 13× to `281,528`)
  * Synthetic samples: `23,000` samples
* **Classifier Accuracy**: **96.82% Validation Accuracy**
* **Multi-Class Taxonomy**: Lowercase alphabet (`a`–`z`), numeric digits (`0`–`9`), and punctuation/special signs (including capitalized indicator `⠠` and numeric mode indicator `⠼`).

> [!TIP]
> For a full, itemized breakdown of classes, splits (80% train / 10% val / 10% test), and taxonomy metrics, view the comprehensive [dataset_info.md](file:///d:/Yash/Music/Live%20Project%20&%20Other%20Things/BrailleVision%20Hackathon%202026/BrailleVision%20AI/dataset/dataset_info.md) documentation.

---

## 🧪 Testing & Validation

### Run the Full Backend Test Suite

```bash
cd backend
python -m pytest tests/ -v
```

The suite covers **25 comprehensive test cases** across both the Braille decoder and the adaptive DBSCAN cell segmenter:

| Test Category | Tests |
|:-------------|:------|
| Single-character Grade 1 decodes (a–z) | 26 |
| Capitalization indicator (`[CAP]`) | 2 |
| Number mode indicator (`[NUM]`) + digits 0–9 | 3 |
| Grade 2 whole-word contractions (`and`, `for`, `of`, `with`) | 4 |
| Grade 2 affixes (`sh`, `th`, `er`, `ou`, etc.) | 4 |
| Fuzzy Hamming correction (1–2 dot errors) | 3 |
| `decode_with_stats()` quality labels | 3 |
| Empty cell / unknown pattern handling | 2 |
| Spacing estimation & cell-splitting (`test_segmenter.py`) | 2 |

All 25 tests are guaranteed to pass on a clean install.

### Integration Test (Backend API)

```bash
cd backend
python -m pytest tests/ -v
```

Includes HTTP integration tests via `httpx` for the scan, TTS, translate, and history endpoints.

### Live Smoke Tests

Each core module has a self-contained `if __name__ == "__main__":` smoke test that can be run individually:

```bash
python backend/core/preprocess.py     # Preprocessing smoke test
python backend/core/detector.py       # Detector smoke test
python backend/core/decoder.py        # Decoder smoke test
python backend/core/corrector.py      # Corrector smoke test
python backend/core/translator.py     # Translator smoke test
python backend/ai/pipeline.py         # End-to-end pipeline smoke test
```

---

## ♿ Accessibility Compliance

BrailleVision AI is built to the highest accessibility standards — it is **itself an assistive technology** and must be usable by people with severe visual impairments:

| Feature | Implementation |
|:--------|:--------------|
| **Screen Reader** | Every UI element has `accessibilityLabel`, `accessibilityRole`, `accessibilityHint`, and `accessibilityState` properties |
| **Real-Time Voice Guidance** | Camera positioning feedback spoken aloud via `expo-speech` on each live frame result |
| **Full Voice Onboarding** | All 5 onboarding steps narrated — no visual reading required |
| **High-Contrast Mode** | Custom HSL palette with bright yellow (`#FFFF00`) accent buttons on deep black backgrounds to eliminate glare |
| **Large Touch Targets** | All interactive elements have minimum 48×48 dp touch areas |
| **Screen Reader Focus Zones** | Critical regions have `accessible={true}` with combined label + hint descriptions |
| **No Silent Failures** | All error states emit a spoken error message via TTS |

---

## 📊 Performance Targets

| Metric | Target | How achieved |
|:-------|:-------|:-------------|
| Live frame latency | < 200 ms | Skips correction and TTS; blob detection is ~5ms |
| Full capture latency | < 1500 ms | Parallel TTS generation; LLM cached by MD5 |
| Dot detection recall | > 90% (good lighting) | Hybrid fusion compensates for individual detector failures |
| Grade 1 decode accuracy | > 95% (good lighting) | Exact lookup + 1-dot fuzzy tolerance |
| LLM correction API latency | < 500 ms | Groq llama-3.1-8b-instant; MD5 cache |
| Translation latency | < 300 ms | deep-translator + MD5 hash cache |
| App startup time | < 3 s | YOLOv8 warm-up at server startup, not on first request |

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](./mobile/LICENSE) file for details.

---

<div align="center">

Built with ❤️ for the **BrailleVision Hackathon 2026**

*"Technology should eliminate barriers, not create them."*

</div>
