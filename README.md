<div align="center">

# ⭐ BrailleVision AI
# Assistive Physical Braille Reader & Translator

![BrailleVision AI](https://img.shields.io/badge/BrailleVision-AI-blueviolet?style=for-the-badge&logo=eye&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React Native](https://img.shields.io/badge/React_Native-Expo_56-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Nano-FF4B4B?style=for-the-badge&logo=ultralytics&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A mobile-first, real-time AI platform that reads physical embossed Braille from a camera feed and converts it to speech in under 200ms — empowering blind and visually impaired users worldwide.**

</div>

---

> [!IMPORTANT]
> **BrailleVision AI** is a fully integrated, assistive technology suite built for the **BrailleVision Hackathon 2026**. It leverages a state-of-the-art hybrid computer vision pipeline and custom neural classifiers to translate physical embossed paper sheets into clear speech and 6 target languages in real-world lighting conditions.

---

<details>
<summary>📖 <b>Table of Contents (Click to Expand)</b></summary>

1. [Project Overview](#-project-overview)
2. [System Architecture](#-system-architecture)
3. [How It Works — End-to-End Pipeline](#-how-it-works--end-to-end-pipeline)
   - [Step 1: Image Preprocessing](#step-1-image-preprocessing)
   - [Step 2: Hybrid Dot Detection](#step-2-hybrid-dot-detection)
   - [Step 3: Cell Segmentation](#step-3-cell-segmentation)
   - [Step 4: Braille Decoding](#step-4-braille-decoding)
   - [Step 5: AI Error Correction](#step-5-ai-error-correction)
   - [Step 6: Multi-Language Translation](#step-6-multi-language-translation)
   - [Step 7: Neural Speech Synthesis](#step-7-neural-speech-synthesis)
   - [Step 8: Context Memory](#step-8-context-memory)
4. [Key Features](#-key-features)
5. [Tech Stack](#-tech-stack)
6. [Codebase Layout](#-codebase-layout)
7. [Setup & Installation](#-setup--installation)
8. [Environment Configuration](#-environment-configuration)
9. [API Reference](#-api-reference)
10. [Mobile App Screens](#-mobile-app-screens)
11. [Model Architecture & Datasets](#-model-architecture--datasets)
12. [Testing & Validation](#-testing--validation)
13. [Accessibility Compliance](#-accessibility-compliance)
14. [Performance Targets](#-performance-targets)
15. [License](#-license)

</details>

---

## 🔭 Project Overview

**BrailleVision AI** solves one of the most challenging computer vision accessibility problems: **reading physical, embossed Braille paper from a mobile camera in real-world, uncontrolled conditions**.

Unlike simple digital Braille simulators, our custom hardware-focused pipeline is engineered to handle:
* **Real embossed paper** with uneven dot heights, creases, and physical texture noise.
* **Adverse lighting** — casting shadows, dim rooms, mixed light sources, and glares.
* **Camera shake** and motion blur from hand-held captures.
* **Perspective tilt** and paper skew at severe angles.
* **Double-sided embossing (interlineations)** — distinguishing front-side dots from back-side indentations.
* **Grade 1 & Grade 2 Braille** standards, including complex multi-cell contractions and affixes.

Decoded text is corrected in real-time by a contextual LLM (Groq Llama-3.1), translated into 6 languages, and read aloud using neural Edge TTS voices — all within a single streamlined scan.

---

## 🏗️ System Architecture

```
                 📱 MOBILE CLIENT (React Native · Expo SDK 56)
         ┌───────────────────────────────────────────────────────┐
         │  expo-camera          Live camera frame capture       │
         │  expo-speech + AV     Native & neural TTS playback    │
         │  react-native-svg     SVG dot overlay on viewfinder   │
         │  Zustand store        Centralized state management    │
         │  Expo Router          File-system based navigation    │
         └──────────────────────────┬────────────────────────────┘
                                    │  HTTPS multipart/form-data
                                    │  (JPEG frames + JSON options)
                                    ▼
                 🐍 BACKEND SERVER (FastAPI · Python 3.12)
         ┌───────────────────────────────────────────────────────┐
         │                                                       │
         │  ① ImagePreprocessor   CLAHE → Shadow Removal →       │
         │                        Perspective Correction →       │
         │                        Side Detection → Auto-Mirror   │
         │                                                       │
         │  ② HybridBrailleDetector                              │
         │      OpenCV SimpleBlobDetector (Fast CV Path)         │
         │    + YOLOv8 Nano Inference (Deep AI Path)             │
         │    → Confidence-Weighted NMS Fusion                   │
         │                                                       │
         │  ③ BrailleCellSegmenter  DBSCAN Spatial Clustering    │
         │                          → Ordered 6-slot Cell Grid   │
         │                                                       │
         │  ④ CellClassifier (EfficientNet-B3 Backbone)          │
         │      Direct 46-class neural character prediction      │
         │                                                       │
         │  ⑤ BrailleDecoder   Grade 1 + Grade 2 lookup tables   │
         │                     + Fuzzy Hamming state-machine     │
         │                                                       │
         │  ⑥ AIErrorCorrector  Groq Llama-3.1-8b-instant        │
         │                      + pyspellchecker (Fallback)      │
         │                                                       │
         │  ⑦ BrailleTranslator  deep-translator → 6 Langs       │
         │                                                       │
         │  ⑧ BrailleTTSEngine   Microsoft Edge Neural TTS       │
         │                        → base64 MP3 Audio Bytes       │
         │                                                       │
         └──────────────────────────┬────────────────────────────┘
                                    │  SQLAlchemy Async ORM
                                    ▼
                         📂 SQLite DATABASE
         ┌───────────────────────────────────────────────────────┐
         │  braillevision.db   ScanHistory table schema          │
         │  ./data/            Audio MP3 file persistent cache   │
         └───────────────────────────────────────────────────────┘
```

---

## 🔬 How It Works — End-to-End Pipeline

Every scanned frame — whether a live camera capture or full gallery upload — is orchestrated by `BrailleAIPipeline` (`backend/ai/pipeline.py`) through two optimized modes:

| Mode | Trigger | Skips | Target Latency |
|:-----|:--------|:------|:---------------|
| **Live Frame** (`process_live_frame`) | Continuous camera viewfinder loop | LLM Correction, Speech Synthesis | **< 200 ms** |
| **Full Capture** (`process_image`) | Tap "Capture" or gallery upload | None (Full Pipeline) | **~500–1200 ms** |

---

### Step 1: Image Preprocessing
The `ImagePreprocessor` (`core/preprocess.py`) normalizes raw images into clean, detector-friendly grayscale arrays:
* **Quality Guard:** Measures average brightness and Laplacian blur to give the user real-time positioning feedback ("Hold steady", "Move to brighter area").
* **CLAHE Contrast Adjustment:** Recovers subtle dot contours under low-contrast, mixed light.
* **Morphological Shadow Removal:** Dynamically estimates cast background lighting gradients using dilation, dividing the foreground by it to eliminate shadows.
* **Perspective Correction:** Runs Canny edge detection, extracts the largest 4-corner contour, and warps the skew into a flat top-down grid view.
* **Auto-Side Mirroring:** Measures Laplacian variance to detect if the sheet's front (embossed dots) or back (indented holes) is scanned, automatically flipping back-view images.

### Step 2: Hybrid Dot Detection
The `HybridBrailleDetector` (`core/detector.py`) fuses two algorithms for maximum dot recall:
* **OpenCV SimpleBlobDetector (Fast Path):** Tuned parameters filter keypoints by area, circularity, convexity, and inertia.
* **YOLOv8 Nano (Deep Path):** Evaluates challenging lighting conditions and maps box outputs to original coordinates.
* **NMS Fusion:** Blends both feeds with confidence-weighted Non-Maximum Suppression to deduplicate adjacent coordinates.

### Step 3: Cell Segmentation
The `BrailleCellSegmenter` (`core/segmenter.py`) groups coordinate lists into 6-slot cells:
* **DBSCAN Clustering:** Clusters rows of dots by Y-coordinates to estimate page tilt.
* **Median Spacing Scale:** Computes dynamic spacing dimensions horizontally and vertically.
* **natural Reading Order:** Groups dots into cells and sorts them left-to-right, top-to-bottom.
* **Sanity Gate:** Automatically drops specks and fills missing dot coordinates with zeros.

### Step 4: Cell Classification
Segmented cell crops are routed to `CellClassifier` (`ai/models/cell_classifier.py`):
* **EfficientNet-B3 Backbone:** Cropped cell images are fed directly to a custom classifier trained on **300K+ images** (achieving **96.82% validation accuracy**).
* **TorchScript Execution:** Runs fast inference on CPU or CUDA via compiled TorchScript serialization (`braille_scripted.pt`).
* **Space Cell Bypass:** Instantly flags blank background cells (dot count of 0) as spaces (`" "`), bypassing neural forward passes entirely to avoid false predictions and preserve word boundaries.

### Step 5: Braille Decoding
The `BrailleDecoder` (`core/decoder.py`) resolves character mappings:
* **Modifier States:** Handles capital indicators (`[CAP]`) and numeric indicators (`[NUM]`) to toggle output registers (mapping `a`-`j` to digits `1`-`0`).
* **Grade 2 Contractions:** Translates whole-word contractions (like `and`, `the`, `with`) and affixes (`sh`, `th`, `ou`, `er`) in their standalone or word contexts.
* **Hamming Fallback:** Falls back to exact or fuzzy Hamming-distance dot pattern lookups if neural predictions are below confidence thresholds (`0.55`).

### Step 6: AI Error Correction
The `AIErrorCorrector` (`core/corrector.py`) runs a two-tier cascade:
* **Tier 1 (Groq Llama-3.1):** Passes decoded text and recent context through Groq with a custom, Braille-aware error prompt to fix letter swaps (e.g., `i` ↔ `e`) and run-on words.
* **Tier 2 (pyspellchecker):** Offline word-by-word spelling correction backup if Groq is unavailable.
* **MD5 Cache:** Hash-caches corrections to bypass API network calls on repeated scans.

### Step 7: Multi-Language Translation
The `BrailleTranslator` (`core/translator.py`) supports instant translations into **6 languages** (English, Hindi, Tamil, Spanish, French, German) using `deep-translator`. All responses are cached by MD5 hashes.

### Step 8: Neural Speech Synthesis
The `BrailleTTSEngine` (`core/tts_engine.py`) synthesizes natural-sounding speech using Microsoft Edge Neural TTS voices, generating base64 MP3 streams on-the-fly and caching files in `/data/`.

---

## 🌟 Key Features

* **Real-Time Visual Overlay:** Viewer displays dynamic SVG bounding boxes with color-coded confidence levels mapping cells in the viewport.
* **Smart Audio Guidance:** Live verbal cues navigate blind users to place their cameras optimally.
* **On-Device Onboarding:** 5-step interactive voice onboarding introduces the application's physical buttons.
* **Zero-Conflict Camera Lifecycle:** Unmounts and releases active camera feeds immediately upon tab blurring to allow gallery uploads without thread lockups.
* **High Contrast Design:** Sleek HSL theme with a togglable high-contrast palette for visually impaired users.
* **Robust Offline Backup:** Handles scanning, decoding, spelling correction, and text-to-speech without active internet.

---

## 🛠️ Tech Stack

### Backend API Server
* **Python 3.12+** / **FastAPI** / **Uvicorn** — Async web framework and server
* **OpenCV (cv2)** — Image preprocessing and CLAHE shadows morphological logic
* **Ultralytics (YOLOv8)** & **PyTorch** — Deep learning dot detection engine
* **scikit-learn (DBSCAN)** — Spatial cell clustering logic
* **edge-tts** — High-fidelity Microsoft Edge neural TTS synthesizer
* **deep-translator** — Google Translate client integration
* **SQLAlchemy** & **aiosqlite** — Async database layer for scanning history logs

### Mobile Client App
* **React Native** & **Expo SDK 56** — Cross-platform physical mobile UI
* **expo-camera** — Viewfinder frame processor client
* **expo-av** — Audio engine for neural TTS persistent stream playback
* **expo-speech** — Offline native TTS fallback synthesis
* **react-native-reanimated** — Micro-animations for high-contrast interface elements
* **zustand** — Global application store and scan history cache
* **lucide-react-native** — Clean, high-contrast visual icon sets

---

## 📂 Codebase Layout

```
BrailleVision AI/
│
├── Dataset & Training/               # Model training resources & datasets
│   ├── Braille-Text.ipynb            # Jupyter notebook used to train the EfficientNet-B3 classifier
│   ├── Test-model.py                 # Evaluation script for character predictions
│   ├── clean_dataset.tar             # Packaged cell images dataset (300K+ augmented)
│   └── dataset_info.md               # Curated dataset classes, taxonomy & split reports
│
├── models/                           # Saved neural network model weights
│   ├── best_model.pth                # PyTorch checkpoint for EfficientNet-B3
│   ├── braille_scripted.pt           # TorchScript compiled model for fast classifier inference
│   ├── class_map.json                # Index-to-char mapping for the 46 Braille classes
│   └── yolov8n.pt                    # Pretrained YOLOv8 Nano weights for dot detection
│
├── backend/                          # Python FastAPI backend
│   ├── main.py                       # FastAPI application entrypoint and startup warmups
│   ├── requirements.txt              # Backend library dependencies
│   ├── .env / .env.example           # Environment configurations (Groq API, DB URL, paths)
│   ├── braillevision.db              # SQLite scan history database
│   │
│   ├── ai/                           # AI pipeline orchestration & sliding window context
│   │   ├── pipeline.py               # BrailleAIPipeline - orchestrates the 8 core modules
│   │   └── context_memory.py         # ContextMemory sliding window for sentence correction
│   │
│   ├── core/                         # Core CV, detection, and decoding engines
│   │   ├── preprocess.py             # ImagePreprocessor - 9-step normalization pipeline
│   │   ├── detector.py               # HybridBrailleDetector - OpenCV Blob + YOLOv8 NMS fusion
│   │   ├── segmenter.py              # BrailleCellSegmenter - DBSCAN cell spacing clustering
│   │   ├── decoder.py                # BrailleDecoder - Grade 1 & 2 lookup and fuzzy Hamming
│   │   ├── corrector.py              # AIErrorCorrector - Groq Llama-3.1 + pyspellchecker
│   │   ├── translator.py             # BrailleTranslator - 6-language cached translation
│   │   └── tts_engine.py             # BrailleTTSEngine - Microsoft Edge Neural Speech
│   │
│   ├── database/                     # SQLite database access layer
│   │   ├── db.py                     # SQLAlchemy async connection engine
│   │   └── models.py                 # ScanHistory SQLAlchemy ORM schema model
│   │
│   ├── routers/                      # REST API routing endpoints
│   │   ├── scan.py                   # /api/scan/frame & /api/scan/capture multipart uploads
│   │   ├── tts.py                    # /api/tts/speak neural speech endpoint
│   │   ├── translate.py              # /api/translate multi-language endpoint
│   │   └── history.py                # /api/history pagination & scan deletion
│   │
│   ├── tests/                        # Comprehensive Pytest test suite
│   │   ├── test_decoder.py           # Core Grade 1 & 2 character/modifier decoder tests
│   │   ├── test_repetition_fix.py    # Repetitive pattern detection and segmentation scale retry
│   │   ├── test_segmenter.py         # Cell segmentation and DBSCAN distance calculations
│   │   ├── test_sentence_decode.py   # Multi-word space cell bypass and classifier override tests
│   │   ├── test_single_cell_bypass.py # direct single-cropped square cell classification tests
│   │   └── test_word_decoding.py     # Clean isolated word-decoding sanity check
│   │
│   └── data/                         # Audio MP3 cache directory (auto-created)
│
└── mobile/                           # React Native Expo 56 mobile app
    ├── app.json                      # Expo application manifest configuration
    ├── package.json                  # Node dependencies (Expo SDK 56)
    ├── tsconfig.json                 # TypeScript compiler configuration
    │
    ├── app/                          # Expo Router navigation screens
    │   ├── index.tsx                 # Entry launcher routing based on onboarding status
    │   ├── _layout.tsx               # Core layout and custom font loader
    │   ├── onboarding.tsx            # Narrated voice-guided onboarding workflow
    │   └── (tabs)/                   # Application bottom tabs
    │       ├── _layout.tsx           # Tab bar with screen reader announcements
    │       ├── scanner.tsx           # Futuristic camera view with SVG overlay and voice guidance
    │       ├── upload.tsx            # Gallery photo upload + annotation overlay scanner
    │       ├── history.tsx           # Paginated scan history with audio replay player
    │       └── settings.tsx          # Settings screen for language, speech, theme and debug
    │
    ├── components/                   # Accessible visual components
    │   ├── BrailleOverlay.tsx        # Dynamic SVG dot and boundary box overlay graphic
    │   ├── ConfidenceDisplay.tsx     # Color-coded circular scanning confidence gauge
    │   ├── GuidanceBanner.tsx        # Voice instruction panel with scanning prompts
    │   ├── ScanAnimation.tsx         # Animated HUD scanning line animation
    │   └── TTSPlayer.tsx             # Interactive speech controller with progress bar
    │
    ├── store/
    │   └── useAppStore.ts            # Global Zustand store (scan caches, high contrast, state)
    │
    ├── services/
    │   ├── api.ts                    # Backend API multipart and JSON communication clients
    │   ├── voice.ts                  # Neural speech local playback and native fallback controllers
    │   └── voiceCommands.ts          # Speech-to-text listener for screen-reader controls
    │
    ├── hooks/
    │   ├── useBrailleScanner.ts      # Camera lifecycle, frame throttling and uploads hook
    │   └── useVoiceCommands.ts       # Speech triggers and router navigation controls hook
    │
    └── constants/
        └── theme.ts                  # HSL color palettes and high contrast theme system
```

---

## ⚡ Setup & Installation

### 1. Backend Server Setup
```bash
# Navigate to backend folder
cd backend

# Initialize and activate Python virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS / Linux

# Install required dependencies
pip install -r requirements.txt

# Copy the configuration environment file
copy .env     # Windows
cp .env       # macOS / Linux

# Start the uvicorn API server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> [!TIP]
> On the first startup, the backend server will automatically:
> 1. Initialize `braillevision.db` SQLite schema.
> 2. Download and save baseline `yolov8n.pt` weights.
> 3. Perform model warm-ups to eliminate subsequent request latency.

---

### 2. Mobile App Setup
```bash
# Navigate to mobile app folder
cd mobile

# Install Node dependencies
npm install --legacy-peer-deps

# Start the local Expo server
npm start
```

Then:
* Press **`a`** — launch on the Android emulator
* Press **`i`** — launch on the iOS simulator
* **Run on a physical device:** Build a custom native development client (`npx expo run:android` or `npx expo run:ios`).
  > To deliver real-time camera processing and native speech synthesis, this application runs as a production-grade custom **Development Build** rather than a sandboxed playground. This allows direct, high-performance access to the device's physical camera and audio hardware.

> [!WARNING]
> **Network Binding:** Open `mobile/services/api.ts` and set `BASE_URL` to your computer's local IP address (e.g., `http://192.168.1.104:8000`) instead of `localhost` so physical devices can communicate with the backend.

---

## 🔧 Environment Configuration

Customize settings in `backend/.env` to control parameters:

```env
# Groq API configuration (Free tier key enables high-context Llama error fixes)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Database target connection string
DATABASE_URL=sqlite+aiosqlite:///./braillevision.db

# Path overrides for detection and classification models
MODEL_PATH=../models/yolov8n.pt

# Debug features and file limits
DEBUG=True
MAX_IMAGE_SIZE_MB=10
CONFIDENCE_THRESHOLD=0.35
```

---

## 📡 API Reference

All routes are fully documented via interactive Swagger UI at **`http://localhost:8000/docs`**.

### 1. Health Endpoints
* **`GET /`** — Welcome message and system state checks.
* **`GET /health`** — Database read/write verification.

### 2. Scanning Actions
#### **`POST /api/scan/frame`**
Optimized live-scanning viewport path.
* **Body:** `multipart/form-data` with `file` (JPEG capture bytes).
* **Bypasses:** Semantic error correction and TTS synthesis to meet <200ms latency budgets.

#### **`POST /api/scan/capture`**
Full pipeline capture path.
* **Body:** `multipart/form-data` with `file` + optional JSON config.
* **Payload Structure:**
  ```json
  {
    "correct": true,
    "translate_to": "hi",
    "speak": true,
    "save_annotated": true
  }
  ```
* **Output Structure:** Includes raw text, LLM corrected text, translations, per-cell confidence arrays, base64 annotated JPEGs, and synthesized MP3 audio tracks.

### 3. Utility Actions
* **`POST /api/tts/speak`** — Synthesizes input text to base64 audio streams using target voice.
* **`POST /api/translate`** — Translates input text into 6 supported languages.
* **`GET /api/history`** — Fetches paginated past scans with search capabilities.
* **`DELETE /api/history/{id}`** — Removes scan records and associated cached audio.

---

## ♿ Accessibility Compliance

BrailleVision AI is designed from the ground up to be **accessible to the blind and visually impaired**:

* **Screen Reader Hooks:** All components are decorated with semantic `accessibilityLabel`, `accessibilityRole`, and `accessibilityState` details.
* **Continuous Verbal Cues:** Preprocessing assessment reads feedback prompts (e.g. *"Hold camera steady"* or *"Excellent lighting"*).
* **Glare Reduction High-Contrast:** Color maps use a distinct black background and `#FFFF00` yellow interactive targets to minimize glare.
* **Optimized Hit Boxes:** Touch controls enforce a minimum `48x48dp` tactile active size.
* **Tactile Scan Animation:** Renders visual scan sweeps alongside matching audio alerts for clean feedback.

---

## 📊 Performance Targets

| Metric | Target | Core Implementation Approach |
|:-------|:-------|:-----------------------------|
| **Live Frame Latency** | **< 200 ms** | Skipping heavy LLM/TTS routines; optimized OpenCV preprocessing |
| **Full Capture Latency** | **< 1200 ms** | Asynchronous Edge TTS generation & Groq API call caches |
| **Dot Detection Recall** | **> 92%** | NMS blending of YOLOv8 Nano & SimpleBlobDetector |
| **Grade 1 Translation Accuracy** | **> 96%** | Exact mappings + fuzzy Hamming distance error recoveries |
| **Multilingual Translation Latency** | **< 300 ms** | MD5 hashed caches for translated sentences |
| **Cold Startup Time** | **< 2.5 s** | YOLOv8 warm-ups run on backend server load lifecycles |

---

## 📝 License

This project is licensed under the **MIT License** — see the [mobile/LICENSE](./mobile/LICENSE) file for details.

<div align="center">

Built with ❤️ for the **BrailleVision Hackathon 2026**

*"Technology should eliminate barriers, not create them."*

</div>
