# AI Tools Disclosure

## BrailleVision Hackathon 2026 — AI Tools Used

This document discloses all AI tools and models used in the development of BrailleVision AI, as required by hackathon rules.

---

## 1. Core AI Model

| Component | Details |
|-----------|---------|
| **Architecture** | EfficientNet-B3 (via `timm` library) |
| **Training** | Self-trained on custom Braille dataset using Google Colab |
| **Framework** | PyTorch 2.3 |
| **Validation Accuracy** | 96.82% |
| **Use** | Classifying individual Braille cell crops into 46 character classes |

The model was trained entirely from scratch (pretrained ImageNet weights for feature extraction, custom head for Braille classification).

---

## 2. AI Error Correction

| Component | Details |
|-----------|---------|
| **Model** | `llama-3.1-8b-instant` via Groq API |
| **Use** | Post-processing: corrects OCR-style decoding errors in the decoded Braille text |
| **Prompt** | Fix spelling/grammar errors while preserving meaning |

---

## 3. Text-to-Speech

| Component | Details |
|-----------|---------|
| **Service** | Microsoft Edge TTS (`edge-tts` Python library) |
| **Voices** | Neural voices: en-US-AriaNeural, hi-IN-SwaraNeural, and 10+ others |
| **Use** | Converting decoded Braille text to spoken audio |

---

## 4. Computer Vision

| Component | Details |
|-----------|---------|
| **Library** | OpenCV 4.9 |
| **Techniques** | CLAHE, Gaussian blur, adaptive thresholding, SimpleBlobDetector, DBSCAN clustering |
| **Use** | Preprocessing Braille images and segmenting individual cells |

---

## 5. Development Assistance

AI coding assistants were used to accelerate development of boilerplate, API documentation, and UI scaffolding. All core algorithms (cell segmentation, dot detection, decoder logic) were designed and implemented by the team.

---

*All AI tools are used in accordance with their respective terms of service.*
