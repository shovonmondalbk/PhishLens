# PhishLens 

**A Multi-Modal, LLM-Powered Phishing Website Detection System**

> **Research Project**  Department of Computer Science & Engineering, University of Dhaka, Bangladesh  
> Supervised by **Prof. Dr. Md. Shariful Islam**, IIT, University of Dhaka

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [System Architecture](#system-architecture)
- [Components](#components)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Chrome Extension](#chrome-extension)
- [Configuration](#configuration)
- [Benchmark Evaluation](#benchmark-evaluation)
- [Comparison with State of the Art](#comparison-with-state-of-the-art)
- [Project Structure](#project-structure)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)
- [Citation](#citation)
- [Authors](#authors)

---

## Overview

PhishLens is a **multimodal, explainable AI framework** for real-time phishing website detection. It addresses five critical gaps identified in existing literature:

| Gap | How PhishLens Addresses It |
|-----|---------------------------|
| No explainability in multimodal systems | Integrated **Llama 3 8B Instruct** as a local LLM explanation generator |
| Fusion strategies are too basic | **Consensus-based weighted ensemble** with dynamic weight adjustment |
| Research stays in the lab | Deployed as **FastAPI web service** + **Chrome Extension (Manifest V3)** |
| Redirect chains are ignored | **Playwright** resolves full redirect chains before analysis |
| LLM approaches require paid APIs | Runs **100% locally** — zero external API dependencies |

The system combines three specialized analyzers — URL (DeBERTa), HTML (ELECTRA), and Visual (CLIP + OCR) — through a consensus-based Fusion Engine to achieve **99.40% accuracy** and **99.32% F1** on a live 20,000-URL benchmark.

---

## Key Results

### 20,000-URL Benchmark (Fusion Engine)

| Metric | Value |
|--------|-------|
| **Accuracy** | **99.40%** |
| **Precision** | **99.71%** |
| **Recall** | **99.09%** |
| **F1 Score** | **99.40%** |
| False Positive Rate | 0.29% |
| False Negative Rate | 0.91% |
| Avg Inference Time | 2.135s / URL |

```
Confusion Matrix (n = 20,000)
                  Predicted Legit    Predicted Phish
Actual Legit           9971               29
Actual Phish             91             9909
```

### Per-Component Performance

| Component | Accuracy | F1 | AUC |
|-----------|----------|----|-----|
| URL Analyzer (DeBERTa v3) | 98.93% | 98.92% | 0.9952 |
| HTML Analyzer (ELECTRA base) | 99.06% | 98.99% | 0.9978 |
| Visual Analyzer (CLIP + OCR) | 84.50% | 84.49% | — |
| **Fusion Engine (Full)** | **99.40%** | **99.32%** | — |

---

## System Architecture

```
Target URL
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Preprocessing & Feature Extraction                          │
└───────────┬──────────────┬──────────────────────────────────┘
            │              │              │
            ▼              ▼              ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ URL Analyzer│ │HTML Analyzer│ │Visual Anlyzr│
    │  DeBERTa v3 │ │  ELECTRA    │ │ CLIP + OCR  │
    │  (86M params)│ │(110M params)│ │  Playwright │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │
           └───────────────┴───────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Fusion Engine V2.3   │
              │  Consensus Voting +    │
              │  Dynamic Weights +     │
              │  FP Prevention         │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │    LLM Explainer       │
              │  Llama 3 8B Instruct   │
              │  (CPU, GGUF Q4-K-M)    │
              └────────────┬───────────┘
                           │
              ┌────────────┴───────────┐
              │                        │
              ▼                        ▼
    ┌──────────────────┐    ┌──────────────────────┐
    │  FastAPI Backend │    │  Chrome Extension    │
    │  + Web Interface │    │  (Manifest V3)       │
    └──────────────────┘    └──────────────────────┘
```

**Fusion Engine Default Weights:**
- URL: 0.40 | HTML: 0.40 | Visual: 0.20
- Visual weight boosts to **0.60** when brand-domain mismatch detected
- Requires **2 of 3 analyzers** to agree before phishing verdict
- FP prevention caps score at 35 for trusted domains (.gov, .edu, .ac) and payment logos (Visa, Mastercard, PayPal)

---

## Components

### 1. URL Analyzer
- **Model:** DeBERTa v3 base (86M parameters)
- Fine-tuned on 100,000 URLs (50K phishing + 50K legitimate)
- Extracts **17 numerical features** (URL length, subdomain depth, entropy, brand mismatch, etc.)
- Input: `[protocol] SEP [subdomain] SEP [domain] SEP [tld] SEP [path] SEP [features]`
- **Inference time:** ~0.12s

### 2. HTML Content Analyzer
- **Model:** ELECTRA base (110M parameters)
- Fine-tuned on 81,998 HTML samples
- Keyword scoring across 5 categories: urgency (28), threat (33), credential (34), action (25), reward (24) keywords
- **Inference time:** ~0.15s

### 3. Visual Analyzer
- **Models:** CLIP ViT-B/32 + EasyOCR + Playwright
- Resolves full redirect chains before screenshot capture (1280×720)
- Zero-shot brand matching against **45 brands** across 14 categories
- Brand-domain mismatch detection with login indicator validation
- **Inference time:** ~3.20s (screenshot-dominant)

### 4. Fusion Engine V2.3
- Consensus-based weighted ensemble
- Dynamic weight adjustment on brand mismatch
- Multi-level consensus voting (2/3 agreement required)
- Built-in false positive prevention logic
- Whitelist/blacklist support via `config.json`

### 5. LLM Explainer
- **Model:** Llama 3 8B Instruct (GGUF Q4-K-M, CPU)
- Generates human-readable explanations for every detection decision
- Template-based fallback for low-latency contexts
- Toggled via `config.json` — `"llm_enabled": true/false`
- **Zero impact on detection accuracy** (confirmed by ablation study)

---

## Dataset

The PhishLens URL dataset consists of **100,000 labeled URLs** (balanced 50K/50K), built through a 5-stage curation pipeline:

| Stage | Phishing | Legitimate | Notes |
|-------|----------|------------|-------|
| Raw Collection | ~1,200,000 | ~800,000 | PhishTank, OpenPhish, GitHub, Tranco |
| Deduplication | ~820,000 | ~480,000 | Exact + normalized URL dedup |
| Liveness Validation | ~100,000 | ~100,000 | HTTP 200 OK required |
| URL Path Expansion | ~100,000 | — | 5 paths per Tranco domain |
| **Final Balanced Set** | **50,000** | **50,000** | Stratified random sampling |

**Sources:**
- Phishing: PhishTank (35K, Jan–Dec 2024) + OpenPhish (15K)
- Legitimate: Tranco Top-100K domains

**Split:** 80% train / 10% validation / 10% test (domain-isolated, seed=42)

> **Dataset file:** `data/raw/url/phishlens_url_dataset.csv`  
> Format: `id, url, label, source` | Label: `0` = legitimate, `1` = phishing

---

## Installation

### Prerequisites

- Python 3.11.8
- CUDA 12.9 (optional — see hardware notes below)
- Google Chrome (for extension)
- Node.js (optional, for extension development)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/PhishLens.git
cd PhishLens

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Playwright browsers
playwright install chromium

# 5. Download model weights
# Place models in the following paths:
# backend/models/url/final_model/        ← DeBERTa (720MB)
# backend/models/html/final_model/       ← ELECTRA
# backend/models/visual/yolo_logos/logo_detector/weights/best.pt  ← YOLOv8
# Download Llama 3 8B GGUF and set path in config.json
```

> **⚠️ Hardware Note (RTX 5070 / Blackwell):**  
> PyTorch does not currently support sm_120 (Blackwell architecture).  
> All inference runs CPU-only. The workaround is already applied in the codebase:
> ```python
> import os
> os.environ['CUDA_VISIBLE_DEVICES'] = ''
> ```

### Requirements

```
torch==2.2.1+cu121
transformers==4.38.1
ultralytics==8.1.24
fastapi==0.109+
uvicorn
playwright==1.41+
easyocr
beautifulsoup4==4.12+
openai-clip
llama-cpp-python
pandas
scikit-learn
```

---

## Usage

### Start the FastAPI Backend

```bash
# Activate venv first
venv\Scripts\activate

# Run the API server
cd backend
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Server starts at: `http://localhost:8000`  
Swagger docs at: `http://localhost:8000/docs`

### Quick URL Check (CLI)

```python
import requests

response = requests.post("http://localhost:8000/api/quick_check", 
    json={"url": "https://example.com"})
print(response.json())
```

### Full Multimodal Analysis (CLI)

```python
response = requests.post("http://localhost:8000/api/analyze",
    json={"url": "https://suspicious-site.com"})
result = response.json()
print(f"Verdict: {result['verdict']}")
print(f"Risk Score: {result['risk_score']}")
print(f"Explanation: {result['explanation']}")
```

### Run Benchmark Evaluation

```bash
python scripts/benchmark_evaluation.py \
    --dataset data/raw/url/phishlens_url_dataset.csv \
    --samples 20000 \
    --seed 42 \
    --output benchmark_results/

# Resume interrupted benchmark
python scripts/benchmark_evaluation.py --resume
```

---

## API Reference

| Endpoint | Method | Description | Avg Time |
|----------|--------|-------------|----------|
| `/api/quick_check` | POST | URL-only rapid analysis | ~0.12s |
| `/api/quick_batch` | POST | Batch URL check (up to 50 URLs) | ~0.5s |
| `/api/analyze` | POST | Full multimodal + LLM explanation | ~2.5s |
| `/api/analyze_batch` | POST | Batch full analysis (up to 10 URLs) | — |
| `/api/whitelist` | GET/POST/DELETE | Manage whitelisted domains | — |
| `/api/blacklist` | GET/POST/DELETE | Manage blacklisted domains | — |
| `/api/status` | GET | System health, analyzer status, cache | — |

**Request body:**
```json
{
  "url": "https://example.com"
}
```

**Response (full analysis):**
```json
{
  "url": "https://example.com",
  "verdict": "LEGITIMATE",
  "risk_score": 2.1,
  "risk_level": "LOW",
  "url_score": 1.2,
  "html_score": 0.8,
  "visual_score": 0.0,
  "detected_brand": null,
  "brand_mismatch": false,
  "threat_type": null,
  "explanation": "This website appears legitimate...",
  "inference_time": 2.48
}
```

---

## Chrome Extension

The extension is built on **Manifest V3** with three-phase scanning:

**Phase 1 — Page Load:** All links (up to 200) collected and sent for quick URL-only batch check.  
**Phase 2 — Real-time Highlighting:** Results returned as color-coded link borders (red = phishing, green = safe).  
**Phase 3 — Click Protection:** When a flagged link is clicked, navigation halts and a warning modal appears with options to cancel, proceed at risk, or run full analysis.

### Installation (Developer Mode)

1. Open Chrome → `chrome://extensions/`
2. Enable **Developer Mode** (top right)
3. Click **Load unpacked**
4. Select the `frontend/extension/` folder
5. The PhishLens icon appears in your toolbar

### Extension Features

- ✅ Real-time link highlighting on every page
- ✅ Click blocking with risk confirmation modal
- ✅ Popup dashboard with per-analyzer scores
- ✅ LLM-powered plain-language explanation
- ✅ Whitelist/blacklist management
- ✅ Tiered analysis (fast scan + deep analysis on demand)
- ✅ Single-page app support (Mutation Observer for dynamic links)

---

## Configuration

All system settings are controlled through a single `backend/inference/config.json` file.

```json
{
  "fusion": {
    "url_weight": 0.40,
    "html_weight": 0.40,
    "visual_weight": 0.20,
    "phishing_threshold": 50.0,
    "high_confidence_threshold": 95.0,
    "brand_mismatch_boost": 0.60,
    "fp_prevention_cap": 35.0
  },
  "llm": {
    "enabled": true,
    "model_path": "backend/models/llm/llama-3-8b-instruct.Q4_K_M.gguf",
    "fallback_to_template": true
  },
  "cache": {
    "ttl_seconds": 3600
  },
  "whitelist": [],
  "blacklist": []
}
```

> Set `"llm_enabled": false` for template-based explanations (faster, no LLM required).

---

## Benchmark Evaluation

The 20K benchmark samples **10,000 phishing + 10,000 legitimate** URLs from the full 100K dataset (random seed 42) and evaluates only the Fusion Engine (URL + HTML + Visual combined).

```bash
python scripts/benchmark_evaluation.py \
    --dataset data/raw/url/phishlens_url_dataset.csv \
    --samples 20000 \
    --seed 42

# Output files saved to: benchmark_results/
# - phishlens_test_20k_dataset.csv
# - detailed_results.csv
# - benchmark_summary.txt
```

**Latest Results (20K, seed=42):**

```
Accuracy:   99.40%    Precision:  99.71%
Recall:     99.09%    F1-Score:   99.40%
FPR:         0.29%    FNR:         0.91%
Avg Time:   2.135s/URL
```

---

## Comparison with State of the Art

| Method | Year | Accuracy | F1 | XAI | Deployment |
|--------|------|----------|----|-----|------------|
| Phishpedia | 2022 | 85.15% | 82.78% | ✗ | — |
| Shark-Eyes | 2023 | 95.35% | 94.34% | ✗ | — |
| ChatPhish | 2023 | 95.80% | 95.91% | Partial | — |
| KnowPhish | 2024 | 92.05% | 91.44% | ✗ | — |
| PhishAgent | 2024 | 96.10% | 96.13% | ✗ | — |
| Phisher | 2025 | 98.13% | 98.00% | ✗ | — |
| **PhishLens** | **2026** | **99.40%** | **99.32%** | **✓** | **Web + Extension** |

PhishLens is the **only system** among all tested methods that provides:
- Structured LLM-powered explainability
- Full browser extension deployment
- Redirect chain resolution
- Zero external API dependencies

---

## Project Structure

```
PhishLens/
├── backend/
│   ├── api/
│   │   └── app.py                  # FastAPI application
│   ├── inference/
│   │   ├── config.json             # System-wide configuration
│   │   ├── fusion_engine.py        # Fusion Engine V2.3
│   │   ├── url_analyzer.py         # DeBERTa URL analyzer
│   │   ├── html_analyzer.py        # ELECTRA HTML analyzer
│   │   ├── visual_analyzer.py      # CLIP + OCR visual analyzer
│   │   └── llm_explainer.py        # Llama 3 explainer
│   ├── models/
│   │   ├── url/final_model/        # DeBERTa weights (720MB)
│   │   ├── html/final_model/       # ELECTRA weights
│   │   └── visual/
│   │       └── yolo_logos/         # YOLOv8 logo model
│   └── training/                   # Training scripts
├── data/
│   ├── raw/
│   │   └── url/
│   │       └── phishlens_url_dataset.csv   # 100K URL dataset
│   ├── logos/
│   │   ├── brand_logos/            # 270 brand classes
│   │   ├── brand_logos_yolo/       # YOLO format
│   │   └── LogoDet-3K_yolo/        # LogoDet-3K dataset
│   └── cache/
├── frontend/
│   ├── web/                        # Web dashboard (HTML/CSS/JS)
│   └── extension/
│       ├── manifest.json           # Chrome Manifest V3
│       ├── background.js           # Service worker
│       ├── content.js              # Page content script
│       └── popup.js                # Extension popup UI
├── scripts/
│   ├── benchmark_evaluation.py     # 20K benchmark runner
│   └── resume_yolo_training.py     # YOLOv8 training resume
├── logs/
├── tests/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## Known Limitations

1. **RTX 5070 / Blackwell (sm_120):** PyTorch CUDA not yet compatible. All inference runs CPU-only. Full GPU support expected in future PyTorch releases.

2. **Screenshot latency:** Playwright screenshot capture accounts for ~80% of pipeline time (~1.8s). Moving to cloud GPU deployment will reduce total inference below 1s.

3. **Visual analyzer scope:** CLIP zero-shot covers 45 brands. Dedicated YOLOv8 logo detector (in development) will expand coverage and enable spatial localization.

4. **Benchmark composition:** 1,000-URL live evaluation reflects a specific time window. Extended longitudinal testing is planned.

5. **English-only:** Current models are optimized for English-language phishing content.

---

## Future Work

- [ ] **YOLOv8 Logo Detector:** Training on LogoDet-3K targeting 500+ brand classes, 50,000+ annotated examples
- [ ] **Automated Retraining Pipeline:** Continuous learning from APWG, PhishTank, OpenPhish feeds with concept drift detection
- [ ] **Mobile App:** Background URL scanning across SMS, WhatsApp, email, and QR codes
- [ ] **Multi-browser Support:** Firefox and Safari extension ports
- [ ] **Cloud Deployment:** GPU-accelerated API server with enterprise features (shared threat intelligence, centralized caching)
- [ ] **Multi-language Support:** Extend analyzers to cover non-English phishing campaigns
- [ ] **Adversarial Robustness Testing:** Systematic evaluation against URL obfuscation, HTML cloaking, and visual perturbations

---

## Citation

If you use PhishLens in your research, please cite:

```bibtex
@article{phishlens2025,
  title   = {PhishLens: A Multi-Modal, LLM-Powered Phishing Website Detection System},
  author  = {Shovon Mondal and Chandan Paul},
  year    = {2025},
  note    = {Department of Computer Science and Engineering, University of Dhaka}
}
```

---

## Authors

| Name | Role |
|------|------|
| **Shovon Mondal** | Researcher |
| **Chandan Paul** | Researcher |

**Supervisor:** Prof. Dr. Md. Shariful Islam — IIT, University of Dhaka

---

## Acknowledgements

This work builds on excellent open-source foundations: DeBERTa-v3 (Microsoft), ELECTRA (Google), CLIP (OpenAI), YOLOv8 (Ultralytics), Llama 3 8B Instruct (Meta AI), and FastAPI. Phishing data sourced from PhishTank and OpenPhish; legitimate domains from the Tranco List.

---

<p align="center">
  <i>PhishLens — Multimodal Phishing Detection with Explainability</i><br>
  University of Dhaka · Department of CSE
</p>
