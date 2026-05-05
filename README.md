# Sybil Lung Cancer Risk Predictor — Desktop Application

A full-featured Electron desktop application wrapping the **Sybil v1.6.0** deep learning model for lung cancer risk prediction from low-dose CT scans (LDCT).

---

## Architecture

```
sybil-app/
├── setup.sh              ← macOS/Linux setup script
├── setup.bat             ← Windows setup script
│
├── python_backend/
│   └── server.py         ← Flask HTTP API wrapping Sybil
│
├── sybil-source/         ← Sybil 1.6.0 source (unchanged)
│   └── sybil/
│       ├── model.py      ← Sybil class
│       ├── serie.py      ← Serie class (DICOM/PNG loading)
│       ├── models/       ← SybilNet architecture
│       ├── loaders/      ← Image loading pipeline
│       └── utils/        ← Visualization, metrics, etc.
│
└── electron_app/
    ├── package.json
    └── src/
        ├── main.js       ← Electron main process, spawns Python
        ├── preload.js    ← Secure IPC bridge
        └── index.html    ← Full UI (HTML/CSS/JS, no framework)
```

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.8–3.10 | Sybil inference backend |
| Node.js | ≥ 16 | Electron runtime |
| npm | ≥ 8 | Package management |

---

## Installation & Setup

### Step 1 — Python environment

**macOS / Linux:**
```bash
bash setup.sh
```

**Windows:**
```
setup.bat
```

This creates a `.venv/` virtual environment, installs Flask + Sybil from the bundled source.

### Step 2 — Electron dependencies

```bash
cd electron_app
npm install
```

### Step 3 — Run the app

```bash
npm start
```

---

## Using Local Checkpoints

The app is designed to work with **pre-downloaded** Sybil checkpoint files. Download them from:

- **GitHub Releases:** https://github.com/reginabarzilaygroup/Sybil/releases
- **Google Drive:** https://drive.google.com/drive/folders/1nBp05VV9mf5CfEO6W5RY4ZpcpxmPDEeR

Place all `.ckpt` files and the calibrator `.json` file in one directory, e.g.:
```
~/sybil_checkpoints/
├── 28a7cd44f5bcd3e6cc760b65c7e0d54d.ckpt
├── 56ce1a7d241dc342982f5466c4a9d7ef.ckpt
├── 624407ef8e3a2a009f9fa51f9846fe9a.ckpt
├── 64a91b25f84141d32852e75a3aec7305.ckpt
├── 65fd1f04cb4c5847d86a9ed8ba31ac1a.ckpt
└── sybil_ensemble_simple_calibrator.json
```

In the app: choose **Local Checkpoints**, browse to this directory, click **Load Model**.

---

## Input Data Requirements

- **DICOM:** Axial LDCT series (one exam per folder), slice thickness ≤ 5mm
  - First frame = abdominal region, last frame = clavicles (auto-sorted by z-position)
  - Supported extensions: `.dcm`, `.dicom`, `.ima` (and without extension)
- **PNG:** Axially-ordered PNG slices (must be in correct anatomical order)
  - Voxel spacing must be known

---

## Prediction Output

Sybil outputs **6 risk scores** (one per year), calibrated probabilities of lung cancer development within that many years from the scan date.

| Score | Meaning |
|-------|---------|
| Year 1 | Probability of cancer within 1 year |
| Year 2 | Probability of cancer within 2 years |
| ... | ... |
| Year 6 | Probability of cancer within 6 years |

Risk classification used in the UI:
- **Low** < 5%
- **Moderate** 5–15%
- **High** > 15%

---

## Attention Maps

Enable **Attention Maps** in the sidebar before running. The model will generate GIF overlays showing which slices and pixels contributed most to the prediction. These are saved in a `sybil_attentions/` subfolder of your DICOM directory.

---

## Building a Distributable Package

```bash
cd electron_app

# macOS DMG
npm run dist

# Windows NSIS installer
npm run dist -- --win

# Linux AppImage
npm run dist -- --linux
```

The Python backend and Sybil source are bundled as `extraResources` in the packaged app.

---


## New in this patched build

This build adds a **paper-inspired audit toolkit** informed by the 2026 preprint *Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions*.

What was added safely:
- **Audit Toolkit tab** in the results screen.
- **Proxy attention diagnostics** for:
  - outside-body attention,
  - boundary-dominant attention,
  - peripheral-ring attention,
  - superior vs inferior slice attention,
  - left-right asymmetry,
  - top-1 / top-3 slice dominance.
- **Flagged high-attention slice list** for targeted visual review.

What was intentionally **not** changed:
- The underlying Sybil prediction path.
- Model weights, inference logic, calibration flow, or checkpoint handling.

Important limitation:
- These additions are **not** a full implementation of SHNAP or SNAP.
- They are **non-causal proxy checks** computed from attention maps already produced by the app.
- They should be used as review aids, not as proof of lesion-level causality.

## Citation

```
@article{mikhael2023sybil,
  title={Sybil: a validated deep learning model to predict future lung cancer risk from a single low-dose chest computed tomography},
  author={Mikhael, Peter G and Wohlwend, Jeremy and Yala, Adam et al.},
  journal={Journal of Clinical Oncology},
  year={2023},
  publisher={Wolters Kluwer Health}
}
```

---

## Disclaimer

**FOR RESEARCH USE ONLY. Not intended for clinical diagnosis or treatment decisions.**
