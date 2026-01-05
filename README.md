# Physics-Informed Deep Learning for Automated Solar Flare Detection in CHASE Hα Observations

This repository contains the official implementation of the paper: **"Physics-Informed Deep Learning for Automated Solar Flare Detection in CHASE Hα Observations"**.

Our framework integrates solar physics diagnostics (Hα line profiles) with a customized YOLO11 architecture to achieve high-accuracy, real-time solar flare detection.

## 🚀 Project Architecture

This project is built upon a modified version of the [Ultralytics](https://github.com/ultralytics/ultralytics) framework.

* `/ultralytics`: The core YOLO source code, modified to support multi-channel "physics-aware" inputs and Frequency-Guided Attention.
* `gs_batch.py`: Script for Physics-Aware Dimensionality Reduction (Gaussian Fitting). It compresses 118-channel Hα spectral data into 3-channel physical parameter maps (I_{core}, V_{dop}, FWHM).

## 📊 Dataset

Due to the massive volume of the raw CHASE data (>1 TB for the full mission), we provide data in two tiers:

1. **Publicly Available (Google Drive)**:
* **Pre-processed Data**: 3-channel "fitted images" derived from Gaussian fitting.
* **Labels**: Expert-refined flare detection annotations in YOLO format.
* *Link: [Data in YOLO format](https://drive.google.com/file/d/1Pc3fXlNYBFqidHVZNHbaxJkN7iEiN27T/view?usp=drive_link)*


2. **Raw Data & Full Set**:
* A **Sample Raw Data** (one full-disk FITS file) is included in the Drive for testing the `gs_batch.py` script.
* **Full 118-channel Dataset**: Due to the 1TB+ size, we recommend contacting us via email.



## 🛠 Installation & Usage

### 1. Environment Setup

```bash
pip install -e .

```

### 2. Data Compression (Physics-Aware Reduction)

To process raw CHASE FITS files into physical maps:

```bash
python gs_batch.py

```

### 3. Training & Inference

Since the core is based on a modified Ultralytics, you can use the standard CLI.

