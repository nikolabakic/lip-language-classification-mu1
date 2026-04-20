# Lip-Reading Language Classification (MU1)

Visual speech classification — Serbian vs English — based solely on lip movement in anonymized video recordings.  
Course project for **Mašinsko Učenje 1 (MU1)**, Faculty of Technical Sciences, Novi Sad.

---

## Task

Binary classification: given a short video of a speaker's lip region, predict whether the spoken language is **Serbian (0)** or **English (1)**.  
No audio is used — the model relies entirely on visual lip dynamics.

## Dataset

- 30 speakers (`spk01`–`spk30`), each with Serbian and English recordings
- ~4 700 video samples total
- Speaker-independent split: 5 speakers held out for testing (`spk04`, `spk08`, `spk14`, `spk26`, `spk28`)
- Labels verified — no train/test speaker overlap

## Feature Extraction

For each video frame (sampled every 2nd frame, up to 150 frames):

| Feature group | Description |
|---|---|
| **Geometry** | Mouth bounding-box height, width, aspect ratio, area |
| **HOG** | Histogram of Oriented Gradients on 96×96 lip ROI (12 orientations, 8×8 cells, L2-Hys norm) |
| **Temporal HOG** | Frame-to-frame HOG difference (motion texture) |
| **Dynamics** | Peak count, speech rate, opening range/mean/std/ratio, velocity/acceleration statistics |

Temporal pooling: mean, std (and max for HOG) across frames → single 54 018-dim vector per video.

## Models & Results

All models use `StandardScaler` + `PCA (whiten=True)` as preprocessing.

| Model | Best config | Test Accuracy |
|---|---|---|
| **SVM (RBF)** | PCA=120, C=0.5 | **65.4%** |
| MLP | PCA=150, arch=(256,128), lr=0.001 | 63.0% |
| Random Forest | PCA=120, 500 trees, depth=None | 61.5% |

Best overall: **SVM with 65.4% accuracy** (random baseline = 50%).

## Repository Structure

```
MU1.ipynb          # Full pipeline: feature extraction → training → evaluation
izvestaj MU1.pdf   # Written report (Serbian)
```

## Requirements

```
opencv-python
numpy
pandas
scikit-image
scikit-learn
scipy
matplotlib
tqdm
openpyxl
joblib
```

Install with:
```bash
pip install opencv-python numpy pandas scikit-image scikit-learn scipy matplotlib tqdm openpyxl joblib
```

## Usage

Place the `speakers/` directory (dataset) next to the notebook, then run all cells in `MU1.ipynb`.  
Pre-extracted features can be saved/loaded as `.npy` files to skip the slow extraction step (~5 s/video).
