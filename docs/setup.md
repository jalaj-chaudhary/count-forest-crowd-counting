# Setup & reproduction

## 1. Requirements

- **Python** 3.10 or newer (the project used 3.13)
- **Hardware:** a CPU machine with at least 7 GB of RAM. No GPU is needed.
- **Python packages:** install them with

```bash
pip install numpy scipy opencv-python joblib matplotlib jupyter
```

## 2. Get the data

The notebook's first section ("0.1 Get the data") downloads the Mall dataset. You can also download it by hand and point the dataset path in the configuration cell at it. The dataset contains:

- 2000 frames (640×480);
- the head annotations;
- the ROI;
- the perspective map.

## 3. Run

```bash
git clone https://github.com/jalaj-chaudhary/count-forest-crowd-counting.git
cd count-forest-crowd-counting
jupyter notebook count_forest_full.ipynb
```

The notebook has one switch in the configuration cell, `TUNING`:

| `TUNING` | Trains on | Evaluates on | Used for |
|---|---|---|---|
| `True` | frames 1–700 | frames 701–800 | choosing hyperparameters, comparing variants (section 5.2) |
| `False` | frames 1–800 | test frames 801–2000 | final test results (5.3–5.5), applications and demo video (Phase 6) |

Run it twice:

1. **Tuning run.** Set `TUNING = True` and run all cells.
2. **Final run.** Set `TUNING = False` and run all cells again.

The Phase 8 ablations always use frames 1–700 and 701–800, whatever `TUNING` is set to. Anything expensive is cached in `cache/`, so re-runs are fast.

## 4. Approximate cost (2 CPU cores)

| Step | Time |
|---|---|
| Training pool, 800 frames | about 3 min |
| Training 32 trees | about 2 min |
| Four variants over the 1200 test frames | about 30 min |
| Density estimation (plain forest / forest permutation) | 360 / 195 ms per frame |

## 5. Notebook map

| Phase | Contents |
|---|---|
| 1 | Data loaders, ground-truth density (Eq. 1), patch labels (Eq. 3), metrics |
| 2 | The 15 feature channels and perspective weighting |
| 3 | Patch pool, split criteria (Eq. 5, 6), tree growth, leaf model, fast density estimator (Eq. 7–9) |
| 4 | Crowdedness prior and forest permutation |
| 5 | Validation and test evaluation, plots |
| 6 | Heat map, stationary time, demo video |
| 8 | Ablations |
