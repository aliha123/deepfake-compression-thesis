# Deepfake Compression Robustness Study

A small, reproducible research codebase for studying how lossy image compression can affect deepfake-detection pipelines.

The goal is not to claim a specific benchmark result. The repository provides the experiment utilities needed to compare detector behavior across compression levels in a consistent way.

## Why this matters

Images and video frames are often recompressed before they reach a detector. Social platforms, messaging apps, storage systems, and preprocessing pipelines can all change the visual signal a model receives.

A detector that performs well on clean data may behave differently after JPEG or WebP compression. This project makes that shift easier to measure.

## Current capabilities

- Apply controlled JPEG and WebP compression at configurable quality levels
- Generate compression sweeps for the same input image
- Evaluate binary detector scores with accuracy, F1, and ROC AUC
- Compare performance across compression qualities
- Run tests automatically in GitHub Actions
- Keep raw datasets and generated experiment outputs outside version control

## Repository structure

```text
.
├── src/
│   ├── compression.py      # image compression transforms
│   ├── metrics.py          # classification and robustness metrics
│   └── smoke_test.py       # environment check
├── scripts/
│   └── evaluate_scores.py  # CSV-based compression evaluation
├── tests/
│   ├── test_compression.py
│   └── test_metrics.py
├── data/                   # local datasets, ignored by git
├── results/                # generated outputs, ignored by git
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/aliha123/deepfake-compression-thesis.git
cd deepfake-compression-thesis

python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Run the test suite:

```bash
pytest -q
```

## Compression example

```python
from PIL import Image
from src.compression import build_compression_grid

image = Image.open("data/example.jpg")

compressed = build_compression_grid(
    image,
    qualities=[95, 80, 60, 40, 20],
    image_format="JPEG",
)

for quality, result in compressed.items():
    result.save(f"results/example_q{quality}.jpg")
```

## Evaluating detector predictions

The evaluation script expects a CSV containing:

| column | description |
| --- | --- |
| `label` | ground-truth class, 0 or 1 |
| `score` | detector probability/confidence for class 1 |
| `quality` | compression quality for that sample |

Example:

```csv
label,score,quality
0,0.12,95
1,0.91,95
0,0.31,60
1,0.73,60
```

Run:

```bash
python scripts/evaluate_scores.py predictions.csv --output results/metrics.csv
```

The generated table reports sample count, accuracy, F1, and ROC AUC for each compression quality.

## Research workflow

A typical experiment can follow this sequence:

1. Select a deepfake detector and evaluation dataset.
2. Produce compressed copies of each sample at fixed quality levels.
3. Run inference using the same detector checkpoint and preprocessing logic.
4. Store labels, model scores, and compression quality in a CSV.
5. Use `scripts/evaluate_scores.py` to compare metrics across conditions.
6. Inspect where performance degrades and investigate failure cases.

## Next directions

- Add detector adapters for image and video models
- Measure calibration shift in addition to classification performance
- Compare codec families and resolution changes
- Add frame-level aggregation for video experiments
- Visualize robustness curves and confidence distributions
- Track experiment configuration and model checkpoints

## Tech

Python • PyTorch • torchvision • Pillow • OpenCV • pandas • NumPy • scikit-learn • pytest • GitHub Actions

## Notes

Datasets, model weights, and generated experiment outputs are intentionally not committed to this repository. Reproducibility depends on documenting the exact dataset split, model checkpoint, preprocessing configuration, and compression settings used for each experiment.
