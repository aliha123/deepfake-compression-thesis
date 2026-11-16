"""Evaluate prediction scores across compression levels.

Expected CSV columns:
- label: 0 for real, 1 for synthetic/deepfake
- score: model probability or confidence for the positive class
- quality: compression quality used for the sample
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.metrics import evaluate_binary_predictions


REQUIRED_COLUMNS = {"label", "score", "quality"}


def evaluate_scores(frame: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    rows: list[dict[str, float | int | None]] = []
    for quality, group in frame.groupby("quality", sort=False):
        metrics = evaluate_binary_predictions(
            group["label"].to_numpy(),
            group["score"].to_numpy(),
            threshold=threshold,
        )
        rows.append(
            {
                "quality": int(quality),
                "samples": int(len(group)),
                "accuracy": metrics.accuracy,
                "f1": metrics.f1,
                "roc_auc": metrics.roc_auc,
            }
        )

    return pd.DataFrame(rows).sort_values("quality", ascending=False).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deepfake detector scores across compression qualities."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/metrics.csv"))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    results = evaluate_scores(frame, threshold=args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
