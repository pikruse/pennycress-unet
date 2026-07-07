#!/usr/bin/env python
"""Paired statistical tests for full-image model performance.

The five test images are treated as matched blocks. For each metric, the script
runs a Friedman omnibus test across model results, followed by paired Wilcoxon
signed-rank tests for all model pairs with Holm correction.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


DEFAULT_METRICS = [
    "wing_iou",
    "envelope_iou",
    "seed_iou",
    "miou",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_test_root = repo_root / "data" / "test"
    parser = argparse.ArgumentParser(
        description=(
            "Run paired Friedman and Holm-corrected Wilcoxon tests on "
            "full-image model metrics."
        )
    )
    parser.add_argument("--test-root", type=Path, default=default_test_root)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help=(
            "Metrics to test. Defaults to tissue-scale wing/envelope/seed IoU "
            "plus overall miou."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model_result names to include.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_test_root / "model_performance_stats",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm adjusted p-values in the original order."""
    n = len(p_values)
    if n == 0:
        return []

    order = sorted(range(n), key=lambda idx: p_values[idx])
    adjusted_sorted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = min((n - rank) * p_values[idx], 1.0)
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = running_max

    adjusted = [0.0] * n
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted


def load_full_image_metrics(test_root: Path, selected_models: set[str] | None) -> pd.DataFrame:
    frames = []
    for metrics_path in sorted(test_root.glob("*_test_metrics/full_image_metrics.csv")):
        model_result = metrics_path.parent.name.removesuffix("_test_metrics")
        if selected_models is not None and model_result not in selected_models:
            continue
        frame = pd.read_csv(metrics_path)
        frame.insert(0, "model_result", model_result)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No full_image_metrics.csv files found under {test_root}")
    return pd.concat(frames, ignore_index=True)


def validate_complete_blocks(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    table = data.pivot(index="image_name", columns="model_result", values=metric)
    missing = table.isna().sum().sum()
    if missing:
        raise ValueError(f"Metric {metric} has {missing} missing model/image values.")
    return table


def friedman_row(metric: str, table: pd.DataFrame) -> dict[str, object]:
    statistic, p_value = friedmanchisquare(*(table[column].to_numpy() for column in table.columns))
    return {
        "metric": metric,
        "num_models": len(table.columns),
        "num_images": len(table),
        "friedman_chi_square": float(statistic),
        "p_value": float(p_value),
    }


def wilcoxon_rows(metric: str, table: pd.DataFrame, alpha: float) -> list[dict[str, object]]:
    raw_rows = []
    for model_a, model_b in combinations(table.columns, 2):
        values_a = table[model_a].to_numpy(dtype=float)
        values_b = table[model_b].to_numpy(dtype=float)
        differences = values_a - values_b
        if np.allclose(differences, 0):
            statistic = 0.0
            p_value = 1.0
        else:
            statistic, p_value = wilcoxon(
                values_a,
                values_b,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
        raw_rows.append(
            {
                "metric": metric,
                "model_a": model_a,
                "model_b": model_b,
                "mean_a": float(np.mean(values_a)),
                "mean_b": float(np.mean(values_b)),
                "mean_difference_a_minus_b": float(np.mean(differences)),
                "median_difference_a_minus_b": float(np.median(differences)),
                "wilcoxon_statistic": float(statistic),
                "p_value": float(p_value),
            }
        )

    adjusted = holm_adjust([float(row["p_value"]) for row in raw_rows])
    for row, adjusted_p in zip(raw_rows, adjusted):
        row["p_value_holm"] = adjusted_p
        row["reject_holm_alpha"] = adjusted_p < alpha
    return raw_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    selected_models = set(args.models) if args.models else None
    data = load_full_image_metrics(args.test_root, selected_models)

    omnibus_rows = []
    pairwise_rows = []
    for metric in args.metrics:
        table = validate_complete_blocks(data, metric)
        omnibus_rows.append(friedman_row(metric, table))
        pairwise_rows.extend(wilcoxon_rows(metric, table, args.alpha))

    omnibus_adjusted = holm_adjust([float(row["p_value"]) for row in omnibus_rows])
    for row, adjusted_p in zip(omnibus_rows, omnibus_adjusted):
        row["p_value_holm_across_metrics"] = adjusted_p
        row["reject_holm_alpha"] = adjusted_p < args.alpha

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "friedman_omnibus.csv", omnibus_rows)
    write_csv(args.output_dir / "wilcoxon_pairwise_holm.csv", pairwise_rows)
    print(f"Wrote omnibus tests to {args.output_dir / 'friedman_omnibus.csv'}")
    print(f"Wrote pairwise tests to {args.output_dir / 'wilcoxon_pairwise_holm.csv'}")


if __name__ == "__main__":
    main()
