#!/usr/bin/env python
"""Evaluate nnU-Net label-map predictions against pennycress test labels."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = {
    1: "wing",
    2: "envelope",
    3: "seed",
}

RGB_COLORS = {
    0: (255, 255, 255),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate nnU-Net integer predictions against integer labels."
    )
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rgb-output-dir",
        type=Path,
        default=None,
        help="Optional directory for RGB copies of nnU-Net predictions.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log summary metrics to Weights & Biases.",
    )
    parser.add_argument("--wandb-project", default="pennycress-nnunet")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    return parser.parse_args()


def load_label(path: Path) -> np.ndarray:
    label = np.asarray(Image.open(path))
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.uint8)


def iou(y_true: np.ndarray, y_pred: np.ndarray, empty_score: float = 1.0) -> float:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return empty_score
    return float(intersection / union)


def dice(y_true: np.ndarray, y_pred: np.ndarray, empty_score: float = 1.0) -> float:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    denominator = y_true.sum() + y_pred.sum()
    if denominator == 0:
        return empty_score
    return float(2 * np.logical_and(y_true, y_pred).sum() / denominator)


def prediction_path_for_label(pred_dir: Path, label_path: Path) -> Path:
    candidates = [
        pred_dir / label_path.name,
        pred_dir / f"{label_path.stem}_0000{label_path.suffix}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No prediction found for {label_path.name}. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def write_rgb_prediction(prediction: np.ndarray, output_path: Path) -> None:
    rgb = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    for label_value, color in RGB_COLORS.items():
        rgb[prediction == label_value] = color
    Image.fromarray(rgb).save(output_path)


def summarize(rows: list[dict[str, float | str]]) -> dict[str, float | int]:
    metric_keys = [key for key in rows[0] if key != "case_id"]
    summary: dict[str, float | int] = {"num_cases": len(rows)}
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"mean_{key}"] = float(values.mean())
        summary[f"std_{key}"] = float(values.std(ddof=0))
    return summary


def evaluate(pred_dir: Path, label_dir: Path, rgb_output_dir: Path | None) -> list[dict[str, float | str]]:
    label_paths = sorted(label_dir.glob("*.png"))
    if not label_paths:
        raise FileNotFoundError(f"No PNG labels found in {label_dir}")

    if rgb_output_dir is not None:
        rgb_output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    for label_path in label_paths:
        pred_path = prediction_path_for_label(pred_dir, label_path)
        label = load_label(label_path)
        prediction = load_label(pred_path)

        if label.shape != prediction.shape:
            raise ValueError(
                f"Shape mismatch for {label_path.name}: "
                f"label={label.shape}, prediction={prediction.shape}"
            )

        if rgb_output_dir is not None:
            write_rgb_prediction(prediction, rgb_output_dir / label_path.name)

        row: dict[str, float | str] = {"case_id": label_path.stem}
        for class_value, class_name in CLASS_NAMES.items():
            row[f"{class_name}_iou"] = iou(label == class_value, prediction == class_value)
            row[f"{class_name}_dice"] = dice(label == class_value, prediction == class_value)

        # This mirrors utils/SegmentImage.py, where envelope IoU is computed as
        # mask[:, :, 1:].sum(-1), combining envelope and seed pixels.
        row["envelope_with_seed_iou"] = iou(label >= 2, prediction >= 2)
        row["envelope_with_seed_dice"] = dice(label >= 2, prediction >= 2)
        row["foreground_iou"] = iou(label > 0, prediction > 0)
        row["foreground_dice"] = dice(label > 0, prediction > 0)
        rows.append(row)

    return rows


def write_outputs(rows: list[dict[str, float | str]], output_dir: Path) -> dict[str, float | int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)

    csv_path = output_dir / "per_case_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    return summary


def log_to_wandb(args: argparse.Namespace, summary: dict[str, float | int]) -> None:
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode,
        name=args.wandb_run_name,
        group=args.wandb_group,
        job_type="external-test",
        config={
            "pred_dir": str(args.pred_dir),
            "label_dir": str(args.label_dir),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        },
    )
    run.log({f"external_test/{key}": value for key, value in summary.items()})
    run.finish()


def main() -> None:
    args = parse_args()
    rows = evaluate(args.pred_dir, args.label_dir, args.rgb_output_dir)
    summary = write_outputs(rows, args.output_dir)
    if args.wandb:
        log_to_wandb(args, summary)

    print(f"Wrote per-case metrics to {args.output_dir / 'per_case_metrics.csv'}")
    print(f"Wrote summary metrics to {args.output_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
