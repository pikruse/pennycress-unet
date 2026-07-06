#!/usr/bin/env python
"""Summarize full-image test metric means and standard deviations.

The default output matches data/test/full_image_model_result_sd_summary.csv.
Standard deviations are computed across the five full test images with ddof=0,
matching the existing summary.json files written by scripts/inference.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


METRIC_COLUMNS = [
    "wing_iou",
    "envelope_iou",
    "seed_iou",
    "envelope_with_seed_iou",
    "foreground_iou",
    "miou",
]

OUTPUT_COLUMNS = [
    "model_result",
    "num_test_images",
    "sd_ddof",
    "source_type",
    "mean_wing_iou",
    "sd_wing_iou",
    "mean_envelope_iou",
    "sd_envelope_iou",
    "mean_seed_iou",
    "sd_seed_iou",
    "mean_envelope_with_seed_iou",
    "sd_envelope_with_seed_iou",
    "mean_foreground_iou",
    "sd_foreground_iou",
    "mean_miou",
    "sd_miou",
    "source_path",
]

DEFAULT_LEGACY_PREDICTION_DIRS = [
    "down_test_predictions",
    "up_test_predictions",
    "vanilla_test_predictions",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_test_root = repo_root / "data" / "test"

    parser = argparse.ArgumentParser(
        description=(
            "Create a consolidated full-image model-result SD summary from "
            "per-image metrics and legacy prediction folders."
        )
    )
    parser.add_argument("--test-root", type=Path, default=default_test_root)
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Mask directory for legacy prediction evaluation. Defaults to TEST_ROOT/test_masks_preproc.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Defaults to "
            "TEST_ROOT/full_image_model_result_sd_summary.csv."
        ),
    )
    parser.add_argument(
        "--expected-images",
        type=int,
        default=5,
        help="Require this many full-image test rows per model result.",
    )
    parser.add_argument(
        "--skip-legacy-predictions",
        action="store_true",
        help="Only summarize existing *_test_metrics/full_image_metrics.csv files.",
    )
    return parser.parse_args()


def read_metric_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def display_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)


def require_metric_columns(rows: list[dict[str, object]], source_path: Path) -> None:
    if not rows:
        raise ValueError(f"{source_path} contains no metric rows.")
    missing = [column for column in METRIC_COLUMNS if column not in rows[0]]
    if missing:
        raise ValueError(f"{source_path} is missing columns: {missing}")


def summarize_metric_rows(
    *,
    model_result: str,
    rows: list[dict[str, object]],
    source_type: str,
    source_path: Path,
    expected_images: int,
) -> dict[str, object]:
    require_metric_columns(rows, source_path)
    if len(rows) != expected_images:
        raise ValueError(
            f"{source_path} has {len(rows)} rows; expected {expected_images} test images."
        )

    summary: dict[str, object] = {
        "model_result": model_result,
        "num_test_images": len(rows),
        "sd_ddof": 0,
        "source_type": source_type,
        "source_path": str(source_path),
    }
    for metric in METRIC_COLUMNS:
        values = np.array([float(row[metric]) for row in rows], dtype=np.float64)
        summary[f"mean_{metric}"] = float(values.mean())
        summary[f"sd_{metric}"] = float(values.std(ddof=0))
    return summary


def load_mask_channels(path: Path) -> dict[str, np.ndarray]:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 2:
        return {
            "wing": mask == 1,
            "envelope": mask == 2,
            "seed": mask == 3,
        }

    mask = mask[:, :, :3]
    white_background = mask.sum(axis=2) == 255 * 3
    mask = mask.copy()
    mask[white_background] = 0
    return {
        "wing": mask[:, :, 0] > 127,
        "envelope": mask[:, :, 1] > 127,
        "seed": mask[:, :, 2] > 127,
    }


def binary_iou(y_true: np.ndarray, y_pred: np.ndarray, empty_score: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return empty_score
    return float(intersection / union)


def segmentation_iou_metrics(mask_path: Path, pred_path: Path) -> dict[str, float]:
    gt = load_mask_channels(mask_path)
    pred = load_mask_channels(pred_path)

    metrics = {}
    per_class_ious = []
    for class_name in ("wing", "envelope", "seed"):
        score = binary_iou(gt[class_name], pred[class_name])
        metrics[f"{class_name}_iou"] = score
        per_class_ious.append(score)

    metrics["envelope_with_seed_iou"] = binary_iou(
        np.logical_or(gt["envelope"], gt["seed"]),
        np.logical_or(pred["envelope"], pred["seed"]),
    )
    metrics["foreground_iou"] = binary_iou(
        np.logical_or.reduce((gt["wing"], gt["envelope"], gt["seed"])),
        np.logical_or.reduce((pred["wing"], pred["envelope"], pred["seed"])),
    )
    metrics["miou"] = float(np.mean(per_class_ious))
    return metrics


def prediction_path_for_mask(pred_dir: Path, mask_path: Path) -> Path:
    candidates = [
        pred_dir / f"pred_{mask_path.name}",
        pred_dir / mask_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No prediction found for {mask_path.name}. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def evaluate_prediction_dir(pred_dir: Path, mask_dir: Path) -> list[dict[str, object]]:
    mask_paths = sorted(mask_dir.glob("*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No PNG masks found in {mask_dir}")

    rows = []
    for mask_path in mask_paths:
        pred_path = prediction_path_for_mask(pred_dir, mask_path)
        row: dict[str, object] = {"image_name": mask_path.name}
        row.update(segmentation_iou_metrics(mask_path, pred_path))
        rows.append(row)
    return rows


def gather_metric_csv_summaries(
    test_root: Path,
    expected_images: int,
) -> list[dict[str, object]]:
    summaries = []
    for metrics_csv in sorted(test_root.glob("*_test_metrics/full_image_metrics.csv")):
        model_result = metrics_csv.parent.name.removesuffix("_test_metrics")
        summaries.append(
            summarize_metric_rows(
                model_result=model_result,
                rows=read_metric_rows(metrics_csv),
                source_type="full_image_metrics_csv",
                source_path=metrics_csv,
                expected_images=expected_images,
            )
        )
    return summaries


def gather_legacy_prediction_summaries(
    test_root: Path,
    mask_dir: Path,
    expected_images: int,
) -> list[dict[str, object]]:
    summaries = []
    for dirname in DEFAULT_LEGACY_PREDICTION_DIRS:
        pred_dir = test_root / dirname
        if not pred_dir.is_dir():
            continue
        model_result = pred_dir.name.removesuffix("_test_predictions")
        summaries.append(
            summarize_metric_rows(
                model_result=model_result,
                rows=evaluate_prediction_dir(pred_dir, mask_dir),
                source_type="legacy_full_image_prediction_dir",
                source_path=pred_dir,
                expected_images=expected_images,
            )
        )
    return summaries


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    test_root = args.test_root.resolve()
    mask_dir = (args.mask_dir or test_root / "test_masks_preproc").resolve()
    output = (args.output or test_root / "full_image_model_result_sd_summary.csv").resolve()

    summaries = gather_metric_csv_summaries(test_root, args.expected_images)
    if not args.skip_legacy_predictions:
        summaries.extend(
            gather_legacy_prediction_summaries(test_root, mask_dir, args.expected_images)
        )

    summaries.sort(key=lambda row: str(row["model_result"]))
    for row in summaries:
        row["source_path"] = display_path(Path(str(row["source_path"])), repo_root)
    write_rows(output, summaries)
    print(f"Wrote {len(summaries)} model result rows to {output}")


if __name__ == "__main__":
    main()
