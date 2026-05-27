#!/usr/bin/env python
"""Run DeepLabV3+ inference and test-set evaluation for pennycress pods."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

import utils.Measure as Measure
import utils.SegmentImage as SegmentImage


DEFAULT_MODEL_KWARGS = {
    "encoder_name": "resnet101",
    "encoder_weights": None,
    "in_channels": 3,
    "classes": 4,
    "activation": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepLabV3+ inference on the pennycress test set and compute IoU metrics."
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/deeplabv3_plus_up"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/test/test_images"))
    parser.add_argument("--mask-dir", type=Path, default=Path("data/test/test_masks_preproc"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("data/test/deeplabv3_plus_test_predictions"))
    parser.add_argument(
        "--pod-prediction-dir",
        type=Path,
        default=Path("data/test/deeplabv3_plus_test_predictions_by_pod"),
    )
    parser.add_argument("--pod-mask-dir", type=Path, default=Path("data/test/test_masks_by_pod"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/test/deeplabv3_plus_test_metrics"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--measure-pods", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def path_with_sep(path: Path) -> str:
    path_str = str(path)
    return path_str if path_str.endswith(os.sep) else path_str + os.sep


def checkpoint_sort_key(path: Path) -> int:
    match = re.search(r"checkpoint_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"), key=checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_*.pt files found in {checkpoint_dir}")
    return checkpoints[-1]


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    checkpoint = load_checkpoint(checkpoint_path, device)
    model_kwargs = dict(checkpoint.get("kwargs", DEFAULT_MODEL_KWARGS))
    model_kwargs["encoder_weights"] = None

    model = smp.DeepLabV3Plus(**model_kwargs)
    model.load_state_dict(strip_module_prefix(checkpoint["model"]))
    model.to(device)
    model.eval()
    return model


def labeled_image_names(image_dir: Path, mask_dir: Path | None) -> list[str]:
    source_dir = mask_dir if mask_dir is not None and mask_dir.is_dir() else image_dir
    names = sorted(path.name for path in source_dir.glob("*.png"))
    if not names:
        raise FileNotFoundError(f"No PNG files found in {source_dir}")

    missing = [name for name in names if not (image_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image files listed by {source_dir} are missing from {image_dir}. "
            f"First missing file: {missing[0]}"
        )
    return names


def write_combined_summary(output_dir: Path, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint
    if checkpoint_path is None and not args.skip_inference:
        checkpoint_path = latest_checkpoint(args.checkpoint_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    args.prediction_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_inference:
        if checkpoint_path is None:
            raise ValueError("A checkpoint is required unless --skip-inference is set.")
        model = load_model(checkpoint_path, device)
        image_names = labeled_image_names(args.image_dir, args.mask_dir)
        SegmentImage.segment_image(
            model=model,
            image_path=path_with_sep(args.image_dir),
            mask_path=None,
            save_path=path_with_sep(args.prediction_dir),
            image_names=image_names,
            plot=args.plot,
            verbose=0,
            device=device,
        )

    combined_summary = {
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "prediction_dir": str(args.prediction_dir),
    }

    if args.mask_dir is not None:
        _, full_summary = Measure.evaluate_prediction_directory(
            pred_dir=args.prediction_dir,
            mask_dir=args.mask_dir,
            output_csv=args.output_dir / "full_image_metrics.csv",
            summary_json=args.output_dir / "full_image_summary.json",
            pred_prefix="pred_",
        )
        combined_summary["full_image"] = full_summary

    if args.measure_pods:
        Measure.measure_prediction_directory(
            pred_path=args.prediction_dir,
            input_path=args.image_dir,
            pod_save_path=args.pod_prediction_dir,
            output_csv=args.output_dir / "pod_measurements.csv",
            plot=args.plot,
        )

        if args.pod_mask_dir is not None and args.pod_mask_dir.is_dir():
            _, pod_summary = Measure.evaluate_prediction_directory(
                pred_dir=args.pod_prediction_dir,
                mask_dir=args.pod_mask_dir,
                output_csv=args.output_dir / "pod_metrics.csv",
                summary_json=args.output_dir / "pod_summary.json",
                pred_prefix="pred_",
            )
            combined_summary["pod"] = pod_summary

    write_combined_summary(args.output_dir, combined_summary)
    print(json.dumps(combined_summary, indent=2))


if __name__ == "__main__":
    main()
