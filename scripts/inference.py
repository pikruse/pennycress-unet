#!/usr/bin/env python
"""Run model-agnostic inference and evaluation for pennycress segmentation."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    kind: str
    prediction_format: str
    checkpoint_dir_template: str | None = None
    default_prediction_dir: str | None = None
    default_mask_dir: str | None = None
    default_output_dir: str | None = None
    default_rgb_output_dir: str | None = None


MODEL_SPECS = {
    "unet": ModelSpec(
        key="unet",
        label="unet",
        kind="torch",
        prediction_format="rgb",
        checkpoint_dir_template="checkpoints/unet_{distance_weights}",
    ),
    "deeplabv3plus": ModelSpec(
        key="deeplabv3plus",
        label="deeplabv3_plus",
        kind="torch",
        prediction_format="rgb",
        checkpoint_dir_template="checkpoints/deeplabv3_plus_{distance_weights}",
    ),
    "segformer": ModelSpec(
        key="segformer",
        label="segformer_mit_b5",
        kind="torch",
        prediction_format="rgb",
        checkpoint_dir_template="checkpoints/segformer_mit_b5_{distance_weights}",
    ),
    "nnunet": ModelSpec(
        key="nnunet",
        label="nnunet",
        kind="predictions",
        prediction_format="label-map",
        default_prediction_dir="nnu_net/predictions/Dataset501_PennycressPods/2d/fold_0/checkpoint_best",
        default_mask_dir="nnu_net/nnUNet_raw/Dataset501_PennycressPods/labelsTs",
        default_output_dir="nnu_net/metrics/Dataset501_PennycressPods/2d/fold_0/checkpoint_best",
        default_rgb_output_dir="nnu_net/predictions_rgb/Dataset501_PennycressPods/2d/fold_0/checkpoint_best",
    ),
    "external": ModelSpec(
        key="external",
        label="external",
        kind="predictions",
        prediction_format="rgb",
    ),
    "sam3": ModelSpec(
        key="sam3",
        label="sam3",
        kind="predictions",
        prediction_format="rgb",
    ),
    "mask2former": ModelSpec(
        key="mask2former",
        label="mask2former_swin_large_ade_semantic",
        kind="torch",
        prediction_format="rgb",
        checkpoint_dir_template="checkpoints/mask2former_swin_large_ade_semantic_{distance_weights}",
    ),
}

MODEL_ALIASES = {
    "auto": "auto",
    "unet": "unet",
    "u_net": "unet",
    "deeplab": "deeplabv3plus",
    "deeplabv3": "deeplabv3plus",
    "deeplabv3+": "deeplabv3plus",
    "deeplabv3plus": "deeplabv3plus",
    "deeplabv3_plus": "deeplabv3plus",
    "segformer": "segformer",
    "segformer_mit_b5": "segformer",
    "mit_b5": "segformer",
    "nnunet": "nnunet",
    "nnu_net": "nnunet",
    "nnunetv2": "nnunet",
    "nnunet_v2": "nnunet",
    "external": "external",
    "predictions": "external",
    "sam3": "sam3",
    "sam_3": "sam3",
    "mask2former": "mask2former",
    "mask_2_former": "mask2former",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference and metrics for registered pennycress segmentation models."
    )
    parser.add_argument(
        "--model",
        default="auto",
        help=(
            "Model family to evaluate. Known values: "
            + ", ".join(["auto", *sorted(MODEL_SPECS)])
        ),
    )
    parser.add_argument("--distance-weights", default=os.getenv("DISTANCE_WEIGHTS", "up"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=Path("data/test/test_images"))
    parser.add_argument("--mask-dir", "--label-dir", dest="mask_dir", type=Path, default=None)
    parser.add_argument("--prediction-dir", type=Path, default=None)
    parser.add_argument("--pod-prediction-dir", type=Path, default=None)
    parser.add_argument("--pod-mask-dir", type=Path, default=Path("data/test/test_masks_by_pod"))
    parser.add_argument("--rgb-output-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--prediction-format",
        choices=("auto", "rgb", "label-map"),
        default="auto",
        help="Use rgb for repo RGB masks or label-map for integer class-map predictions.",
    )
    parser.add_argument("--pred-prefix", default="pred_")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--measure-pods", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def path_with_sep(path: Path) -> str:
    path_str = str(path)
    return path_str if path_str.endswith(os.sep) else path_str + os.sep


def normalize_model_name(name: str | None) -> str:
    if name is None:
        return "auto"

    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    candidates = [
        normalized,
        normalized.replace("_", ""),
        normalized.replace("+", "plus"),
    ]
    for candidate in candidates:
        if candidate in MODEL_ALIASES:
            return MODEL_ALIASES[candidate]

    known = ", ".join(["auto", *sorted(MODEL_SPECS)])
    raise ValueError(f"Unsupported model '{name}'. Known model families: {known}")


def checkpoint_sort_key(path: Path) -> int:
    match = re.search(r"checkpoint_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"), key=checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_*.pt files found in {checkpoint_dir}")
    return checkpoints[-1]


def load_checkpoint(path: Path, device: torch.device) -> dict:
    import torch

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


def detect_model_from_path(path: Path | None) -> str | None:
    if path is None:
        return None
    path_str = str(path).lower()
    if "segformer" in path_str or "mit_b5" in path_str:
        return "segformer"
    if "mask2former" in path_str:
        return "mask2former"
    if "deeplab" in path_str:
        return "deeplabv3plus"
    if "nnunet" in path_str or "nnu_net" in path_str:
        return "nnunet"
    if "unet" in path_str:
        return "unet"
    return None


def detect_model_from_checkpoint(checkpoint: dict, model_kwargs: dict, checkpoint_path: Path) -> str:
    architecture = checkpoint.get("architecture") or checkpoint.get("model_name")
    if architecture is not None:
        return normalize_model_name(str(architecture))

    if "out_channels" in model_kwargs and "layer_sizes" in model_kwargs:
        return "unet"

    encoder_name = str(model_kwargs.get("encoder_name", "")).lower()
    if encoder_name.startswith("mit_"):
        return "segformer"
    if encoder_name:
        return "deeplabv3plus"

    path_model = detect_model_from_path(checkpoint_path)
    if path_model in {"unet", "deeplabv3plus", "segformer", "mask2former"}:
        return path_model

    raise ValueError(
        f"Could not infer model family from checkpoint {checkpoint_path}. "
        "Pass --model explicitly."
    )


def default_checkpoint_dir(model_key: str, distance_weights: str) -> Path:
    spec = MODEL_SPECS[model_key]
    if spec.checkpoint_dir_template is None:
        raise ValueError(f"Model '{model_key}' does not use repo PyTorch checkpoints.")
    return Path(spec.checkpoint_dir_template.format(distance_weights=distance_weights))


def resolve_checkpoint_path(args: argparse.Namespace, requested_model: str) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        if requested_model == "auto":
            raise ValueError(
                "Pass --model with a registered PyTorch model, --checkpoint, "
                "or --checkpoint-dir when running inference."
            )
        checkpoint_dir = default_checkpoint_dir(requested_model, args.distance_weights)

    return latest_checkpoint(checkpoint_dir)


def build_torch_model(model_key: str, model_kwargs: dict) -> torch.nn.Module:
    if model_key == "unet":
        import utils.BuildUNet as BuildUNet

        return BuildUNet.UNet(**model_kwargs)

    if model_key == "mask2former":
        from utils.Mask2FormerSemantic import Mask2FormerSemantic

        model_kwargs = dict(model_kwargs)
        model_kwargs["pretrained"] = False
        model_kwargs["local_files_only"] = True
        return Mask2FormerSemantic(**model_kwargs)

    import segmentation_models_pytorch as smp

    model_kwargs = dict(model_kwargs)
    model_kwargs["encoder_weights"] = None
    builders = {
        "deeplabv3plus": smp.DeepLabV3Plus,
        "segformer": smp.Segformer,
    }
    try:
        return builders[model_key](**model_kwargs)
    except KeyError as exc:
        raise ValueError(
            f"Model '{model_key}' is not registered for tiled PyTorch inference. "
            "Evaluate an existing prediction directory with --skip-inference, or add a builder."
        ) from exc


def load_torch_model(
    checkpoint_path: Path,
    requested_model: str,
    device: torch.device,
) -> tuple[torch.nn.Module, str, dict]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} is not a repo training checkpoint.")

    model_kwargs = dict(checkpoint.get("kwargs", {}))
    if requested_model == "auto":
        model_key = detect_model_from_checkpoint(checkpoint, model_kwargs, checkpoint_path)
    else:
        model_key = requested_model

    if MODEL_SPECS[model_key].kind != "torch":
        raise ValueError(f"Model '{model_key}' cannot be loaded from a repo PyTorch checkpoint.")

    model = build_torch_model(model_key, model_kwargs)
    model.load_state_dict(strip_module_prefix(checkpoint["model"]))
    model.to(device)
    model.eval()
    return model, model_key, checkpoint


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


def run_label(model_key: str) -> str:
    if model_key in MODEL_SPECS:
        return MODEL_SPECS[model_key].label
    return model_key


def default_prediction_dir(model_key: str, distance_weights: str) -> Path:
    spec = MODEL_SPECS[model_key]
    if spec.default_prediction_dir is not None:
        return Path(spec.default_prediction_dir)
    return Path(f"data/test/{spec.label}_{distance_weights}_test_predictions")


def default_pod_prediction_dir(model_key: str, distance_weights: str) -> Path:
    spec = MODEL_SPECS[model_key]
    return Path(f"data/test/{spec.label}_{distance_weights}_test_predictions_by_pod")


def default_mask_dir(model_key: str, prediction_format: str) -> Path:
    spec = MODEL_SPECS[model_key]
    if spec.default_mask_dir is not None:
        return Path(spec.default_mask_dir)
    if prediction_format == "label-map":
        raise ValueError("Pass --label-dir/--mask-dir when evaluating label-map predictions.")
    return Path("data/test/test_masks_preproc")


def default_output_dir(model_key: str, distance_weights: str) -> Path:
    spec = MODEL_SPECS[model_key]
    if spec.default_output_dir is not None:
        return Path(spec.default_output_dir)
    return Path(f"data/test/{spec.label}_{distance_weights}_test_metrics")


def default_rgb_output_dir(model_key: str) -> Path | None:
    spec = MODEL_SPECS[model_key]
    if spec.default_rgb_output_dir is None:
        return None
    return Path(spec.default_rgb_output_dir)


def resolve_prediction_format(args: argparse.Namespace, model_key: str) -> str:
    if args.prediction_format != "auto":
        return args.prediction_format
    return MODEL_SPECS[model_key].prediction_format


def should_run_torch_inference(args: argparse.Namespace, requested_model: str) -> bool:
    if args.skip_inference:
        return False
    if requested_model == "auto":
        return args.checkpoint is not None or args.checkpoint_dir is not None
    return MODEL_SPECS[requested_model].kind == "torch"


def import_label_map_evaluator():
    try:
        from scripts import evaluate_nnunet_predictions as evaluator
    except ModuleNotFoundError:
        import evaluate_nnunet_predictions as evaluator
    return evaluator


def evaluate_rgb_predictions(
    args: argparse.Namespace,
    prediction_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    pod_prediction_dir: Path,
    pod_mask_dir: Path,
) -> dict:
    import utils.Measure as Measure

    _, full_summary = Measure.evaluate_prediction_directory(
        pred_dir=prediction_dir,
        mask_dir=mask_dir,
        output_csv=output_dir / "full_image_metrics.csv",
        summary_json=output_dir / "full_image_summary.json",
        pred_prefix=args.pred_prefix,
    )

    summary = {"full_image": full_summary}
    if args.measure_pods:
        Measure.measure_prediction_directory(
            pred_path=prediction_dir,
            input_path=args.image_dir,
            pod_save_path=pod_prediction_dir,
            output_csv=output_dir / "pod_measurements.csv",
            plot=args.plot,
        )

        if pod_mask_dir is not None and pod_mask_dir.is_dir():
            _, pod_summary = Measure.evaluate_prediction_directory(
                pred_dir=pod_prediction_dir,
                mask_dir=pod_mask_dir,
                output_csv=output_dir / "pod_metrics.csv",
                summary_json=output_dir / "pod_summary.json",
                pred_prefix=args.pred_prefix,
            )
            summary["pod"] = pod_summary

    return summary


def evaluate_label_map_predictions(
    prediction_dir: Path,
    label_dir: Path,
    output_dir: Path,
    rgb_output_dir: Path | None,
) -> dict:
    evaluator = import_label_map_evaluator()
    rows = evaluator.evaluate(prediction_dir, label_dir, rgb_output_dir)
    summary = evaluator.write_outputs(rows, output_dir)
    return {"label_map": summary}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    requested_model = normalize_model_name(args.model)

    checkpoint_path = None
    checkpoint = None
    model_key = requested_model
    run_inference = should_run_torch_inference(args, requested_model)

    if run_inference:
        import torch

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint_path = resolve_checkpoint_path(args, requested_model)
        model, model_key, checkpoint = load_torch_model(checkpoint_path, requested_model, device)
    elif model_key == "auto":
        detected_model = detect_model_from_path(args.prediction_dir or args.checkpoint_dir or args.checkpoint)
        model_key = detected_model or "external"

    distance_weights = args.distance_weights
    if checkpoint is not None and checkpoint.get("distance_weights") is not None:
        distance_weights = str(checkpoint["distance_weights"])

    prediction_format = resolve_prediction_format(args, model_key)
    prediction_dir = args.prediction_dir or default_prediction_dir(model_key, distance_weights)
    output_dir = args.output_dir or default_output_dir(model_key, distance_weights)
    mask_dir = args.mask_dir or default_mask_dir(model_key, prediction_format)
    pod_prediction_dir = args.pod_prediction_dir or default_pod_prediction_dir(model_key, distance_weights)
    rgb_output_dir = args.rgb_output_dir
    if rgb_output_dir is None and prediction_format == "label-map":
        rgb_output_dir = default_rgb_output_dir(model_key)

    prediction_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_inference:
        import utils.SegmentImage as SegmentImage

        image_names = labeled_image_names(args.image_dir, mask_dir)
        SegmentImage.segment_image(
            model=model,
            image_path=path_with_sep(args.image_dir),
            mask_path=None,
            save_path=path_with_sep(prediction_dir),
            image_names=image_names,
            plot=args.plot,
            verbose=0,
            device=device,
        )

    combined_summary = {
        "model": model_key,
        "model_label": run_label(model_key),
        "distance_weights": distance_weights,
        "prediction_format": prediction_format,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "prediction_dir": str(prediction_dir),
        "mask_dir": str(mask_dir),
        "output_dir": str(output_dir),
    }

    if not args.skip_evaluation:
        if prediction_format == "label-map":
            combined_summary.update(
                evaluate_label_map_predictions(
                    prediction_dir=prediction_dir,
                    label_dir=mask_dir,
                    output_dir=output_dir,
                    rgb_output_dir=rgb_output_dir,
                )
            )
            write_json(output_dir / "run_summary.json", combined_summary)
        else:
            combined_summary.update(
                evaluate_rgb_predictions(
                    args=args,
                    prediction_dir=prediction_dir,
                    mask_dir=mask_dir,
                    output_dir=output_dir,
                    pod_prediction_dir=pod_prediction_dir,
                    pod_mask_dir=args.pod_mask_dir,
                )
            )
            write_json(output_dir / "summary.json", combined_summary)

    print(json.dumps(combined_summary, indent=2))


if __name__ == "__main__":
    main()
