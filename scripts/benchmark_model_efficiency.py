#!/usr/bin/env python
"""Benchmark model size, inference time, and peak GPU memory.

This script uses the same checkpoint loading path and tiled full-image inference
path as scripts/inference.py. The default model set is the checkpoint-backed
full-image test results under data/test/*_test_metrics/summary.json.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))


SUMMARY_COLUMNS = [
    "model_result",
    "model",
    "distance_weights",
    "checkpoint",
    "parameter_count",
    "trainable_parameter_count",
    "parameter_count_millions",
    "benchmark_device",
    "num_images",
    "mean_inference_seconds_per_image",
    "std_inference_seconds_per_image",
    "total_inference_seconds",
    "peak_gpu_memory_allocated_mb",
    "peak_gpu_memory_reserved_mb",
    "measurement_status",
    "notes",
]

PER_IMAGE_COLUMNS = [
    "model_result",
    "image_name",
    "inference_seconds",
    "peak_gpu_memory_allocated_mb",
    "peak_gpu_memory_reserved_mb",
]


def parse_args() -> argparse.Namespace:
    default_test_root = REPO_ROOT / "data" / "test"
    parser = argparse.ArgumentParser(
        description=(
            "Measure parameter count, full-image inference time, and peak GPU "
            "memory for checkpoint-backed pennycress model results."
        )
    )
    parser.add_argument("--test-root", type=Path, default=default_test_root)
    parser.add_argument("--image-dir", type=Path, default=default_test_root / "test_images")
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=default_test_root / "test_masks_preproc",
        help="Used only to select the five full-image test image names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_test_root / "model_efficiency_summary.csv",
    )
    parser.add_argument(
        "--per-image-output",
        type=Path,
        default=default_test_root / "model_efficiency_per_image.csv",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model_result names to benchmark, e.g. unet_none sam3_up.",
    )
    parser.add_argument("--image-name", action="append", default=[])
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Benchmark only the first N selected images.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--parameter-only",
        action="store_true",
        help="Load each model and count parameters without running image inference.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge new rows into existing output files by model_result.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show the underlying tiled inference progress bars.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort on the first model error instead of writing an error row.",
    )
    return parser.parse_args()


def repo_relative(path: Path | str | None) -> str:
    if path is None:
        return ""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def configure_model_caches() -> None:
    cache_root = REPO_ROOT / "nnu_net" / "cache" / "model_weights"
    hf_home = cache_root / "huggingface"
    os.environ.setdefault("MODEL_CACHE_ROOT", str(cache_root))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def infer_distance_weights(model_result: str) -> str:
    for suffix in ("_none", "_down", "_up"):
        if model_result.endswith(suffix):
            return suffix.removeprefix("_")
    return ""


def load_model_specs(test_root: Path, selected_models: set[str] | None) -> list[dict[str, str]]:
    from scripts.inference import detect_model_from_path

    specs = []
    for summary_path in sorted(test_root.glob("*_test_metrics/summary.json")):
        model_result = summary_path.parent.name.removesuffix("_test_metrics")
        if selected_models is not None and model_result not in selected_models:
            continue

        payload = json.loads(summary_path.read_text())
        checkpoint = payload.get("checkpoint")
        model = payload.get("model") or detect_model_from_path(Path(checkpoint or model_result))
        distance_weights = payload.get("distance_weights") or infer_distance_weights(model_result)
        if not checkpoint:
            continue

        specs.append(
            {
                "model_result": model_result,
                "model": model or "",
                "distance_weights": distance_weights or "",
                "checkpoint": checkpoint,
            }
        )
    return specs


def selected_image_names(mask_dir: Path, explicit_names: list[str], max_images: int | None) -> list[str]:
    if explicit_names:
        names = explicit_names
    else:
        names = sorted(path.name for path in mask_dir.glob("*.png"))
    if max_images is not None:
        names = names[:max_images]
    if not names:
        raise FileNotFoundError(f"No benchmark images selected from {mask_dir}")
    return names


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_rows_by_key(
    path: Path,
    fieldnames: list[str],
    new_rows: list[dict[str, object]],
    key: str,
) -> list[dict[str, object]]:
    if not path.exists():
        return new_rows
    with path.open(newline="") as f:
        existing = list(csv.DictReader(f))
    merged = {row[key]: row for row in existing}
    for row in new_rows:
        merged[str(row[key])] = row
    return [merged[key_value] for key_value in sorted(merged)]


def merge_per_image_rows(
    path: Path,
    fieldnames: list[str],
    new_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not path.exists():
        return new_rows
    with path.open(newline="") as f:
        existing = list(csv.DictReader(f))
    merged = {(row["model_result"], row["image_name"]): row for row in existing}
    for row in new_rows:
        merged[(str(row["model_result"]), str(row["image_name"]))] = row
    return [merged[key] for key in sorted(merged)]


def load_torch_model(spec: dict[str, str], device):
    from scripts.inference import load_torch_model, normalize_model_name

    model_key = normalize_model_name(spec["model"])
    checkpoint_path = REPO_ROOT / spec["checkpoint"]
    model, resolved_model_key, _ = load_torch_model(checkpoint_path, model_key, device)
    return model, resolved_model_key


def cuda_memory_mb(torch_module, device, kind: str) -> float | str:
    if device.type != "cuda":
        return ""
    if kind == "allocated":
        value = torch_module.cuda.max_memory_allocated(device)
    elif kind == "reserved":
        value = torch_module.cuda.max_memory_reserved(device)
    else:
        raise ValueError(kind)
    return value / (1024**2)


def fallback_parameter_count_from_logs(model: str) -> int | None:
    patterns = {
        "mask2former": r"Mask2Former.*parameters=([0-9,]+)",
        "segformer": r"SegFormer.*parameters=([0-9,]+)",
        "sam3": r"SAM3.*parameters=([0-9,]+)",
    }
    pattern = patterns.get(model)
    if pattern is None:
        return None

    for log_path in sorted((REPO_ROOT / "logs").glob("*.out")):
        match = re.search(pattern, log_path.read_text(errors="ignore"))
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def short_error(exc: Exception, max_chars: int = 240) -> str:
    message = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    if len(message) <= max_chars:
        return message
    return message[: max_chars - 3] + "..."


def benchmark_image(
    *,
    model,
    image_dir: Path,
    image_name: str,
    device,
    show_progress: bool,
) -> dict[str, object]:
    import torch
    import utils.SegmentImage as SegmentImage

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    with tempfile.TemporaryDirectory(prefix="pennycress_benchmark_") as tmpdir:
        save_path = Path(tmpdir)
        start = time.perf_counter()
        if show_progress:
            SegmentImage.segment_image(
                model=model,
                image_path=str(image_dir) + os.sep,
                mask_path=None,
                save_path=str(save_path) + os.sep,
                image_names=[image_name],
                plot=False,
                verbose=0,
                device=device,
            )
        else:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    SegmentImage.segment_image(
                        model=model,
                        image_path=str(image_dir) + os.sep,
                        mask_path=None,
                        save_path=str(save_path) + os.sep,
                        image_names=[image_name],
                        plot=False,
                        verbose=0,
                        device=device,
                    )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    return {
        "image_name": image_name,
        "inference_seconds": elapsed,
        "peak_gpu_memory_allocated_mb": cuda_memory_mb(torch, device, "allocated"),
        "peak_gpu_memory_reserved_mb": cuda_memory_mb(torch, device, "reserved"),
    }


def benchmark_spec(
    spec: dict[str, str],
    image_names: list[str],
    image_dir: Path,
    device,
    parameter_only: bool,
    show_progress: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    import numpy as np
    import torch

    model, resolved_model = load_torch_model(spec, device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    per_image_rows: list[dict[str, object]] = []
    if parameter_only:
        status = "parameter_only"
        notes = "Timing and peak GPU memory not measured; rerun without --parameter-only."
    else:
        status = "measured"
        notes = ""
        for image_name in image_names:
            row = benchmark_image(
                model=model,
                image_dir=image_dir,
                image_name=image_name,
                device=device,
                show_progress=show_progress,
            )
            row["model_result"] = spec["model_result"]
            per_image_rows.append(row)

    timings = [float(row["inference_seconds"]) for row in per_image_rows]
    allocated_peaks = [
        float(row["peak_gpu_memory_allocated_mb"])
        for row in per_image_rows
        if row["peak_gpu_memory_allocated_mb"] != ""
    ]
    reserved_peaks = [
        float(row["peak_gpu_memory_reserved_mb"])
        for row in per_image_rows
        if row["peak_gpu_memory_reserved_mb"] != ""
    ]

    summary = {
        "model_result": spec["model_result"],
        "model": resolved_model,
        "distance_weights": spec["distance_weights"],
        "checkpoint": repo_relative(spec["checkpoint"]),
        "parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "parameter_count_millions": total_params / 1_000_000,
        "benchmark_device": str(device),
        "num_images": len(per_image_rows),
        "mean_inference_seconds_per_image": float(np.mean(timings)) if timings else "",
        "std_inference_seconds_per_image": float(np.std(timings, ddof=0)) if timings else "",
        "total_inference_seconds": float(np.sum(timings)) if timings else "",
        "peak_gpu_memory_allocated_mb": max(allocated_peaks) if allocated_peaks else "",
        "peak_gpu_memory_reserved_mb": max(reserved_peaks) if reserved_peaks else "",
        "measurement_status": status,
        "notes": notes,
    }

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, per_image_rows


def error_summary(spec: dict[str, str], device, exc: Exception) -> dict[str, object]:
    fallback_params = fallback_parameter_count_from_logs(spec["model"])
    if fallback_params is None:
        status = "error"
        notes = short_error(exc)
        trainable_params: int | str = ""
        parameter_count_millions: float | str = ""
    else:
        status = "parameter_count_from_log_error"
        notes = (
            "Model reload failed in this environment; parameter_count was recovered "
            f"from logs. Original error: {short_error(exc)}"
        )
        # Mask2Former/SegFormer training used optimizer over model.parameters().
        trainable_params = fallback_params
        parameter_count_millions = fallback_params / 1_000_000

    return {
        "model_result": spec["model_result"],
        "model": spec["model"],
        "distance_weights": spec["distance_weights"],
        "checkpoint": repo_relative(spec["checkpoint"]),
        "parameter_count": fallback_params or "",
        "trainable_parameter_count": trainable_params,
        "parameter_count_millions": parameter_count_millions,
        "benchmark_device": str(device),
        "num_images": 0,
        "mean_inference_seconds_per_image": "",
        "std_inference_seconds_per_image": "",
        "total_inference_seconds": "",
        "peak_gpu_memory_allocated_mb": "",
        "peak_gpu_memory_reserved_mb": "",
        "measurement_status": status,
        "notes": notes,
    }


def main() -> None:
    configure_model_caches()
    args = parse_args()

    import torch

    selected_models = set(args.models) if args.models else None
    specs = load_model_specs(args.test_root, selected_models)
    image_names = selected_image_names(args.mask_dir, args.image_name, args.max_images)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    summary_rows = []
    per_image_rows = []
    for spec in specs:
        try:
            summary, image_rows = benchmark_spec(
                spec=spec,
                image_names=image_names,
                image_dir=args.image_dir,
                device=device,
                parameter_only=args.parameter_only,
                show_progress=args.show_progress,
            )
            summary_rows.append(summary)
            per_image_rows.extend(image_rows)
            print(f"{spec['model_result']}: {summary['measurement_status']}")
        except Exception as exc:
            if args.strict:
                raise
            summary_rows.append(error_summary(spec, device, exc))
            print(f"{spec['model_result']}: error: {short_error(exc)}", file=sys.stderr)

    summary_rows.sort(key=lambda row: str(row["model_result"]))
    per_image_rows.sort(key=lambda row: (str(row["model_result"]), str(row["image_name"])))

    if args.append:
        summary_rows = merge_rows_by_key(args.output, SUMMARY_COLUMNS, summary_rows, "model_result")
        per_image_rows = merge_per_image_rows(args.per_image_output, PER_IMAGE_COLUMNS, per_image_rows)

    write_csv(args.output, SUMMARY_COLUMNS, summary_rows)
    if per_image_rows:
        write_csv(args.per_image_output, PER_IMAGE_COLUMNS, per_image_rows)

    print(f"Wrote summary to {args.output}")
    if per_image_rows:
        print(f"Wrote per-image timings to {args.per_image_output}")


if __name__ == "__main__":
    main()
