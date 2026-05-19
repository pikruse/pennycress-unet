#!/usr/bin/env python
"""Convert pennycress pod segmentation data to nnU-Net v2 format."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


LABELS = {
    "background": 0,
    "wing": 1,
    "envelope": 2,
    "seed": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create nnUNet_raw/DatasetXXX_PennycressPods from the repo's "
            "train/test image and RGB mask folders."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("nnu_net"))
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="PennycressPods")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing raw nnU-Net dataset directory before writing.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Only write imagesTr and labelsTr. By default test images/labels are also converted.",
    )
    return parser.parse_args()


def sanitize_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned or fallback


def dataset_folder_name(dataset_id: int, dataset_name: str) -> str:
    return f"Dataset{dataset_id:03d}_{sanitize_name(dataset_name, 'PennycressPods')}"


def collect_pairs(data_root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = data_root / split / f"{split}_images_by_pod"
    mask_dir = data_root / split / f"{split}_masks_by_pod"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")

    images = {p.name: p for p in image_dir.glob("*.png")}
    masks = {p.name: p for p in mask_dir.glob("*.png")}

    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        details = []
        if missing_masks:
            details.append(f"{len(missing_masks)} image(s) without masks")
        if missing_images:
            details.append(f"{len(missing_images)} mask(s) without images")
        raise ValueError(f"{split} pairing failed: {', '.join(details)}")

    return [(images[name], masks[name]) for name in sorted(images)]


def case_id(split: str, index: int, image_path: Path) -> str:
    clean_stem = sanitize_name(image_path.stem, f"{split}_{index:04d}")
    return f"{split}_{index:04d}_{clean_stem}"


def mask_to_labelmap(mask_path: Path) -> Image.Image:
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image.convert("RGB"))
    labelmap = np.zeros(mask.shape[:2], dtype=np.uint8)
    labelmap[mask[:, :, 0] > 127] = LABELS["wing"]
    labelmap[mask[:, :, 1] > 127] = LABELS["envelope"]
    labelmap[mask[:, :, 2] > 127] = LABELS["seed"]
    return Image.fromarray(labelmap)


def convert_pair(
    image_path: Path,
    mask_path: Path,
    output_image_path: Path,
    output_label_path: Path,
) -> None:
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        if image.size != mask.size:
            raise ValueError(
                f"Image/mask size mismatch for {image_path.name}: "
                f"image={image.size}, mask={mask.size}"
            )
        image.convert("RGB").save(output_image_path)

    mask_to_labelmap(mask_path).save(output_label_path)


def write_split(
    pairs: list[tuple[Path, Path]],
    split: str,
    images_dir: Path,
    labels_dir: Path,
    manifest_writer: csv.DictWriter,
) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for index, (image_path, mask_path) in enumerate(pairs):
        cid = case_id(split, index, image_path)
        output_image_path = images_dir / f"{cid}_0000.png"
        output_label_path = labels_dir / f"{cid}.png"
        convert_pair(image_path, mask_path, output_image_path, output_label_path)
        manifest_writer.writerow(
            {
                "split": split,
                "case_id": cid,
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "nnunet_image": str(output_image_path),
                "nnunet_label": str(output_label_path),
            }
        )


def write_dataset_json(dataset_dir: Path, num_training: int) -> None:
    dataset_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": LABELS,
        "numTraining": num_training,
        "file_ending": ".png",
    }
    with (dataset_dir / "dataset.json").open("w") as f:
        json.dump(dataset_json, f, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    raw_root = output_root / "nnUNet_raw"
    dataset_dir = raw_root / dataset_folder_name(args.dataset_id, args.dataset_name)

    if args.overwrite and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "nnUNet_preprocessed").mkdir(parents=True, exist_ok=True)
    (output_root / "nnUNet_results").mkdir(parents=True, exist_ok=True)

    train_pairs = collect_pairs(args.data_root, "train")
    test_pairs = [] if args.skip_test else collect_pairs(args.data_root, "test")

    manifest_path = dataset_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        fieldnames = [
            "split",
            "case_id",
            "source_image",
            "source_mask",
            "nnunet_image",
            "nnunet_label",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        write_split(
            train_pairs,
            "train",
            dataset_dir / "imagesTr",
            dataset_dir / "labelsTr",
            writer,
        )
        if test_pairs:
            write_split(
                test_pairs,
                "test",
                dataset_dir / "imagesTs",
                dataset_dir / "labelsTs",
                writer,
            )

    write_dataset_json(dataset_dir, len(train_pairs))

    print(f"Wrote {len(train_pairs)} training case(s) to {dataset_dir}")
    if test_pairs:
        print(f"Wrote {len(test_pairs)} test case(s) to {dataset_dir / 'imagesTs'}")
    print(f"Manifest: {manifest_path}")
    print()
    print("Set these before running nnU-Net:")
    print(f'export nnUNet_raw="{raw_root}"')
    print(f'export nnUNet_preprocessed="{output_root / "nnUNet_preprocessed"}"')
    print(f'export nnUNet_results="{output_root / "nnUNet_results"}"')


if __name__ == "__main__":
    main()
