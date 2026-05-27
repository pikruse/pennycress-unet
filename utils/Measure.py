# Import packages 
import os, glob, sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import cv2
import math
import skimage
try:
        import multiprocess as mp
except ModuleNotFoundError:
        mp = None

from tqdm.auto import tqdm
try:
        from ipywidgets import FloatProgress
except ModuleNotFoundError:
        FloatProgress = None
from scipy import ndimage
from skimage import measure
from PIL import Image
from importlib import reload
try:
        from IPython.display import clear_output
except ModuleNotFoundError:
        def clear_output(*args, **kwargs):
                return None
from functools import partial
from pathlib import Path

# append path
sys.path.append('../')

# custom
from utils.BuildUNet import UNet
from utils.GetLowestGPU import GetLowestGPU
from utils.TileGenerator import TileGenerator
from utils.Metrics import iou
import utils.Traits as Traits
import utils.SegmentImage as SegmentImage


reload(Traits)

MEASUREMENT_COLUMNS = [
        'image_name',
        'seed_count',
        'wing_area',
        'env_area',
        'seed_area',
        'wing_perimeter',
        'env_perimeter',
        'seed_perimeter',
        'wing_to_total_area',
        'env_to_total_area',
        'seed_to_total_area',
        'wing_to_total_perimeter',
        'env_to_total_perimeter',
        'seed_to_total_perimeter',
        'env_to_seed_area',
        'wing_to_seed_area',
        'env_to_seed_perimeter',
        'wing_to_seed_perimeter',
        'wing_to_env_area',
        'seed_to_env_area',
        'wing_to_env_perimeter',
        'seed_to_env_perimeter',
        'seed_to_wing_area',
        'env_to_wing_area',
        'seed_to_wing_perimeter',
        'env_to_wing_perimeter',
        'wing_r',
        'wing_g',
        'wing_b',
        'wing_h',
        'wing_s',
        'wing_v',
        'wing_l',
        'wing_a',
        'wing_B',
        'env_r',
        'env_g',
        'env_b',
        'env_h',
        'env_s',
        'env_v',
        'env_l',
        'env_a',
        'env_B',
        'seed_r',
        'seed_g',
        'seed_b',
        'seed_h',
        'seed_s',
        'seed_v',
        'seed_l',
        'seed_a',
        'seed_B',
]


def _dir_with_sep(path):
        path = str(path)
        return path if path.endswith(os.sep) else path + os.sep


def load_rgb_mask_channels(path):
        mask = np.array(Image.open(path))
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


def binary_iou(y_true, y_pred, empty_score=1.0):
        y_true = np.asarray(y_true).astype(bool)
        y_pred = np.asarray(y_pred).astype(bool)
        if y_true.shape != y_pred.shape:
                raise ValueError("Shape mismatch: y_true and y_pred must have the same shape.")

        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        if union == 0:
                return empty_score
        return float(intersection / union)


def segmentation_iou_metrics(mask_path, pred_path, empty_score=1.0):
        gt = load_rgb_mask_channels(mask_path)
        pred = load_rgb_mask_channels(pred_path)

        rows = {}
        per_class_ious = []
        for class_name in ("wing", "envelope", "seed"):
                score = binary_iou(gt[class_name], pred[class_name], empty_score=empty_score)
                rows[f"{class_name}_iou"] = score
                per_class_ious.append(score)

        gt_envelope_with_seed = np.logical_or(gt["envelope"], gt["seed"])
        pred_envelope_with_seed = np.logical_or(pred["envelope"], pred["seed"])
        gt_foreground = np.logical_or.reduce((gt["wing"], gt["envelope"], gt["seed"]))
        pred_foreground = np.logical_or.reduce((pred["wing"], pred["envelope"], pred["seed"]))

        rows["envelope_with_seed_iou"] = binary_iou(
                gt_envelope_with_seed,
                pred_envelope_with_seed,
                empty_score=empty_score,
        )
        rows["foreground_iou"] = binary_iou(
                gt_foreground,
                pred_foreground,
                empty_score=empty_score,
        )
        rows["miou"] = float(np.mean(per_class_ious))
        return rows


def prediction_path_for_mask(pred_dir, mask_path, pred_prefix="pred_"):
        pred_dir = Path(pred_dir)
        mask_path = Path(mask_path)
        candidates = [
                pred_dir / f"{pred_prefix}{mask_path.name}",
                pred_dir / mask_path.name,
        ]
        for candidate in candidates:
                if candidate.exists():
                        return candidate
        raise FileNotFoundError(
                f"No prediction found for {mask_path.name}. Tried: "
                + ", ".join(str(candidate) for candidate in candidates)
        )


def evaluate_prediction_directory(pred_dir,
                                  mask_dir,
                                  output_csv=None,
                                  summary_json=None,
                                  pred_prefix="pred_"):
        pred_dir = Path(pred_dir)
        mask_dir = Path(mask_dir)
        mask_paths = sorted(mask_dir.glob("*.png"))
        if not mask_paths:
                raise FileNotFoundError(f"No PNG masks found in {mask_dir}")

        rows = []
        for mask_path in mask_paths:
                pred_path = prediction_path_for_mask(pred_dir, mask_path, pred_prefix=pred_prefix)
                row = {"image_name": mask_path.name}
                row.update(segmentation_iou_metrics(mask_path, pred_path))
                rows.append(row)

        metrics = [key for key in rows[0] if key != "image_name"]
        summary = {"num_images": len(rows)}
        for metric in metrics:
                values = np.array([row[metric] for row in rows], dtype=np.float64)
                summary[f"mean_{metric}"] = float(values.mean())
                summary[f"std_{metric}"] = float(values.std(ddof=0))

        if output_csv is not None:
                output_csv = Path(output_csv)
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(output_csv, index=False)

        if summary_json is not None:
                summary_json = Path(summary_json)
                summary_json.parent.mkdir(parents=True, exist_ok=True)
                with summary_json.open("w") as f:
                        json.dump(summary, f, indent=2)
                        f.write("\n")

        return rows, summary


def measure_prediction_directory(pred_path,
                                 input_path,
                                 pod_save_path,
                                 output_csv=None,
                                 verbose=False,
                                 plot=False):
        pred_path = Path(pred_path)
        input_path = Path(input_path)
        pod_save_path = Path(pod_save_path)
        pod_save_path.mkdir(parents=True, exist_ok=True)

        measurements = []
        pred_names = sorted(path.name for path in pred_path.glob("*.png"))
        for image_name in tqdm(pred_names, desc="Measure pods"):
                measurements.extend(
                        measure_pods(
                                image_name=image_name,
                                pred_path=_dir_with_sep(pred_path),
                                input_path=_dir_with_sep(input_path),
                                pod_save_path=_dir_with_sep(pod_save_path),
                                verbose=verbose,
                                plot=plot,
                        )
                )

        measurements = pd.DataFrame(measurements, columns=MEASUREMENT_COLUMNS)
        if output_csv is not None:
                output_csv = Path(output_csv)
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                measurements.to_csv(output_csv, index=False)
        return measurements


def measure_pods(image_name,
                 pred_path,
                 input_path,
                 pod_save_path,
                 verbose = False,
                 plot = False):
        
        """
        Function to measure the area of leaves, count seeds, and plot a segmented image.

        Parameters:
                image_name (str): name of image to measure
                pred_path (str): path to directory containing predicted segmentations
                input_path (str): path to directory containing input images
                pod_save_path (str): path to save pod images
                verbose (bool): whether to print IoU scores
                plot (bool): whether to plot the images

        Returns:
                None
        """

        pod_measurements = []

        ## PREPROCESSING

        # for a single image:
        pred_image = Image.open(pred_path + image_name)
        pred_image = np.array(pred_image).astype(np.uint8) / 255

        # remove "pred_" from image name
        input_name = image_name[5:]
        input_image = Image.open(input_path + input_name)
        input_image = np.array(input_image).astype(np.uint8) / 255

        # revert white background to black
        pred_image[pred_image.sum(axis=2) == 3] = 0

        # pad image so that we can draw bounding box around it
        pred_image = np.pad(pred_image, ((100, 100), (100, 100), (0, 0)), mode='constant')
        input_image = np.pad(input_image, ((100, 100), (100, 100), (0, 0)), mode='edge')

        # extract bool mask for object detection
        bool_mask = np.array(pred_image).sum(axis=2) > .5 # convert to boolean mask

        # label each object in the image and draw bounding box around it
        labels = ndimage.label(bool_mask)[0]
        bboxes = ndimage.find_objects(labels)

        # add padding to bounding boxes
        x_pad, y_pad = 100, 100
        for i in range(len(bboxes)):
                x, y = bboxes[i]
                bboxes[i] = slice(x.start-x_pad, x.stop+x_pad), slice(y.start-y_pad, y.stop+y_pad)

        ## WATERSHEDDING / AREA CALC

        # loop through split images
        for i, bbox in enumerate(bboxes):
                y, x = bbox
                split_image = pred_image[y, x, :]
                split_input = input_image[y, x, :]

                split_image = (split_image * 255.0).astype(np.uint8)
                split_input = (split_input * 255.0).astype(np.uint8)
                
                if split_image.shape != split_input.shape:
                        print(f"WARNING: {image_name}... Shapes don't match!")

                # calculate area
                wing_area = Traits.area_calc(split_image[:, :, 0])
                env_area = Traits.area_calc(split_image[:, :, 1:2])
                seed_area = Traits.area_calc(split_image[:, :, 2])

                # get perimeters
                wing_p, env_p, seed_p = Traits.perimeter(split_image)

                # -to-total area ratios
                wing_to_total_area = Traits.to_total_ratio(split_image, feature="wing")
                env_to_total_area = Traits.to_total_ratio(split_image, feature="env")
                seed_to_total_area = Traits.to_total_ratio(split_image, feature="seed")

                # -to-total perimeter ratios
                wing_to_total_perimeter = Traits.to_total_ratio(split_image, feature="wing", type="perimeter")
                env_to_total_perimeter = Traits.to_total_ratio(split_image, feature="env", type="perimeter")
                seed_to_total_perimeter = Traits.to_total_ratio(split_image, feature="seed", type="perimeter")

                # -to-seed ratios
                env_to_seed_area = Traits.between_ratio(split_image, feature1="env", feature2="seed", type="area")
                wing_to_seed_area = Traits.between_ratio(split_image, feature1="wing", feature2="seed", type="area")
                env_to_seed_perimeter = Traits.between_ratio(split_image, feature1="env", feature2="seed", type="perimeter")
                wing_to_seed_perimeter = Traits.between_ratio(split_image, feature1="wing", feature2="seed", type="perimeter")

                # -to-env ratios
                seed_to_env_area = Traits.between_ratio(split_image, feature1="seed", feature2="env", type="area")
                wing_to_env_area = Traits.between_ratio(split_image, feature1="wing", feature2="env", type="area")
                seed_to_env_perimeter = Traits.between_ratio(split_image, feature1="seed", feature2="env", type="perimeter")
                wing_to_env_perimeter = Traits.between_ratio(split_image, feature1="wing", feature2="env", type="perimeter")

                # -to-wing ratios
                seed_to_wing_area = Traits.between_ratio(split_image, feature1="seed", feature2="wing", type="area")
                env_to_wing_area = Traits.between_ratio(split_image, feature1="env", feature2="wing", type="area")
                seed_to_wing_perimeter = Traits.between_ratio(split_image, feature1="seed", feature2="wing", type="perimeter")
                env_to_wing_perimeter = Traits.between_ratio(split_image, feature1="env", feature2="wing", type="perimeter")

                # color
                wing_color, env_color, seed_color = Traits.get_color_features(split_input, split_image)

                # extract only the blue channel (seeds) from the image
                image = split_image[:, :, 2].astype(np.int64)

                # change image type and make 3-channel mask
                image = image.astype(np.uint8)

                # define kernel for operations
                kernel = np.ones((3,3),np.uint8)

                # # erode image to define splits
                # image = cv2.erode(image, kernel, iterations=2)

                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                # noise removal
                opening = cv2.morphologyEx(image,
                                        cv2.MORPH_OPEN,kernel,
                                        iterations = 2)

                # sure background area
                sure_bg = cv2.dilate(opening,
                                kernel,
                                iterations=3)

                # Finding sure foreground area
                dist_transform = cv2.distanceTransform(opening,
                                                cv2.DIST_L2,
                                                5)
                ret, sure_fg = cv2.threshold(dist_transform,
                                        0.6*dist_transform.max(),
                                        255,
                                        0)

                # Finding unknown region
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(sure_bg,sure_fg)

                # Marker labelling
                ret, markers = cv2.connectedComponents(sure_fg)

                # Add one to all labels so that sure background is not 0, but 1
                markers = markers+1

                # Now, mark the region of unknown with zero
                markers[unknown==255] = 0

                #plot markers
                markers_to_plot = markers

                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                markers = cv2.watershed(rgb, markers)
                rgb[markers == -1] = [255,0,0]

                # count seeds
                seed_count = len(np.unique(markers)) - 2

                if verbose:
                        print("seed count:", seed_count)

                if plot:
                        ## PLOTTING

                        # plot fg, bg, unknown, distances, markers, and labeled output
                        fix, ax = plt.subplots(1, 3, figsize=(10, 5))

                        ax[0].imshow(image, cmap="gray")
                        ax[0].set_title("Input Image")

                        ax[1].imshow(dist_transform, cmap="gray")
                        ax[1].set_title("Distances")

                        ax[2].imshow(markers_to_plot)
                        ax[2].set_title("Markers")


                        for axis in ax:
                                axis.set_axis_off()
                        plt.tight_layout
                        plt.show()

                ## BOOKKEEPING

                # save split predicted image
                save_name = image_name[:-4] + "_" + str(i) + ".png"
                split_image = Image.fromarray(split_image.astype(np.uint8))
                split_image.save(pod_save_path + save_name)

                # save seed count
                pod_measurements.append((save_name,
                                                # seed count 
                                                seed_count, 

                                                # areas
                                                wing_area, 
                                                env_area, 
                                                seed_area,

                                                # perimeters
                                                wing_p,
                                                env_p,
                                                seed_p,

                                                # ...-to-total_area ratios
                                                wing_to_total_area,
                                                env_to_total_area,
                                                seed_to_total_area,

                                                # ...-to-total_perimeter ratios
                                                wing_to_total_perimeter,
                                                env_to_total_perimeter,
                                                seed_to_total_perimeter,

                                                # ...-to-seed ratios
                                                env_to_seed_area,
                                                wing_to_seed_area,
                                                env_to_seed_perimeter,
                                                wing_to_seed_perimeter,

                                                # ...-to-env ratios
                                                wing_to_env_area,
                                                seed_to_env_area,
                                                wing_to_env_perimeter,
                                                seed_to_env_perimeter,

                                                # ...-to-wing ratios
                                                seed_to_wing_area,
                                                env_to_wing_area,
                                                seed_to_wing_perimeter,
                                                env_to_wing_perimeter,

                                                # wing color
                                                wing_color[0],
                                                wing_color[1],
                                                wing_color[2],
                                                wing_color[3],
                                                wing_color[4],
                                                wing_color[5],
                                                wing_color[6],
                                                wing_color[7],
                                                wing_color[8],

                                                # env color
                                                env_color[0],
                                                env_color[1],
                                                env_color[2],
                                                env_color[3],
                                                env_color[4],
                                                env_color[5],
                                                env_color[6],
                                                env_color[7],
                                                env_color[8],

                                                # seed color
                                                seed_color[0],
                                                seed_color[1],
                                                seed_color[2],
                                                seed_color[3],
                                                seed_color[4],
                                                seed_color[5],
                                                seed_color[6],
                                                seed_color[7],
                                                seed_color[8]))
        return pod_measurements
