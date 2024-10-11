# Import packages 
import os, glob, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import cv2
import math
import skimage
import multiprocess as mp

from tqdm.auto import tqdm
from ipywidgets import FloatProgress
from scipy import ndimage
from skimage import measure
from PIL import Image
from importlib import reload
from IPython.display import clear_output
from functools import partial

# append path
sys.path.append('../')

# custom
from utils.BuildUNet import UNet
from utils.GetLowestGPU import GetLowestGPU
from utils.TileGenerator import TileGenerator
from utils.Metrics import iou
import utils.Traits as Traits
import utils.SegmentImage as SegmentImage



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
        pred_image = np.array(pred_image) / 255

        # remove "pred_" from image name
        input_name = image_name[5:]
        input_image = Image.open(input_path + input_name)
        input_image = np.array(input_image) / 255

        # revert white background to black
        pred_image[pred_image.sum(axis=2) == 3] = 0

        # pad image so that we can draw bounding box around it
        pred_image = np.pad(pred_image, ((100, 100), (100, 100), (0, 0)), mode='constant')
        input_image = np.pad(input_image, ((100, 100), (100, 100), (0, 0)), mode='constant')

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
                split_image = pred_image[y, x, :].astype(np.uint8) * 255
                split_input = input_image[y, x, :].astype(np.uint8) * 255
                
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

                if verbose:
                        print(f"wing area: {wing_area:.2f} cm", "|", f"env area: {env_area:.2f} cm", "|", f"seed area: {seed_area:.2f} cm")

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
                split_image = Image.fromarray((split_image * 255).astype(np.uint8))
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