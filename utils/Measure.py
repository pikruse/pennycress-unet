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
from utils.AreaCalc import area_calc
import utils.SegmentImage as SegmentImage

device = torch.device(GetLowestGPU(verbose=0))

def measure_pods(pred_path,
                 pod_save_path,
                 measurement_save_path,
                 image_names,
                 verbose = True,
                 plot = True):
    
    """
    Function to measure the area of leaves, count seeds, and plot a segmented image.

    Parameters:
        pred_path (str): path to directory containing predicted segmentations
        pod_save_path (str): path to save pod images
        measurement_save_path (str): path to save measurements
        image_names (list): list of image names to process
        verbose (bool): whether to print IoU scores
        plot (bool): whether to plot the images
    
    Returns:
        None
    """

    # create list to store seed counts and area
    measurements = []

    def measure_func(pred_image_name):
        
        pod_measurements = []
        
        ## PREPROCESSING

        # for a single image:
        pred_image = Image.open(pred_path + pred_image_name)
        pred_image = np.array(pred_image) / 255 


        # revert white background to black
        pred_image[pred_image.sum(axis=2) == 3] = 0

        # pad image so that we can draw bounding box around it
        pred_image = np.pad(pred_image, ((100, 100), (100, 100), (0, 0)), mode='constant')

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

                # calculate area
                wing_area = area_calc(split_image[:, :, 0])
                env_area = area_calc(split_image[:, :, 1:2])
                seed_area = area_calc(split_image[:, :, 2])

                if verbose:
                        print(f"wing area: {wing_area:.2f} cm", "|", f"env area: {env_area:.2f} cm", "|", f"seed area: {seed_area:.2f} cm")

                # extract only the blue channel (seeds) from the image
                image = split_image[:, :, 2].astype(np.int64)

                # change image type and make 3-channel mask
                image = image.astype(np.uint8) * 255

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
                save_name = pred_image_name[:-4] + "_" + str(i) + ".png"
                split_image = Image.fromarray((split_image * 255).astype(np.uint8))
                split_image.save(pod_save_path + save_name)

                # save seed count
                pod_measurements.append((save_name, seed_count, wing_area, env_area, seed_area))

        return pod_measurements
    
    with mp.Pool(mp.cpu_count()) as pool:
        result = tqdm(pool.imap(measure_func, image_names),
                total = len(image_names))
        for r in result:
               measurements.extend(r)
           
    # save seed counts to csv
    print(len(measurements))
    measurements = pd.DataFrame(measurements, columns=["image_name", "seed_count", "wing area", "env area", "seed area"])


    if verbose:
            avg_seed_count = measurements["seed_count"].mean()
            avg_wing = measurements["wing area"].mean()
            avg_env = measurements["env area"].mean()
            avg_seed = measurements["seed area"].mean()

            print(f"Avg. Seed Count: {avg_seed_count:.2f} seeds",
                f"Avg. Wing Area: {avg_wing:.2f} cm",
                f"Avg. Envelope Area: {avg_env:.2f} cm",
                f"Avg. Seed Area: {avg_seed:.2f} cm")
            
    measurements.to_csv(measurement_save_path + "measurements.csv", index=False)
