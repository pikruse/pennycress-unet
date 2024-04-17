# import necessary packages
import os, glob, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import cv2

from tqdm.auto import tqdm
from ipywidgets import FloatProgress
from scipy import ndimage
from PIL import Image

sys.path.append('../')

def mask_preprocessing(image_path,
                       image_names,
                       save_path,
                       verbose=False,
                       plot=False):
    
    """
    Function to preprocess masks for the segmentation task.

    Parameters:
        image_path (str): path to the images
        image_names (list): list of image names
        save_path (str): path to save the masks
    
    Returns:
        None
    """
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for image_name in image_names:

        if verbose:
            print("preprocessing: ", image_name)

        # load image
        image = np.asarray(Image.open(image_path + image_name)) / 255

        # break up mask
        wing = image[:, :, 0] > 0.5
        env  = image[:, :, 1] > 0.5
        seed = image[:, :, 2] > 0.5

        # fill holes
        wing = ndimage.binary_fill_holes((wing+env+seed) > 0.5)
        env  = ndimage.binary_fill_holes((env+seed) > 0.5)
        seed = ndimage.binary_fill_holes(seed)

        # delete env from wing, seed from env
        wing[env] = 0
        env[seed] = 0

        # create background
        bkgd = np.ones_like(wing)
        bkgd[wing] = 0
        bkgd[env]  = 0
        bkgd[seed] = 0

        # merge
        mask = np.stack([bkgd, wing, env, seed], axis=2).astype(np.uint8) * 255
        mask = Image.fromarray(mask)

        # save mask
        mask.save(save_path + image_name)

        if verbose:
            print("Shape: ", mask.shape)
            print("Values:", np.unique(mask))

        if plot:
            fig, axs = plt.subplots(1, 5, figsize=(20, 5))
            axs[0].imshow(bkgd)
            axs[0].set_title("Background")

            axs[1].imshow(wing)
            axs[1].set_title("Wing")

            axs[2].imshow(env)
            axs[2].set_title("Env")

            axs[3].imshow(seed)
            axs[3].set_title("Seed")

            axs[4].imshow(mask)
            axs[4].set_title("Mask")

            plt.show()

    return None