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
    
    for image_name in tqdm(image_names):

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
        mask = (np.stack([bkgd, wing, env, seed], axis=2).astype(np.uint8) * 255)[:, :, 1:]

        if verbose:
            print("Shape: ", mask.shape)
            print("Values:", np.unique(mask))
        
        mask = Image.fromarray(mask)

        # save mask
        mask.save(save_path + image_name)

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

def split_image(image_names,
                image_path,
                image_save_path,
                mask_path=None,
                mask_save_path=None,
                plot = False):
    
    """
    Function to split images and masks by individual pods.

    Parameters:
        image_names (list): list of image names
        image_path (str): path to the images
        mask_path (str): path to masks
        image_save_path (str): path to save images
        mask_save_path (str): path to save masks
        plot (bool, default=True): whether to plot images or not
    
    Returns:
        None
    """


    # create empty list for bounding boxes
    bounding_boxes = []

    # loop through each image
    for image_name in image_names:

        print("Processing image: ", image_name)

        # load image and mask
        image = Image.open(image_path + image_name).convert('RGB')

        if mask_path is not None:
            mask = Image.open(mask_path + image_name).convert('RGB')

        # convert to numpy array and normalize
        image = np.array(image) / 255.0

        if mask_path is not None:
            mask_rgb = np.array(mask) / 255.0 #normalized rgb mask for saving
            mask = np.array(mask).sum(axis=2) > 128 # convert to boolean mask

        # pad image and and mask
        image = np.pad(image, ((100, 100), (100, 100), (0, 0)), mode='edge')

        if mask_path is not None:
            mask = np.pad(mask, ((100, 100), (100, 100)), mode='constant')
            mask_rgb = np.pad(mask_rgb, ((100, 100), (100, 100), (0, 0)), mode='constant')

        if plot:
            # plot image and mask for sanity
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(image)
            ax[1].imshow(mask, cmap='gray')
            plt.show()

        # label each component in mask and create bounding boxes
        labels = ndimage.label(mask)[0]
        bboxes = ndimage.find_objects(labels)

        # add padding to bounding boxes
        x_pad, y_pad = 100, 100
        for i in range(len(bboxes)):
            x, y = bboxes[i]
            bboxes[i] = slice(x.start-x_pad, x.stop+x_pad), slice(y.start-y_pad, y.stop+y_pad)

        if plot:
            if mask_path is not None:
            # plot image and mask with bounding boxes
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(image)
                axs[1].imshow(mask)
                for bbox in bboxes:
                    y, x = bbox
                    axs[0].plot([x.start, x.start, x.stop, x.stop, x.start], [y.start, y.stop, y.stop, y.start, y.start], '--', color='r')
                    axs[1].plot([x.start, x.start, x.stop, x.stop, x.start], [y.start, y.stop, y.stop, y.start, y.start], '--', color='r')
                plt.tight_layout()
                plt.show()
            else:
                fig = plt.figure(figsize=(5, 5))
                fig.imshow(image)
                for bbox in bboxes:
                    y, x = bbox
                    fig.plot([x.start, x.start, x.stop, x.stop, x.start], [y.start, y.stop, y.stop, y.start, y.start], '--', color='r')
                plt.tight_layout()
                plt.show()
                

        # save split images
        for i, bbox in enumerate(tqdm(bboxes)):
            y, x = bbox
            split_image = image[y, x, :]
            if mask_path is not None:
                split_mask = mask_rgb[y, x, :]

            # convert to PIL image
            split_image = Image.fromarray((split_image * 255).astype(np.uint8))
            if mask_path is not None:
                split_mask = Image.fromarray((split_mask * 255).astype(np.uint8))

            # save img, msk
            split_image.save(image_save_path + image_name[:-4] + "_" + str(i) + ".png")
            if mask_path is not None:
                split_mask.save(mask_save_path + image_name[:-4] + "_" + str(i) + ".png")

    print("Image Splitting Complete!")

    return 

def remove_artifacts(img):
    """
    Function to remove small artifacts from segmentations

    Parameters:
        img (np.array): input image with artifacts, range [0,1]
    Returns:
        img (np.array): image with artifacts removed, range [0,1]
    """

    # create binary mask
    mask = img.sum(axis=2) != 3

    # remove small artifacts
    size = 5000
    labels = ndimage.label(mask)[0]
    sizes = np.bincount(labels.reshape(-1))
    for j in range(1, len(sizes)):
        if sizes[j] < size:
            mask[labels == j] = False

    img[mask == 0 ] = 1

    return img