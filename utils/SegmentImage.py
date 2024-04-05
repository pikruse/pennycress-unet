# Import packages 
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

# custom imports
from utils.IOU import iou
from utils.GetLowestGPU import GetLowestGPU

device = torch.device(GetLowestGPU(verbose=0))

def segment_image(model,
                  image_path,
                  mask_path,
                  save_path,
                  image_names,
                  plot = True,
                  verbose = True,
                  device=device):       
    
    """
    Function to create predicted segmentations for images in a given directory.

    Parameters:
        image_path (str): path to directory containing images
        mask_path (str): path to directory containing masks
        save_path (str): path to save predicted segmentations
        image_names (list): list of image names to process
        plot (bool): whether to plot the images
        verbose (bool): whether to print IoU scores

    Returns:
        None
    """

    # create list to store predicted segmentations so that we can count seeds later
    pred_images = []

    # loop over images
    for image_name in image_names:
        print(f"Processing {image_name}")

        # ----------------
        ## PREPROCESSING
        # ----------------

        # load in image and convert to a numpy array
        image = Image.open(image_path + image_name)
        image = np.array(image) / 255.0

        mask = np.array(Image.open(mask_path + image_name))[:,:,:3] / 255.0

        # fix weird mask behavior - all px. values are 0 or 1
        mask[mask > 0] = 1

        # pad first 2 dimensions, (H, W) by 32 px
        image = np.pad(image,
                    pad_width= ((32, 32), (32, 32), (0, 0)),
                    mode='edge')

        # ensure each image is divisible by 128 by adding extra padding

        rpad = 128 - image.shape[0] % 128
        cpad = 128 - image.shape[1] % 128

        # do we really need this? or can we optimize?
        image = np.pad(image,
                    pad_width=((0, rpad), (0, cpad), (0, 0)),
                    mode='edge')

        # initialize global prb. map with same spatial (H, W) dimensions as image and channel for each class
        global_prb_map = np.zeros((image.shape[0], image.shape[1], 4))

        ## BOOKKEEPING / PREDICTION

        # Double for loop over rows and columns with step size of 32 used for tile generation
        rows = range(0, image.shape[0]-(3*32), 32)
        cols = range(0, image.shape[1]-(3*32), 32)
        with tqdm(total=len(rows)*len(cols)) as pbar:
            for row in range(0, image.shape[0]-(3*32), 32): #don't index completely to the edge - otherwise, we will be indexing into a space that doesn't exist in the image
                for col in range(0, image.shape[1]-(3*32), 32):

                    # select image tile
                    tile = image[row:row+128, col:col+128]

                    # make prediction and store the *center context* in global prb. map
                    unet_input = torch.tensor(tile, dtype=torch.float32).to(device).permute(2, 0, 1).unsqueeze(0)
                    prediction = model(unet_input) 

                    # softmax output
                    prediction = torch.nn.functional.softmax(prediction, dim=1)

                    # shuffle to be rows-cols-channels order and only take the center 64x64 square
                    prediction = prediction[0].permute(1, 2, 0)[32:-32, 32:-32, :]

                    # add predicted probabilities to prb map at index
                    global_prb_map[row+32:row+96, col+32:col+96, :] += prediction.detach().cpu().numpy()
                    pbar.update(1)

        # now that we have full prb. map, remove the padding 
        global_prb_map = global_prb_map[32:-(32+rpad), 32:-(32+cpad), :]
        image = image[32:-(32+rpad), 32:-(32+cpad), :]

        # and normalize prbs. s.t they sum to one
        global_prb_map = global_prb_map / global_prb_map.sum(axis=-1, keepdims=True)

        # convert predicted probabilities to predicted classes and retain four channels
        argmaxes = np.argmax(global_prb_map, axis=2)
        preds = np.zeros(global_prb_map.shape)
        for i in range(4):
            preds[:, :, i] = (argmaxes == i)

        # remove bg channel
        pred_image = preds[:, :, 1:]

        # ---------------
        # IOU
        # ---------------

        wing_gt = mask[:, :, 0] == 1
        wing_pred = pred_image[:, :, 0] == 1
        wing_iou = iou(wing_gt, wing_pred)

        env_gt = mask[:, :, 1:].sum(-1) == 1 
        env_pred = pred_image[:, :, 1:].sum(-1) == 1  
        env_iou = iou(env_gt, env_pred)

        seed_gt = mask[:, :, 2] == 1
        seed_pred = pred_image[:, :, 2] == 1
        seed_iou = iou(seed_gt, seed_pred)

        if verbose:
            print(f"Jaccard Distance (IoU) for wing: {wing_iou}\n")
            print(f"Jaccard Distance (IoU) for envelope: {env_iou}\n")
            print(f"Jaccard Distance (IoU) for seeds: {seed_iou}\n")

        # ----------------
        ## ARTFACT REMOVAL
        # ----------------

        # set bg to white
        pred_image[pred_image.sum(axis=2) == 0] = 1
        mask[mask.sum(axis=2) == 0] = 1

        # create binary mask
        pred_mask = pred_image.sum(axis=2) != 3

        # remove small artifacts
        size = 500
        labels = ndimage.label(pred_mask)[0]
        sizes = np.bincount(labels.reshape(-1))
        for j in range(1, len(sizes)):
            if sizes[j] < size:
                pred_mask[labels == j] = False

        pred_mask = cv2.cvtColor(pred_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        pred_image[pred_mask.sum(2) == 0] = 1

        # ----------------
        ## PLOTTING
        # ----------------
        
        if plot:
            # plot the image and prediction together
            fig, ax = plt.subplots(1, 6, figsize=(40, 20))
            ax[0].imshow(image)
            ax[0].set_title("Original Image")

            # set mask bg to white and overlay on image
            ax[1].imshow(mask)
            ax[1].set_title("Ground Truth Mask")

            ax[2].imshow(pred_image)
            ax[2].set_title("Predicted Mask")

            ax[3].imshow(wing_pred, cmap='gray')
            ax[3].set_title("Wing Prediction")

            ax[4].imshow(env_pred, cmap='gray')
            ax[4].set_title("Envelope Prediction")

            ax[5].imshow(seed_pred, cmap='gray')
            ax[5].set_title("Seed Prediction")
            plt.show

        # append to pred images list for counting
        pred_images.append(pred_image)

        # save the predicted mask
        pred_image = Image.fromarray((pred_image * 255).astype(np.uint8))
        pred_image.save(save_path + "pred_" + image_name)
    
    return None