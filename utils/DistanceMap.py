# import packages
import os, sys, glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from skimage.io import imshow
from scipy import ndimage
from skimage.measure import label
from scipy.ndimage import distance_transform_edt

# define Distance Map function

def distance_map_bw(image, wb = 10, bwidth = 5):
    """
    function to create distance map from grayscale image

    Parameters:
        image (numpy array): input 1-channel boolean mask (MUST BE T-F)
        wb (int): weight for border pixels
        bwidth (int): border width parameter
    
    Returns:
        distance weight map (numpy array of same size as input image)
    """

    # multiply image by inverse of binary erosion
    image = image * ~ndimage.binary_erosion(image, iterations=1)

    # calculate distances away from border
    distance = distance_transform_edt(~image)

    # set distance cutoff at bwidth max and invert
    weights = bwidth - distance.clip(max = bwidth)

    # normalize weights and multiply by wb param.
    w = (weights / weights.max()) * wb

    if wb < 1:
        # invert the weights

        # min/max normalize to 0-1
        w = (w - w.min()) / (w.max() - w.min())

        # invert the weights
        w = 1 - w

        # multiply by (1 - wb) and then add wb to get min wb and max 1 
        w = (w * (1 - wb)) + (wb)  
    else:
        w = w.clip(min = 1)

    return w

def distance_map_rgb(image, wb = 10, bwidth = 5, wc = None):

    """
    (code modified from https://gist.github.com/rok/5f4314ed3c294521456c6afda36a3a50)

    Function to create distance map from grayscale image

    Parameters:
        image (numpy array): input 3-channel RGB image (MUST BE 0-255!!!!)
        wc (dictionary): dictionary containing class weights
        wb (int): weight for border pixels
        bwidth (int): border width parameter
    
    Returns:
        distance weight map (numpy array of same size as input image)
    """

    # convert image to PIL and grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

    # label objects in grayscale image
    labels = label(gray)

    # get list of label ids
    label_ids = sorted(np.unique(labels))
    
    # initialize blank distance matrix
    distances = np.zeros((image.shape[0], image.shape[1], len(label_ids)))

    # calculate distance between each labeled object and other pixels
    for i, label_id in enumerate(label_ids):
        distances[:,:,i] = distance_transform_edt(labels != label_id)
    
    # sort distances
    distances = np.sort(distances, axis=2)

    # get two smallest distances (closest objects to each other)
    d1 = distances[:,:,0]
    d2 = np.zeros_like(d1)
    if distances.shape[2] > 1:
        d2 = distances[:,:,1]

    # calculate weights (border param. * exp(-1/2 * (d1 + d2) / sigma) * no_labels
    w = wb * np.exp(-1/2*((d1 + d2) / bwidth)**2)

    if wb < 1:
        # invert the weights

        # min/max normalize to 0-1
        w = (w - w.min()) / (w.max() - w.min())

        # invert the weights
        w = 1 - w

        # multiply by (1 - wb) and then add wb to get min wb and max 1 
        w = (w * (1 - wb)) + (wb)  
    else:
        w = w.clip(min = 1)

    # if weight classes are provided
    if wc:
        class_weights = np.ones_like(gray)
        # loop through weights and add to map
        for k, v in wc.items():
            if k == 'wing':
                class_weights[image[:, :, 0] == 255] = v
            elif k == 'env':
                class_weights[image[:, :, 1:].sum(-1) == 255] = v
            elif k == 'seed':
                class_weights[image[:, :, 2] == 255] = v
        w = w * class_weights
    
    return w
