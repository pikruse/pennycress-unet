# import packages
import os, sys, glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from skimage.io import imshow
from skimage.measure import label
from scipy.ndimage import distance_transform_edt

# define Distance Map function

def distance_map(image, wc = None, wb = 10, bwidth = 5):

    """
    (code modified from https://gist.github.com/rok/5f4314ed3c294521456c6afda36a3a50)

    Function to create distance map from grayscale image

    Parameters:
        image (numpy array): input 3-channel RGB image
        wc (dictionary): dictionary containing class weights
        wb (int): weight for border pixels
        bwidth (int): border width parameter
    
    Returns:
        distance weight map (numpy array of same size as input image)
    """

    # convert image to PIL and grayscale
    image = Image.fromarray(image)
    gray = np.array(image.convert('L'))

    # convert back to array
    image = np.array(image)

    # label objects in grayscale image
    labels = label(gray)

    # get list of label ids
    label_ids = sorted(np.unique(labels))

    # # if no labels, return distance weights of zeros
    # if len(label_ids) == 1:
    #     return np.zeros_like(image)


    # # filter out wing/pod px
    # if border_type == 'seed':
    #     label_ids = label_ids[2:]

    # elif border_type == "pod":
    #     label_ids = label_ids[1:]
    
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
    w += 1

    # if weight classes are provided
    if wc:
        class_weights = np.ones_like(gray)

        # loop through weights and add to map
        for k, v in wc.items():
            if k == 'wing':
                class_weights[image[:, :, 0] == 1] = v
            if k == 'env':
                class_weights[image[:, :, 1] == 1] = v
            if k == 'seed':
                class_weights[image[:, :, 2] == 1] = v
        w = w * class_weights
    
    return w
