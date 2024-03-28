import numpy as np

def area_calc(mask, dpi = 600, scale = 'in'):

    """
    Converts mask pixel area to in^2

    Parameters:
        mask (np array): one-channel, boolean input mask image (H x W) 
        dpi (int): resolution of image
        scale (str): scale of image (in or cm)
    
    Returns:
        area (float): area of mask in in/cm
    """

    # calculate pixel area
    pixel_area = np.sum(mask == 1)

    # convert pixel area to in

    if scale == 'in':
        area = pixel_area / dpi 
        return area

    if scale == 'cm':
        area = pixel_area / (dpi * 2.54)
        return area
    
    
