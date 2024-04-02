import numpy as np

def area_calc(mask, dpi = 600, scale = 'cm'):

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

    # convert pixel area to in^2
    dpi_2 = dpi ** 2
    area_in_2 = pixel_area / dpi_2

    # convert in^2 to cm^2
    area_cm_2 = area_in_2 * 6.4516

    return area_cm_2


    
    
