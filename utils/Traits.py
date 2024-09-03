# import packages
import numpy as np
import cv2

from skimage import measure
from PIL import Image


### DEFINE TRAIT UTILITY FUNCTIONS ###


# get_color_features function
def get_color_features(img, mask):
    """
    Extracts color features from an image (r, g, b, h, s, v, l, a, B).

    Parameters:
        Image (np.array): An 3-channel image with range 0-255.
        Mask (np.array): A 3-channel mask with range 0-255.
    
    Returns:
        Features (tuple): A 3-object tuple, each containing a 9-dim feature vector with each object representing a color trait.
    """
    # get rgb, hsv, and lab color spaces
    rgb = img
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    r, g, b = rgb.astype(float).transpose(2, 0, 1)
    h, s, v = hsv.astype(float).transpose(2, 0, 1)
    l, a, B = lab.astype(float).transpose(2, 0, 1)

    # extract color features for segmentation classes
    # concat all feature maps into a tensor, index in with class label mask
    features = np.stack([r, g, b, h, s, v, l, a, B], axis=-1)

    # separate out the classes
    wing = test_mask[:,:,0] > 128
    env = test_mask[:,:,1] > 128
    seed = test_mask[:,:,2] > 128 

    # get features by class
    wing_features = features[wing] # output is (n_pixels, n_features)
    env_features = features[env]
    seed_features = features[seed]

    # aggregate features into phenotypes
    wing_phenotype = wing_features.mean(axis=0)
    env_phenotype = env_features.mean(axis=0)
    seed_phenotype = seed_features.mean(axis=0)

    return (wing_phenotype, env_phenotype, seed_phenotype)

# perimeter function
def perimeter(mask):

    """
    Calculates the perimeter of a multiclass mask.

    Parameters:
        Mask (np.array): A 3-channel mask with range 0-255.

    Returns: 
        Perimeter (tuple): A 3-object tuple, each containing the perimeter of the mask.
    """

    # separate out the classes
    wing = mask[:,:,0] > 128
    env = mask[:,:,1] > 128
    seed = mask[:,:,2] > 128

    # get perimeters by class
    wing_perimeter = measure.perimeter(wing + env + seed)
    env_perimeter = measure.perimeter(env + seed)
    seed_perimeter = measure.perimeter(seed)

    return (wing_perimeter, env_perimeter, seed_perimeter)

def to_total_ratio(mask, feature: str, type: str = "area"):

    """
    Calculates the ratio of a feature to the total area of a mask.

    Parameters:
        Mask (np.array): A 3-channel mask with range 0-255.
        Feature (str): The feature to calculate the ratio of.
        Type (str): The type of feature to calculate the ratio of. Must be one of 'area' or 'perimeter.'

    Returns:
        Ratio (float): The ratio of the feature to the total area of the mask.
    """

    assert feature in ["wing", "env", "seed"], "Feature must be one of 'wing', 'env', or 'seed'."
    assert type in ["area", "perimeter"], "Type must be one of 'area' or 'perimeter'."

    if type == "area":
        # separate out the classes
        wing = mask[:,:,0] > 128
        env = mask[:,:,1] > 128
        seed = mask[:,:,2] > 128

        # get areas by class
        wing_area, env_area, seed_area = wing.sum(), env.sum(), seed.sum()

        # get total area
        total_area = wing_area + env_area + seed_area

        # get ratio of feature to total area
        if feature == "wing":
            return wing_area / total_area
        elif feature == "env":
            return env_area / total_area
        elif feature == "seed":
            return seed_area / total_area
    
    elif type == "perimeter":
        # get perimeters by class
        wing_perimeter, env_perimeter, seed_perimeter = perimeter(mask)

        # get total perimeter
        total_perimeter = wing_perimeter + env_perimeter + seed_perimeter

        # get ratio of feature to total perimeter
        if feature == "wing":
            return wing_perimeter / total_perimeter
        elif feature == "env":
            return env_perimeter / total_perimeter
        elif feature == "seed":
            return seed_perimeter / total_perimeter

