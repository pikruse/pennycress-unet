import sys
import numpy as np
from importlib import reload

sys.path.append("../")

# custom imports
import utils.DistanceMap as DistanceMap

reload(DistanceMap)
def gradient_transform(mask, bwidth):

    """
    Function to transform mask edges to reflect uncertainty in segmentation
    
    Parameters:
        mask (numpy array): input mask
        bwidth (int): border width parameter

    Returns:
        mask (numpy array): input mask
    """

    # extract seed, env, wing
    seed = mask[:,:,2] > 0.5
    env = mask[:,:,1] > 0.5
    wing = mask[:,:,0]

    # extract seed/env border
    weights = DistanceMap.distance_map_bw(seed, wb = 0.5, bwidth = 5)


    weighted_seed = weights * seed
    weighted_seed[env] = 1 - weights[env]

    weighted_env = weights * env
    weighted_env[seed] = 1 - weights[seed]

    smooth_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    smooth_mask[:,:,0] = wing
    smooth_mask[:,:,1] = weighted_env
    smooth_mask[:,:,2] = weighted_seed

    return smooth_mask

