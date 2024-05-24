
def gradient_transform(mask, bwidth):

    """
    Function to transform mask edges to reflect uncertainty in segmentation
    
    Parameters:
        mask (numpy array): input mask
        bwidth (int): border width parameter

    Returns:
        mask (numpy array): input mask
    """

    # extract seed/env border
    weights = DistanceMap.distance_map_bw(seed, wb = 0.5, bwidth = 5)


    weighted_seed = weights * seed
    weighted_seed[env] = 1 - weights[env]

    weighted_env = weights * env
    weighted_env[seed] = 1 - weights[seed]
