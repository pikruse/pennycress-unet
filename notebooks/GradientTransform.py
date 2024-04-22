
def gradient_transform(mask, bwidth):

    """
    Function to transform mask edges to reflect uncertainty in segmentation
    
    Parameters:
        mask (numpy array): input mask
        bwidth (int): border width parameter

    Returns:
        mask (numpy array): input mask
    """

