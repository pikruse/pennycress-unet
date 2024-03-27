import numpy as np

def iou(y_true, y_pred):

    """
    Intersection over Union (IoU) metric for semantic segmentation
    
    Parameters:
        y_true (np.array): boolean, H x W array of ground truth labels
        y_pred (np.array): boolean, H x W array of predicted labels
    
    Returns:
        IoU (float): Intersection over Union score
    """

    # flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # calculate intersection and union
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection

    # calculate IoU and return
    return intersection / union