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

def dice(y_true, y_pred, empty_score=1.0):

    """
    From: https://gist.github.com/brunodoamaral/e130b4e97aa4ebc468225b7ce39b3137

    Computes the Dice coefficient, a measure of set similarity.

    Parameters
    ----------
    y_true : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    y_pred : array-like, bool
        Any other array of identical size. If not boolean, will be converted.

    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    y_true = np.asarray(y_true).astype(np.bool)
    y_pred = np.asarray(y_pred).astype(np.bool)

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    y_sum = y_true.sum() + y_pred.sum()
    if y_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(y_true, y_pred)

    return 2. * intersection.sum() / y_sum

def sensitivity(tp, fn):

    """
    Sensitivity metric for binary classification
    
    Parameters:
        tp (int): number of true positives
        fn (int): number of false negatives
    
    Returns:
        sensitivity (float): Sensitivity score
    """

    sensitivity = tp / (tp + fn)

    return sensitivity

def specificity(tn, fp):

    """
    Specificity metric for binary classification
    
    Parameters:
        tn (int): number of true negatives
        fp (int): number of false positives
    
    Returns:
        specificity (float): Specificity score
    """

    

    return specificity

def precision(y_true, y_pred):

    return precision