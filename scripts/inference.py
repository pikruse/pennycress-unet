# Import packages 
import os, glob, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import cv2

from tqdm.auto import tqdm
from ipywidgets import FloatProgress
from scipy import ndimage
from PIL import Image
from importlib import reload
from DGXutils import GetFileNames

# append path
sys.path.append('../')

# custom
from utils.BuildUNet import UNet
from utils.TileGenerator import TileGenerator
from utils.Metrics import iou
from utils.Traits import area_calc
import utils.SegmentImage as SegmentImage
import utils.Measure as Measure

### IMPORT MODEL
# set up argparsing to load different models in different ways

### LOAD IMAGES
image_path = "../data/test/test_images/"
mask_path = "../data/test/test_masks_preproc/"
save_path = f"../data/test/{model_type}_test_predictions/"

# load image data
image_names = GetFileNames(mask_path, "png")
