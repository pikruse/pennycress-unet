# imports
import albumentations as A
import torch
import numpy as np
import cv2

from torch.utils.data import Dataset
from skimage.io import imshow
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
from PIL import Image

# import custom weight function
import utils.DistanceMap as DistanceMap



# define tilegenerator class, inheriting from dataset class
class TileGenerator(Dataset):

    """
    TileGenerator class for creating tiles from images and masks

    Parameters:
        images (np array): input 3-channel RGB image
        masks (np array): input mask image (RGB)
        tile_size (int): size of tile (H and W)
        split (str): dataset mode (train or val)
        n_pad (int): padding size
        distance_weights (bool): whether to use distance weighting in loss
    
    Returns:
        tile (torch tensor): input tile (C x H x W)
        mask (torch tensor): target mask (C x H x W)
        (optional) distance weight map (torch tensor): distance weight map (C x H x W) appended to last mask dimension
    """
    
    def __init__(self,
                images,
                masks,
                tile_size,
                split, #split is a string specifying 'train' or 'val',
                n_pad,
                distance_weights = False
                ):
        #init attrs/methods from dataset class w/ super()
        super().__init__()
        self.images = images
        self.masks = masks
        self.tile_size = tile_size
        self.split = split
        self.width = tile_size//2
        self.distance_weights = distance_weights

        # extract image/row/col indices from mask pixels
        self.indices = [] #list pixel indices we want to extract tiles from
        for i, m in enumerate(masks): #enumerate to get mask index and mask values
            row_col_ixs = np.argwhere(m[n_pad:-n_pad, n_pad:-n_pad] >= 0) + n_pad #return array containing indices w/ all values inside padding
            img_ixs = i * np.ones(len(row_col_ixs), dtype=np.int32) #return image number repeated for however many indices we found
            self.indices.append(np.concatenate([img_ixs[:, None], row_col_ixs], axis=1)) # create array of image number, row, col indices
        self.indices = np.concatenate(self.indices, axis=0) # concatenate all image index arrays into one

        # make an augmenataion pipeline
        self.augment = A.Compose([
            A.Affine(rotate=[-180, 180],
                     mode=0,
                     p=1),
            A.Blur(blur_limit=7, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5)
            ], p=1)
        
    def __len__(self): # length is the total number of tiles
        return len(self.indices) # number of tiles is equal to number of indices
    
    def __getitem__(self, index):

        i, r, c, _ = self.indices[index]
        w = int((2*self.width**2)**0.5)+1 #use pythagorean theorem to grab a slightly larger tile
        tile = self.images[i][r-w:r+w, c-w:c+w]
        mask = self.masks[i][r-w:r+w, c-w:c+w]

        # augment image and mask
        if self.split == 'train':
            tile = (255*tile).astype(np.uint8)
            mask = mask.astype(np.uint8)
            augmented = self.augment(image=tile, mask=mask)
            tile = (augmented['image'] / 255.0).astype(np.float32)
            mask = augmented['mask'].astype(np.float32)
        
        #zoom into final size of width x width
        cent = tile.shape[0]//2
        tile = tile[cent-self.width:cent+self.width, cent-self.width:cent+self.width]
        mask = mask[cent-self.width:cent+self.width, cent-self.width:cent+self.width]

        # distance weighting 
        if self.distance_weights:
            weights_input = mask.astype(np.uint8)
            weights = DistanceMap.distance_map(weights_input, wc = {'wing': 1, 'env': 1, 'seed': 2}, wb = 10, bwidth = 5)
        
        #convert to torch tensor
        tile = torch.from_numpy(tile.transpose(2, 0, 1)) #convert to torch tensor and transpose to channels first
        mask = torch.from_numpy(mask.transpose(2, 0, 1)) #convert to torch tensor and transpose to channels first
        
        if self.distance_weights:
            weights = torch.from_numpy(weights).unsqueeze(0) #convert to torch tensor and add channel dimension

        # convert to float
        if self.distance_weights:
            tile, mask, weights = tile.float(), mask.float(), weights.float()
            targets = torch.cat([mask, weights], dim=0)
            
        else:
            tile, mask = tile.float(), mask.float()
            
        if self.distance_weights:
            return tile, targets
        
        else:
            return tile, mask