#import dataset class
from torch.utils.data import Dataset
import albumentations as A
import torch

import numpy as np

# define tilegenerator class, inheriting from dataset class
class TileGenerator(Dataset):
    
    def __init__(self,
                images,
                masks,
                tile_size,
                split, #split is a string specifying 'train' or 'val',
                n_pad
                ):
        #init attrs/methods from dataset class w/ super()
        super().__init__()
        self.images = images
        self.masks = masks
        self.tile_size = tile_size
        self.split = split
        self.width = tile_size//2

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
        
        #convert to torch tensor
        tile = torch.from_numpy(tile.transpose(2, 0, 1)) #convert to torch tensor and transpose to channels first
        mask = torch.from_numpy(mask.transpose(2, 0, 1)) #convert to torch tensor and transpose to channels first

        # convert to float
        tile, mask = tile.float(), mask.float()

        return tile, mask