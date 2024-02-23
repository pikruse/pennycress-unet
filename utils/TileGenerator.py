import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from scipy import ndimage

class TileGenerator(Dataset):
    
    def __init__(
        self, 
        images, 
        masks,
        tile_size, 
        split,
        pad,
    ):
        
        super().__init__()
        self.images = images
        self.masks = masks
        self.tile_size = tile_size
        self.split = split

        # extract image/row/col indices from mask pixels
        self.indices = []
        for i, m in enumerate(masks):
            m = m[pad:-pad, pad:-pad].sum(-1) > 0
            m = ndimage.binary_dilation(m, iterations=max(1, pad//2))
            pix_idx = np.argwhere(m == 1) + pad
            img_idx = i * np.ones(len(pix_idx), dtype=np.int32)
            self.indices.append(
                np.concatenate([img_idx[:, None], pix_idx], axis=1))
        self.indices = np.concatenate(self.indices, axis=0)
        
        # image augmentation
        self.apply_augmentation = A.Compose([
            A.Affine(
                rotate=[-180, 180], 
                scale=[0.8, 1.2],
                keep_ratio=True, 
                mode=1, # reflect
                p=1), # always rotate
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.5,
                contrast_limit=0.1,
                p=0.5),
            A.RandomGamma(p=0.5),
            # A.CLAHE(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=50, 
                val_shift_limit=50, 
                p=0.5),
            A.Blur(blur_limit=5, p=0.5),
        ], p=1)
        
    def __len__(self):
        return len(self.indices)
    
    # convert numpy array to torch tensor
    def to_torch(self, x):
        return torch.from_numpy(x).float()
    
    # convert torch tensor to numpy array
    def to_numpy(self, x):
        return x.detach().cpu().numpy()
    
    # get image and mask pair
    def __getitem__(self, index):

        # unpack indices and do some book keeping
        i, j, k = self.indices[index]
        w = int(self.tile_size / 2)
        
        # get image and mask
        width = int(1 + (2*w**2)**0.5)
        image = self.images[i][j-width:j+width, k-width:k+width].copy()
        mask = self.masks[i][j-width:j+width, k-width:k+width].copy()

        # optionally apply augmentation
        if self.split == 'train':
            image = (255*image).astype(np.uint8)
            mask = mask.astype(np.uint8)
            aug = self.apply_augmentation(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']
            image = image.astype(np.float32) / 255

        # resize image and mask
        r, c = image.shape[0] // 2, image.shape[1] // 2
        image = image[r-w:r+w, c-w:c+w] # [2*w, 2*w, 3]
        mask = mask[r-w:r+w, c-w:c+w] # [2*w, 2*w, C]
        
        # add background class to mask
        background = (mask.sum(-1) == 0)[:, :, None]
        mask = np.concatenate([background, mask], axis=-1)

        # convert to torch tensor
        image = self.to_torch(image.transpose(2, 0, 1))
        mask = self.to_torch(mask.transpose(2, 0, 1))

        return image, mask