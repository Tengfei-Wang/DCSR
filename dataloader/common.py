import random

import numpy as np
import skimage.color as sc

import torch

def get_patch(lr, hr, ref, patch_size=108, scale=2):

    L_h, L_w = lr.shape[:2]
    L_p = patch_size
    
    L_x = random.randrange(L_w//4, 3*L_w//4 - L_p + 1 - 15)
    L_y = random.randrange(L_h//4, 3*L_h//4 - L_p + 1 - 15)

    H_x, H_y = scale * L_x, scale * L_y
    H_p =  scale * L_p   

    patch_LR = lr[L_y:L_y + L_p, L_x:L_x + L_p, :]
    patch_HR = hr[H_y:H_y + H_p, H_x:H_x + H_p, :] 
    delta = random.randint(0,30)
    patch_ref = ref[(L_y -  L_h//4)*scale + delta :(L_y -  L_h//4)*scale + H_p + delta , (L_x -  L_w//4)*scale + delta:(L_x -  L_w//4)*scale + H_p + delta, :]
    if hr.shape == lr.shape:
        return patch_LR, patch_LR, patch_ref
    return patch_LR, patch_HR, patch_ref

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 4:
            img = img[:,:,:3]
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)       
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5
    k1 = np.random.randint(0, 4)
    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]        
        
        img = np.rot90(img, k1)
        
        return img

    return [_augment(a) for a in args]

