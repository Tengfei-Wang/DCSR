import os
import glob
import random
import pickle

import dataloader.common as common

import numpy as np
import imageio
import torch
import torch.utils.data as data
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
class Train_Dataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.do_eval = True

        self.dir_hr = os.path.join(args.dir_train_HR)
        self.dir_lr = os.path.join(args.dir_train_LR)        
        self.dir_ref = os.path.join(args.dir_train_ref)  

        self.hr_filelist = sorted(os.listdir(self.dir_hr))
        self.lr_filelist = sorted(os.listdir(self.dir_lr))
        self.ref_filelist = sorted(os.listdir(self.dir_ref))

        self.num_img = len(self.hr_filelist)
        self.num_patch = args.num_patch
       

    def __getitem__(self, idx):
        lr, hr, ref, filename = self._load_file(idx)
        lr, hr, ref = common.get_patch(lr, hr, ref, patch_size=self.args.patch_size, scale=self.args.scale )
        if not self.args.no_augment: 
            lr, hr, ref = common.augment(lr, hr, ref)

        pair = [lr, hr, ref]

        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], pair_t[2],filename

    def __len__(self):
        return self.num_img * self.num_patch

    def _load_file(self, idx):
        idx = idx % self.num_img
        hr_filename = self.hr_filelist[idx]
        lr_filename = self.lr_filelist[idx]
        ref_filename = self.ref_filelist[idx]
        filename = hr_filename
        hr = imageio.imread(os.path.join(self.dir_hr, hr_filename))
        lr = imageio.imread(os.path.join(self.dir_lr, lr_filename))
        ref = imageio.imread(os.path.join(self.dir_ref, ref_filename))
        return lr, hr, ref, filename


class Test_Dataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.name = args.data_test
        self.do_eval = True

        self.dir_lr = args.dir_test_LR     
        self.dir_ref = args.dir_test_ref
        self.dir_hr = args.dir_test_HR

        self.lr_filelist = sorted(os.listdir(self.dir_lr))
        self.hr_filelist = sorted(os.listdir(self.dir_hr))
        self.ref_filelist = sorted(os.listdir(self.dir_ref))

        self.num_img = len(self.lr_filelist)

       

    def __getitem__(self, idx):
        lr, hr, ref, filename = self._load_file(idx)

        pair = [lr, hr, ref]

        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], pair_t[2], filename

    def __len__(self):
        return self.num_img 


    def _load_file(self, idx):
        lr_filename = self.lr_filelist[idx]
        hr_filename = self.hr_filelist[idx]
        ref_filename = self.ref_filelist[idx]
        filename = lr_filename

        lr = imageio.imread(os.path.join(self.dir_lr, lr_filename))
        hr = imageio.imread(os.path.join(self.dir_hr, hr_filename))
        ref = imageio.imread(os.path.join(self.dir_ref, ref_filename))
        return lr,  hr, ref, filename    

class myData:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            trainset = Train_Dataset(args)
            self.loader_train = dataloader.DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )    

        self.loader_test = []
        testset = Test_Dataset(args)

        self.loader_test.append(
            dataloader.DataLoader(
                testset,
                batch_size=1,
                shuffle=False,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )
        )       



