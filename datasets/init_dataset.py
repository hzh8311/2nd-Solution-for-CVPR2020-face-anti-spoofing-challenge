import torch.utils.data as data
import torch
from PIL import Image
import pandas as pd
import numpy as np
import os

import torch.utils.data as data

def rgb_loader(path):
    return Image.open(path)

def rgb_prune_loader(path, pad=5):
    im = Image.open(path)
    im = np.asarray(im)
    im_ = np.sum(im, -1)

    row_sum = np.sum(im_, 0)
    col_sum = np.sum(im_, 1)

    idxs = np.where(row_sum != 0)[0]
    left, right = idxs[0], idxs[-1]
    idxs = np.where(col_sum != 0)[0]
    top, bottom = idxs[0], idxs[-1]

    left = max(0, left-pad)
    right = min(im.shape[1], right+pad)
    top = max(0, top-pad)
    bottom = min(im.shape[0], bottom+pad)

    im = im[top:bottom, left:right, :]
    im = Image.fromarray(im)
    return im


class ImageListDataset(data.Dataset):
    """
    Builds a dataset based on a list of images.
    data_root - image path prefix
    data_list - annotation list location
    """
    def __init__(self, opt,data_root, data_list, transform=None):
        self.data_root = data_root
        #self.df = pd.read_csv(data_list)
        self.df = pd.read_csv(data_list,sep=' ')
        # print(self.df)
        if 'label' not in self.df.columns:
            self.df['label'] = -1
        self.transform = transform

        if opt.net_type=='ResNet34DLAS_B':
            self.loader = rgb_prune_loader
        else:
            self.loader = rgb_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (rgb_img, ir_img, depth_img, target) 
        """
        dict_elem = self.__get_simple_item__(index)

        dict_elem['meta'] = {
            'idx': index,
            'max_idx': len(self.df),
            'get_item_func': self.__get_simple_item__
        }

        if self.transform is not None:
            dict_elem = self.transform(dict_elem)
            
        return dict_elem['rgb'], dict_elem['depth'], dict_elem['ir'], dict_elem['label'],  dict_elem['dir']

    def __get_simple_item__(self, index):
        rgb_path = self.data_root + self.df.rgb.iloc[index]
        # import pdb
        # pdb.set_trace()
        # print(self.df.rgb.iloc[index])
        # print(self.df.ir.iloc[index])
        ir_path = self.data_root + self.df.ir.iloc[index]
        depth_path = self.data_root + self.df.depth.iloc[index]
        target = self.df.label.iloc[index]
        
        rgb_img = self.loader(rgb_path)
        ir_img = self.loader(ir_path)
        depth_img = self.loader(depth_path)
        
        dict_elem = {
            'rgb': rgb_img,
            'ir': ir_img,
            'depth': depth_img,
            'label': target,
            'dir': rgb_path
        }
        return dict_elem

    def __len__(self):
        return len(self.df)