from wrapper import CacheDataset,cache_read_img
import os
import numpy as np
import scipy.io as scio
import torch
from numpy import *
class SeqSource(CacheDataset):
    def __init__(
        self,
        root,
        imgs_arr,
        frame_num=5,
        cache=False,
        cache_type="ram",
    ):
        self.frame_num = frame_num
        self.root = root
        self.imgs_arr = imgs_arr
        self.cache = cache
        self.cache_type = cache_type
        self.train_mean = 105.4025
        self.train_std = 26.6452
        self.num_imgs = len(self.imgs_arr)
        super().__init__(
            input_dimension=(512,512),
            num_imgs=self.num_imgs,
            cache=cache,
            cache_type=cache_type
        )
    def __len__(self):
        return self.num_imgs
    @cache_read_img(use_cache=True)
    def read_img(self, index):
        file_path = os.path.join(self.root,self.imgs_arr[index])
        MixData_mat = scio.loadmat(file_path)
        MixData_Img = MixData_mat.get('Mix')
        MixData_Img = MixData_Img.astype(np.float32)
        MixData = MixData_Img[-self.frame_num:,:,:]
        MixData_out = np.expand_dims(MixData, 0)
        return MixData_out
    def __len__(self):
        return len(self.imgs_arr)
    def _input_dim(self):
        return self.img_size
    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        return self.read_img(index)

if __name__ == '__main__':
    root = '/home/greek/files/Video_structure/dataset/NUDT-MIRSDT'
    txtpath = os.path.join(root , 'train.txt')
    txts = np.loadtxt(txtpath, dtype=bytes).astype(str)
    img_source = SeqSource(root=root,imgs_arr=txts,img_size=(512,512),cache=True,cache_type="ram")
    