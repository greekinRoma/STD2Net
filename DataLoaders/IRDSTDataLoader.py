import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from numpy import *
import numpy as np
import scipy.io as scio
from .seqsource import SeqSource
from wrapper import CacheDataset,cache_read_img
class IRDSTDataLoader(CacheDataset):
    def __init__(self, root,mode='train', fullSupervision=False,use_cache=True,cache_type="ram",num_frames=5):
        txtpath = root + f'{mode}.txt'
        txt = np.loadtxt(txtpath, dtype=bytes).astype(str)
        #读取train.txt文件
        self.seqs_arr = txt
        self.root = root
        img_size = (256,256)
        self.frame_num = num_frames
        self.img_size = img_size
        self.img_heigh = self.img_size[0]
        self.img_width = self.img_size[1]
        self.use_cache = use_cache
        self.cache_type = cache_type
        self.fullSupervision = fullSupervision
        self.seq_datasets = SeqSource(root=root,imgs_arr=self.seqs_arr,frame_num=self.frame_num,use_cache=self.use_cache,cache_type=self.cache_type)
        txts = self.seqs_arr
        self.num_imgs = len(self.seqs_arr)
        if self.fullSupervision:
            txts = [txt.replace('Mix', 'Mix_masks') for txt in txts]
        else:
            txts = [txt.replace('Mix', 'Mix_masks_centroid') for txt in txts]
        self.imgs_arr = txts
        self.img_datasets = SeqSource(root=root,imgs_arr=self.imgs_arr,use_cache=self.use_cache,cache_type=self.cache_type)
        super().__init__(
            input_dimension=self.img_size,
            num_imgs=self.num_imgs,
            cache=use_cache,
            cache_type=cache_type
        )
    @cache_read_img(use_cache=True)
    def read_img(self, index):
        # Mix preprocess
        MixData_Img = self.seq_datasets[index]
        MixData_out = MixData_Img.astype(np.float32)/255.0
        # Tgt preprocess
        TgtData_out = (self.img_datasets[index]/255.0).astype(np.float32)
        a,b,m_L,n_L = TgtData_out.shape
        if m_L == self.img_heigh and n_L == self.img_width:
            # Tgt preprocess
            return np.array([MixData_out, TgtData_out, m_L, n_L],dtype=object)
        else:
            # Tgt preprocess
            [n, t, m_M, n_M] = shape(MixData_out)
            TgtData_out_1 = np.zeros([n,t,self.img_heigh,self.img_width],dtype=np.float32)
            MixData_out_1 = np.zeros([n,t,self.img_heigh,self.img_width],dtype=np.float32)
            TgtData_out_1[0:a, 0:b, 0:m_L, 0:n_L] = TgtData_out
            MixData_out_1[0:n, 0:t, 0:m_M, 0:n_M] = MixData_out
        return np.array([MixData_out_1, TgtData_out_1, m_L, n_L],dtype=object)
    
    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
       img, mask, m_L, n_L = self.read_img(index)
       img = torch.from_numpy(img)
       mask = torch.from_numpy(mask)
       return img, mask, m_L, n_L
    def __len__(self):
        return len(self.imgs_arr)