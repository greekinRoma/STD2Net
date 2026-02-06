import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from numpy import *
import numpy as np
import scipy.io as scio
from .seqsource import SeqSource
from .imgsource import ImgSource
class SeqSetLoader(Dataset):
    def __init__(self, root,mode='train', fullSupervision=False,cache=True,cache_type="ram",):
        txtpath = root + f'{mode}.txt'
        txt = np.loadtxt(txtpath, dtype=bytes).astype(str)
        #读取train.txt文件
        self.seqs_arr = txt
        self.root = root
        
        self.frame_num = 5
        self.img_size = (150,200)
        self.img_heigh = self.img_size[0]
        self.img_width = self.img_size[1]
        self.cache = cache
        self.cache_type = cache_type
        self.fullSupervision = fullSupervision
        self.seq_datasets = SeqSource(root=root,imgs_arr=self.seqs_arr,frame_num=self.frame_num,cache=self.cache,cache_type=self.cache_type)
        txts = self.seqs_arr
        if self.fullSupervision:
            txts = [txt.replace('Mix', 'Mix_masks') for txt in txts]
        else:
            txts = [txt.replace('Mix', 'Mix_masks_centroid') for txt in txts]
        self.imgs_arr = txts
        self.img_datasets = SeqSource(root=root,imgs_arr=self.imgs_arr,cache=self.cache,cache_type=self.cache_type)
        self.train_mean = 105.4025
        self.train_std = 26.6452
    def read_img(self, index):
        # Mix preprocess
        MixData_Img = (self.seq_datasets[index] - self.train_mean)/self.train_std
        MixData_out = torch.from_numpy(MixData_Img)
        TgtData_out = torch.from_numpy(self.img_datasets[index]/255.0).float()
        a,b,m_L,n_L = TgtData_out.shape
        if m_L == self.img_heigh and n_L == self.img_width:
            # Tgt preprocess
            return MixData_out, TgtData_out, m_L, n_L
        else:
            # Tgt preprocess
            [n, t, m_M, n_M] = shape(MixData_out)
            TgtData_out_1 = torch.zeros([n,t,self.img_heigh,self.img_width]).float()
            MixData_out_1 = torch.zeros([n,t,self.img_heigh,self.img_width]).float()
            TgtData_out_1[0:a, 0:b, 0:m_L, 0:n_L] = TgtData_out
            MixData_out_1[0:n, 0:t, 0:m_M, 0:n_M] = MixData_out
        return MixData_out_1, TgtData_out_1, m_L, n_L
    def __getitem__(self, index):
       return self.read_img(index)
    def __len__(self):
        return len(self.imgs_arr)

class TestSetLoader(Dataset):
    def __init__(self, root, fullSupervision=True,cache=True,cache_type="ram",):
        txtpath = root + 'test.txt'
        txt = np.loadtxt(txtpath, dtype=bytes).astype(str)
        #读取train.txt文件
        self.seqs_arr = txt
        self.root = root
        self.frame_num = 5
        self.img_size = (512,512)
        self.img_heigh = self.img_size[0]
        self.img_width = self.img_size[1]
        self.cache = cache
        self.cache_type = cache_type
        self.fullSupervision = fullSupervision
        self.seq_datasets = SeqSource(root=root,imgs_arr=self.seqs_arr,frame_num=self.frame_num,cache=self.cache,cache_type=self.cache_type)
        txts = [txt.replace('.mat', '.png') for txt in self.seqs_arr]
        if self.fullSupervision:
            txts = [txt.replace('Mix', 'masks') for txt in txts]
        else:
            txts = [txt.replace('Mix', 'masks_centroid') for txt in txts]
        self.imgs_arr = txts
        self.maks_datasets = ImgSource(root=root,imgs_arr=self.imgs_arr,cache=self.cache,cache_type=self.cache_type)
        self.train_mean = 105.4025
        self.train_std = 26.6452
    def read_img(self, index):
        LabelData_Img = self.maks_datasets[index]/255.0
        # print(LabelData_Img.shape)
        # print(np.sum(LabelData_Img))
        [m_L, n_L] = np.shape(LabelData_Img)
        MixData_Img = (self.seq_datasets[index] - self.train_mean)/self.train_std
        MixData_out = torch.from_numpy(MixData_Img)
        if m_L == self.img_heigh and n_L == self.img_width:
            # Tgt preprocess
            LabelData = torch.from_numpy(LabelData_Img)
            TgtData_out = torch.unsqueeze(LabelData, 0)
            return MixData_out, TgtData_out, m_L, n_L

        else:
            # Tgt preprocess
            [n, t, m_M, n_M] = shape(MixData_out)
            LabelData_Img_1 = np.zeros([self.img_heigh,self.img_width])
            LabelData_Img_1[0:m_L, 0:n_L] = LabelData_Img
            LabelData = torch.from_numpy(LabelData_Img_1)
            TgtData_out = torch.unsqueeze(LabelData, 0)
            MixData_out_1 = torch.zeros([n,t,self.img_heigh,self.img_width])
            MixData_out_1[0:n, 0:t, 0:m_M, 0:n_M] = MixData_out
        return MixData_out_1, TgtData_out, m_L, n_L
    def __getitem__(self, index):
       return self.read_img(index)
    def __len__(self):
        return len(self.imgs_arr)