import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import nn
import math
from ..AttentionModule.nonlocal_module import _NonLocalBlockND
from .contrast_and_atrous import AttnContrastLayer
class ExpansionContrastModule(nn.Module):
    def __init__(self,in_channels,out_channels,shifts,kernel_size,use_nonlocal=False):
        super().__init__()
        #The hyper parameters settting
        self.convs_list=nn.ModuleList()
        delta1=np.array([[[-1, 0, 0], [0, 1, 0], [0, 0, 0]],
                         [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
                         [[0, 0, -1], [0, 1, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 1, -1], [0, 0, 0]]])
        delta1=delta1.reshape(4,1,3,3)
        delta2=delta1[:,:,::-1,::-1].copy()
        delta=np.concatenate([delta1,delta2],axis=0)
        w1,w2,w3,w4,w5,w6,w7,w8=np.array_split(delta,8)
        self.in_channels = max(in_channels,1)
        self.shifts =shifts
        self.num_heads = len(self.shifts)
        self.value_convs = nn.ModuleList()
        self.query_convs = nn.ModuleList()
        self.key_convs = nn.ModuleList()
        self.kernel_size = kernel_size
        #After Extraction, we analyze the outcome of the extraction.
        self.num_layer= 8
        self.num_shift= len(shifts)
        self.value_key_convs=nn.ModuleList()
        self.kernel1 = torch.Tensor(w1).cuda()
        self.kernel2 = torch.Tensor(w2).cuda()
        self.kernel3 = torch.Tensor(w3).cuda()
        self.kernel4 = torch.Tensor(w4).cuda()
        self.kernel5 = torch.Tensor(w5).cuda()
        self.kernel6 = torch.Tensor(w6).cuda()
        self.kernel7 = torch.Tensor(w7).cuda()
        self.kernel8 = torch.Tensor(w8).cuda()
        self.kernels = torch.concat([self.kernel1,self.kernel2,self.kernel3,self.kernel4,self.kernel5,self.kernel6,self.kernel7,self.kernel8],dim=0)
        self.kernels = self.kernels.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.use_nonlocal = use_nonlocal
        self.out_conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False),
                                      nn.BatchNorm2d(self.in_channels),
                                      nn.ReLU())
        self.trans_layer = nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=self.kernel_size+1,stride=self.kernel_size,padding=self.kernel_size//2)
        self.query_conv = nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False)
        for _ in range(len(self.shifts)):
            self.value_convs.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False))
            self.key_convs.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False))
    def Extract_layer(self,cen,b,w,h):
        keys = []
        values = []
        size = w*h//(self.kernel_size*self.kernel_size)
        key_value_cen = self.trans_layer(cen)
        for i in range(len(self.shifts)):
            value = self.value_convs[i](cen)
            key = self.key_convs[i](key_value_cen)
            value = torch.nn.functional.conv2d(weight=self.kernels,stride=1,padding="same",input=value,groups=self.in_channels,dilation=self.shifts[i])
            key = torch.nn.functional.conv2d(weight=self.kernels, stride=1, padding="same", input=key,groups=self.in_channels,dilation=self.shifts[i])
            keys.append(key)
            values.append(value)
        querys = self.query_conv(key_value_cen)
        keys = torch.stack(keys,dim=2)
        values = torch.stack(values,dim=2)
        keys = torch.nn.functional.normalize(keys.view(b,self.in_channels,8*self.num_shift,size),dim=-1).transpose(-2,-1)
        querys = torch.nn.functional.normalize(querys.view(b,self.in_channels,1,size),dim=-1)
        values = values.view(b,self.in_channels,8*self.num_shift,w*h)
        weight_score = torch.matmul(querys,keys)
        weight_score = torch.nn.functional.normalize(weight_score,dim=-1)
        out = torch.matmul(weight_score,values).view(b,self.in_channels,w,h)
        return out
    def forward(self,cen):
        b,_,w,h= cen.shape
        out = self.Extract_layer(cen,b,w,h)
        out = self.out_conv(out)
        return out
class ExpansionInfoModule(nn.Module):
    def __init__(self,in_channels,out_channels,shifts,kernel_size,use_activate=True,use_bias=False,use_avg=True):
        super(ExpansionInfoModule,self).__init__()
        #The hyper parameters settting
        self.use_avg = use_avg
        self.convs_list=nn.ModuleList()
        delta1=np.array([[[1, 0, 0], [0, -1, 0], [0, 0, 0]],
                         [[0, 1, 0], [0, -1, 0], [0, 0, 0]],
                         [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, -1, 1], [0, 0, 0]]])
        delta1=delta1.reshape(4,1,3,3)
        delta2=delta1[:,:,::-1,::-1].copy()
        delta=np.concatenate([delta1,delta2],axis=0)
        w1,w2,w3,w4,w5,w6,w7,w8=np.array_split(delta,8)
        self.in_channels = max(in_channels,1)
        self.shifts =shifts
        self.use_activate = use_activate
        self.num_heads = len(self.shifts)
        self.value_convs = nn.ModuleList()
        self.query_convs = nn.ModuleList()
        self.key_convs = nn.ModuleList()
        self.kernel_size = kernel_size
        #After Extraction, we analyze the outcome of the extraction.
        self.num_layer= 8
        self.num_shift= len(shifts)
        self.value_key_convs=nn.ModuleList()
        self.kernel1 = torch.Tensor(w1).cuda()
        self.kernel2 = torch.Tensor(w2).cuda()
        self.kernel3 = torch.Tensor(w3).cuda()
        self.kernel4 = torch.Tensor(w4).cuda()
        self.kernel5 = torch.Tensor(w5).cuda()
        self.kernel6 = torch.Tensor(w6).cuda()
        self.kernel7 = torch.Tensor(w7).cuda()
        self.kernel8 = torch.Tensor(w8).cuda()
        # self.kernel9 = torch.Tensor(center).cuda()
        self.kernels = torch.concat([self.kernel1,self.kernel2,self.kernel3,self.kernel4,self.kernel5,self.kernel6,self.kernel7,self.kernel8],dim=0)
        self.kernels = self.kernels.repeat(self.in_channels, 1, 1, 1).contiguous()
        if self.use_activate:
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=in_channels*2,kernel_size=1,stride=1,bias=use_bias),
                nn.BatchNorm2d(in_channels*2),
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels*2,out_channels=in_channels,kernel_size=1,stride=1,bias=use_bias),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            )
        else:
            self.out_conv = nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=use_bias)
        if self.use_avg:
            self.pool_layer = nn.AvgPool2d((self.kernel_size,self.kernel_size))
        else:
            self.pool_layer = nn.MaxPool2d((self.kernel_size,self.kernel_size))
        self.pool_layer = nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=self.kernel_size+1,stride=self.kernel_size,padding=self.kernel_size//2,bias=False)
        self.query_conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False))
        self.value_conv = nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False)
        self.key_conv = nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False)
        for _ in range(len(self.shifts)):
            self.value_convs.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False))
            self.key_convs.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False))
    def Extract_layer(self,cen,b,w,h):
        keys = []
        values = []
        querys = []
        size = w*h//(self.kernel_size*self.kernel_size)
        if self.kernel_size!=1:
            key_value_cen = self.pool_layer(cen)
        else:
            key_value_cen = cen
        keys.append(self.key_conv(key_value_cen).view(b,self.in_channels,1,size))
        values.append(self.value_conv(cen).view(b,self.in_channels,1,w*h))
        for i in range(len(self.shifts)):
            value = self.value_convs[i](cen)
            key = self.key_convs[i](key_value_cen)
            value = torch.nn.functional.conv2d(weight=self.kernels,stride=1,padding="same",input=value,groups=self.in_channels,dilation=self.shifts[i]).view(b,self.in_channels,8,w*h)
            key = torch.nn.functional.conv2d(weight=self.kernels, stride=1, padding="same", input=key,groups=self.in_channels,dilation=self.shifts[i]).view(b,self.in_channels,8,size)
            keys.append(key)
            values.append(value)
        querys = self.query_conv(key_value_cen)
        keys = torch.concat(keys,dim=2)
        values = torch.concat(values,dim=2)
        keys = torch.nn.functional.normalize(keys,dim=-1).transpose(-2,-1)
        querys = torch.nn.functional.normalize(querys.view(b,self.in_channels,1,size),dim=-1)
        weight_score = torch.matmul(querys,keys)
        weight_score = torch.nn.functional.normalize(weight_score,dim=-1)
        out = torch.matmul(weight_score,values).view(b,self.in_channels,w,h)
        return out
    def forward(self,cen):
        b,_,w,h= cen.shape
        out = self.Extract_layer(cen,b,w,h)
        out = self.out_conv(out)
        return out
class ExpansionDownModule(nn.Module):
    def __init__(self,in_channels,out_channels,shifts,down_scale,use_activate=True,use_bias=False,use_avg=True):
        super(ExpansionDownModule,self).__init__()
        #The hyper parameters settting
        self.use_avg = use_avg
        self.convs_list=nn.ModuleList()
        delta1=np.array([[[1, 0, 0], [0, -1, 0], [0, 0, 0]],
                         [[0, 1, 0], [0, -1, 0], [0, 0, 0]],
                         [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, -1, 1], [0, 0, 0]]])
        delta1=delta1.reshape(4,1,3,3)
        delta2=delta1[:,:,::-1,::-1].copy()
        delta=np.concatenate([delta1,delta2],axis=0)
        w1,w2,w3,w4,w5,w6,w7,w8=np.array_split(delta,8)
        self.in_channels = max(in_channels,1)
        self.shifts =shifts
        self.use_activate = use_activate
        self.num_heads = len(self.shifts)
        self.value_convs = nn.ModuleList()
        self.query_convs = nn.ModuleList()
        self.key_convs = nn.ModuleList()
        self.down_scale = down_scale
        #After Extraction, we analyze the outcome of the extraction.
        self.num_layer= 8
        self.num_shift= len(shifts)
        self.value_key_convs=nn.ModuleList()
        self.kernel1 = torch.Tensor(w1).cuda()
        self.kernel2 = torch.Tensor(w2).cuda()
        self.kernel3 = torch.Tensor(w3).cuda()
        self.kernel4 = torch.Tensor(w4).cuda()
        self.kernel5 = torch.Tensor(w5).cuda()
        self.kernel6 = torch.Tensor(w6).cuda()
        self.kernel7 = torch.Tensor(w7).cuda()
        self.kernel8 = torch.Tensor(w8).cuda()
        # self.kernel9 = torch.Tensor(center).cuda()
        self.kernels = torch.concat([self.kernel1,self.kernel2,self.kernel3,self.kernel4,self.kernel5,self.kernel6,self.kernel7,self.kernel8],dim=0)
        self.kernels = self.kernels.repeat(self.in_channels, 1, 1, 1).contiguous()
        if self.use_activate:
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,bias=use_bias),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            )
        else:
            self.out_conv = nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=use_bias)
        if self.use_avg:
            self.pool_layer = nn.AvgPool2d((self.down_scale,self.down_scale))
        else:
            self.pool_layer = nn.MaxPool2d((self.down_scale,self.down_scale))
        self.query_conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False))
        self.value_conv = nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False)
        self.key_conv = nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False)
        for _ in range(len(self.shifts)):
            self.value_convs.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False))
            self.key_convs.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False))
    def Extract_layer(self,cen,b,w,h):
        keys = []
        values = []
        querys = []
        size = w*h//(self.down_scale*self.down_scale)
        if self.down_scale!=1:
            key_value_cen = self.pool_layer(cen)
        else:
            key_value_cen = cen
        keys.append(self.key_conv(key_value_cen).view(b,self.in_channels,1,size))
        values.append(self.value_conv(key_value_cen).view(b,self.in_channels,1,size))
        for i in range(len(self.shifts)):
            value = self.value_convs[i](key_value_cen)
            key = self.key_convs[i](key_value_cen)
            value = torch.nn.functional.conv2d(weight=self.kernels,stride=1,padding="same",input=value,groups=self.in_channels,dilation=self.shifts[i]).view(b,self.in_channels,8,size)
            key = torch.nn.functional.conv2d(weight=self.kernels, stride=1, padding="same", input=key,groups=self.in_channels,dilation=self.shifts[i]).view(b,self.in_channels,8,size)
            keys.append(key)
            values.append(value)
        querys = self.query_conv(key_value_cen)
        keys = torch.concat(keys,dim=2)
        values = torch.concat(values,dim=2)
        keys = torch.nn.functional.normalize(keys,dim=-1).transpose(-2,-1)
        querys = torch.nn.functional.normalize(querys.view(b,self.in_channels,1,size),dim=-1)
        weight_score = torch.matmul(querys,keys)
        weight_score = torch.nn.functional.normalize(weight_score,dim=-1)
        out = torch.matmul(weight_score,values).view(b,self.in_channels,w//self.down_scale,h//self.down_scale)
        return out
    def forward(self,cen):
        b,_,w,h= cen.shape
        out = self.Extract_layer(cen,b,w,h)
        out = self.out_conv(out)
        return out