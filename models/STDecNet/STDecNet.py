from torch import nn
import torch
from .TDecM import TDecM
from .SDecNet.segmentation import SDecNet
class STDecNet(nn.Module):
    def __init__(self,mid_channel,num_frame):
        super().__init__()


        self.num_frame = num_frame
        self.single_model = SDecNet(n_channels=1,n_classes=mid_channel)
        self.multi_model = TDecM(mid_channels=mid_channel,num_class=1,num_frames=num_frame)
        self.inp_conv = nn.Sequential(nn.Conv2d(in_channels=mid_channel,out_channels=mid_channel,kernel_size=3,padding=1))
        self.bn = nn.BatchNorm3d(num_features=num_frame)
    def forward(self,inp):
        features = []
        for i in range(self.num_frame):
            features.append(self.single_model(inp[:,:,i]))
        features = torch.stack(features,dim=1)
        features = self.bn(features)
        out = self.multi_model(features)
        out = out[:,None,:,:,:]
        return out