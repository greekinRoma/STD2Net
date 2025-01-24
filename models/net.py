from torch import nn
import os
from .single_frame import *
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
class SingleNet(nn.Module):
    def __init__(self, model_name):
        super(SingleNet, self).__init__()
        self.model_name = model_name
        if model_name == 'ACM':
            self.model = ACM()
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        elif model_name == 'AGPCNet':
            self.model = AGPCNet()
        elif model_name == 'ISTDU-Net':
            self.model = ISTDU_Net()
        elif model_name == 'RDIAN':
            self.model = RDIAN()
        elif model_name == 'ISTDU_Net':
            self.model = ISTDU_Net()
        elif model_name == 'res_UNet':
            self.model = res_UNet()
    def forward(self, img):
        return self.model(img)