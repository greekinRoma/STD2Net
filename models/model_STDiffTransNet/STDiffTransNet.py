from .SDiffTransNet.segmentation import SDiffTransNet
from torch import nn
import torch
from models.layers import DTUM
class STDiffTransNet(nn.Module):
    def __init__(self):
        super(STDiffTransNet,self).__init__()
        self.SDiffTransNet = SDiffTransNet(n_classes=1)
    def forward(self,input):
        output = self.SDiffTransNet(input)
        return output

class SDiffTransNet_DTUM(nn.Module):
    def __init__(self):
        super(SDiffTransNet_DTUM,self).__init__()
        
        self.SDiffTransNet = SDiffTransNet(n_classes=16)
        self.DTUM = DTUM(16,num_classes=1,num_frames=5)
        
    def forward(self,X_In,Old_Feat,OldFlag):
        
        FrameNum = X_In.shape[2]
        Features = X_In[:,:,-1,:,:]
        Features = self.SDiffTransNet(Features)
        Features = torch.unsqueeze(Features, 2)
        
        if OldFlag ==1:
            Features = torch.cat([Old_Feat, Features], 2)
        #进行特征拼接
        
        elif OldFlag == 0 and FrameNum >1:
            for i_fra in range(FrameNum-1):
                x_t = X_In[:,:,-2-i_fra,:,:]
                x_t = self.SDiffTransNet(x_t)
                x_t = torch.unsqueeze(x_t, 2)
                Features = torch.cat([x_t, Features],2)
        
        X_Out = self.DTUM(Features)
        Old_Feat = Features[:,:,1:,:,:]
        return X_Out, Old_Feat