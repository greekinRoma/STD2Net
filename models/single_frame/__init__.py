# 其他网络
from .ALCNet.model_ALCNet import MPCMResNetFPN as ALCNet
from .AGPCNet.network import AGPCNet
from .ACM.model_ACM import ASKCResUNet as ACM
from .DNANet.model_DNANet import DNANet as DNANet
from .UIUNet.model_UIUNet import UIUNET as UIUNet
from .RDIAN.model_RDIAN import RDIAN as RDIAN
from .ISTDU_Net.ctNet.ctNet import ISTDU_Net
from .ResUNet.model_res_UNet import res_UNet
from .SDecNet.segmentation import SDecNet
from .DATransNet.segmentation import DATransNet
from torch import nn
class SingleNet(nn.Module):
    def __init__(self, model_name,in_channel,num_classes):
        super(SingleNet, self).__init__()
        self.model_name = model_name
        if model_name == 'ACM':
            self.model = ACM(in_channels=in_channel,classes=num_classes)
        elif model_name == 'ALCNet':
            self.model = ALCNet(in_channels=in_channel,num_classes=num_classes)
        elif model_name == 'AGPCNet':
            self.model = AGPCNet(in_channels=in_channel)
        elif model_name == 'ISTDU-Net':
            self.model = ISTDU_Net(in_channels=in_channel)
        elif model_name == 'DNANet':
            self.model = DNANet(in_channels=in_channel)
        elif model_name == 'RDIAN':
            self.model = RDIAN(in_channels=in_channel)
        elif model_name == 'ISTDU_Net':
            self.model = ISTDU_Net(in_channels=in_channel)
        elif model_name == 'res_UNet':
            self.model = res_UNet(input_channels=in_channel,num_classes=num_classes)
        elif model_name == 'SDecNet':
            self.model = SDecNet(n_classes=num_classes,n_channels=in_channel)
        elif model_name =='DATransNet':
            self.model = DATransNet(n_classes=in_channel,n_channels=num_classes)
    def forward(self, img):
        return self.model(img)