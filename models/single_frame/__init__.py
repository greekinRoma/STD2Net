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
from torch import nn
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
        elif model_name == 'SDecNet':
            self.model = SDecNet()
    def forward(self, img):
        return self.model(img)