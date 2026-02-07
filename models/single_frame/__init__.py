# 其他网络
from .ALCNet.model_ALCNet import MPCMResNetFPN as ALCNet
from .AGPCNet.network import AGPCNet
from .ACM.model_ACM import ASKCResUNet as ACM
from .DNANet.model_DNANet import DNANet as DNANet
from .RPCANet.RPCANet import RPCANet 
# from .ISNet.model_ISNet import ISNet as ISNet
# from .RISTDnet.model_RISTDnet import RISTDnet as RISTDnet
from .UIUNet.model_UIUNet import UIUNET as UIUNet
from .RDIAN.model_RDIAN import RDIAN as RDIAN
from .DATransNet.segmentation import DATransNet
from .ISTDU_Net.ctNet.ctNet import ISTDU_Net
from .ResUNet.model_res_UNet import res_UNet
from .SDiffFormer.segmentation import SDiffFormer
from .Algorithm import Algorithms
from .L2SKNet.L2SKNet import L2SKNet_UNet
from .MSHNet.MSHNet import MSHNet
from .SDecNet.segmentation import SDecNet
from .SCtransNet.SCTransNet import SCTransNet
from .STBDNet.segmentation import STBDNet
from .HDNet.HDNet import HDNet
from .DRPCANet.DRPCANet import DRPCANet
from .RPCANet_plus.deepunfolding import RPCANet9
from .RPCANet_plus.deepunfolding import RPCANet_LSTM
from .LRPCANet.LRPCANet import LRPCANet
from .SDecNet_with_Haar.segmentation import SDecNet_Haar
from .SDecNet_with_DHPF.segmentation import SDecNet_DHPF
from .MiM.MiM import MiM
# from .VMamba.main import VMambaSeg
# from .LocalMamba.LocalMamba import build_seg_model
try:
    from .IRSAM.IRSAM import build_sam_IRSAM
except Exception as e:
    print(e)
from .SDecNet_orho.segmentation import SDecNet_orho
from torch import nn
import torch
from .loss import SoftIoULoss, ISNetLoss, DiceLoss
from .utils.loss.IRSAM_loss import SigmoidMetric, SamplewiseSigmoidMetric
class SingleNet(nn.Module):
    def __init__(self, model_name, mode='test',size=256, in_channel=1, num_classes=1,is_multi_frames=False,image_size=(256,256)):
        super(SingleNet, self).__init__()
        self.model_name = model_name
        self.softiou_loss = SoftIoULoss()
        self.dice_loss = DiceLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.mce_loss = torch.nn.BCELoss()
        self.model = Algorithms()
        self.is_alg =False
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.model_name = model_name
        if self.model.detect(model_name):
            self.model.set_algorithm(model_name)
            self.is_alg = True
        elif model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train',deep_supervision=False)
            else:
                self.model = DNANet(mode='test',deep_supervision=False)
        elif model_name == 'ACM':
            self.model = ACM()
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        elif model_name == 'AGPCNet':
            self.model = AGPCNet()
        elif model_name == 'UIUNet':
            if mode == 'train':
                self.model = UIUNet(mode='train',supervised=False)
            else:
                self.model = UIUNet(mode='test',supervised=False)
        elif model_name == 'ISTDU-Net':
            self.model = ISTDU_Net()
        elif model_name == 'RDIAN':
            self.model = RDIAN()
        elif model_name == 'ISTDU_Net':
            self.model = ISTDU_Net()
        elif model_name == 'DATransNet':
            self.model = DATransNet(img_size=size)
        elif model_name == 'SDiffFormer':
            self.model = SDiffFormer(img_size=size)
        elif model_name == 'res_UNet':
            self.model = res_UNet()
        elif model_name == 'L2SKNet':
            self.model = L2SKNet_UNet()
        elif model_name == 'MSHNet':
            self.model = MSHNet(input_channels=1)
        elif model_name == 'SDecNet':
            self.model = SDecNet(is_multi_frames=is_multi_frames, n_channels=self.in_channel, n_classes=self.num_classes, img_size=size)
        elif model_name == 'SCTransNet':
            self.model = SCTransNet()
        elif model_name == "RPCANet":
            self.model = RPCANet()
        elif model_name == "DRPCANet":
            self.model = DRPCANet()
        elif model_name =="RPCANet_plus":
            self.model = RPCANet_LSTM()
        elif model_name == "LRPCANet":
            self.model = LRPCANet()
        elif model_name == "SDecNet_DHPF":
            self.model = SDecNet_DHPF()
        elif model_name == "SDecNet_Haar":
            self.model  = SDecNet_Haar()
        elif model_name == "MiM":
            self.model = MiM([2]*3,[8, 16, 32, 64, 128],img_size=size)
        elif model_name == "IRSAM":
            self.model = build_sam_IRSAM(image_size=size)
        elif model_name == "SDecNet_orho":
            self.model = SDecNet_orho()
        elif model_name == "HDNet":
            self.model = HDNet(input_channels=in_channel, sueprvised=False)
        else:
            raise NotImplementedError(f'Network [{model_name}] is not found.')
    def forward(self, imgs, mode='train'):
        if self.model_name in ["RPCANet", "DRPCANet", "RPCANet_plus", "LRPCANet"]:
            return self.model(imgs, mode=mode)
        elif self.model_name == "IRSAM":
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = imgs[b_i].to(self.model.device)
                dict_input['image'] = input_image
                dict_input['original_size'] = imgs[b_i].shape[2:]
                batched_input.append(dict_input)
            if mode == "train":
                masks, edges = self.model(batched_input)
                return edges, masks
            else:
                masks, edges = self.model(batched_input)
                return masks
        else:
            return self.model(imgs)

    def loss(self, pred, gt_mask, image):
        if "RPCANet" == self.model_name:
            D, T = pred
            loss =  self.mse_loss(D, image) * 0.01 + self.softiou_loss(T,gt_mask)
        elif self.model_name == "DRPCANet":
            D, T = pred
            loss =  self.mse_loss(D, image) * 0.1 + self.softiou_loss(T,gt_mask)
        elif self.model_name == "RPCANet_plus":
            D, T = pred
            loss =  self.mse_loss(D, image) * 0.1 + self.softiou_loss(T,gt_mask)
        elif self.model_name == "LRPCANet":
            D, T = pred
            loss =  self.mse_loss(D, image) * 0.1 + self.softiou_loss(T,gt_mask)
        elif self.model_name == "IRSAM":
            edges, masks = pred
            loss = self.mce_loss(edges, gt_mask)*10. + self.dice_loss(inputs=masks, targets=gt_mask) 
        elif self.model_name == "MiM":
            loss = self.dice_loss(pred,gt_mask)
        else:
            loss = self.softiou_loss(pred, gt_mask)
        return loss