import torch
from .single_frame import SingleNet
from .DTUM import DTUMNet

def model_chose(model, loss_func, SpatialDeepSup):
    num_classes = 1
    if model in ['ALCNet','AGPCNet','ISTDU-Net','RDIAN','ISTDU_Net','res_UNet','SDecNet','DNANet',"DATransNet"]:
        net = SingleNet(model_name=model,in_channel=3)
    elif 'DTUM' in model:
        model = model.strip('DTUM_')
        net = DTUMNet(model,in_channel=1)
    return net


def run_model(net, model, SeqData, Old_Feat, OldFlag):
    if model in ['ALCNet','AGPCNet','ISTDU-Net','RDIAN','ISTDU_Net','res_UNet','SDecNet','DNANet','DATransNet']:
        input = SeqData[:, :, -1, :, :].repeat(1, 3, 1, 1)
        outputs = net(input)
    elif 'DTUM' in model:
        outputs = net(SeqData, Old_Feat, OldFlag)
    return outputs
