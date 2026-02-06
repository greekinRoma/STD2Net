import torch
from .single_frame import SingleNet
from .DTUM import DTUMNet
from .RFR.RFR_framework import RFR
from .STDecNet.STDecNet import STDecNet
def model_chose(model_name, loss_func=None, SpatialDeepSup=None,in_channel=1,num_classes = 1,num_frame=5):
    model_name = model_name.strip()
    if model_name in ['ALCNet','AGPCNet','ISTDU-Net','RDIAN','ISTDU_Net','res_UNet','SDecNet','DNANet',"DATransNet","ACM","MSHNet","DNANet"]:
        net = SingleNet(model_name=model_name,in_channel=in_channel,num_classes=num_classes)
    elif 'DTUM' in model_name:
        model_name = model_name[5:]
        if model_name == "AC":
            model_name = "ACM"
        net = DTUMNet(model_name,in_channel=in_channel)
    elif 'RFR' in model_name:
        model_name = model_name[4:]
        net = RFR(head_name=model_name)
    elif model_name == "STDecNet":
        net = STDecNet(mid_channel=32,num_frame=num_frame)
    else:
        raise
    return net


def run_model(net, model, SeqData, Old_Feat, OldFlag):
    if model in ['ALCNet','AGPCNet','ISTDU-Net','RDIAN','ISTDU_Net','res_UNet','SDecNet','DNANet',"DATransNet","ACM","MSHNet","DNANet"]:
        input = SeqData[:, :, -1, :, :]
        outputs = net(input)
        outputs=outputs.unsqueeze(2)
    elif 'DTUM' in model:
        outputs = net(SeqData, Old_Feat, OldFlag)
    elif 'RFR' in model:
        input = SeqData.transpose(2,1)
        outputs = net(input)
    elif model == "STDecNet":
        outputs = net(SeqData)
    else:
        raise
    return outputs
