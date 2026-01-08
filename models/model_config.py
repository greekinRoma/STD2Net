import torch
from .single_frame import SingleNet
from .DTUM import DTUMNet
# from .RFR.RFR_framework import RFR
from .STDecNet.STDecNet import STDecNet
def model_chose(model, loss_func=None, SpatialDeepSup=None):
    num_classes = 1
    if model in ['ALCNet','AGPCNet','ISTDU-Net','RDIAN','ISTDU_Net','res_UNet','SDecNet','DNANet',"DATransNet","ACM"]:
        net = SingleNet(model_name=model,in_channel=1,num_classes=1)
    elif 'DTUM' in model:
        model = model[5:]
        if model == "AC":
            model = "ACM"
        net = DTUMNet(model,in_channel=1)
    elif 'RFR' in model:
        model = model[4:]
        net = RFR(head_name=model)
    elif model == "STDecNet":
        net = STDecNet(mid_channel=32,num_frame=5)
    else:
        raise
    return net


def run_model(net, model, SeqData, Old_Feat, OldFlag):
    if model in ['ALCNet','AGPCNet','ISTDU-Net','RDIAN','ISTDU_Net','res_UNet','SDecNet','DNANet','DATransNet']:
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
    return outputs
