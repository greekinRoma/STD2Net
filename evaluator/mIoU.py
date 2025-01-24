import numpy as np
import torch
class mIoU():
    
    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()
    def update(self, preds, labels):
        preds = torch.unsqueeze(preds.float(), axis=1)
        inter, union = batch_intersection_union(preds, labels)
        self.total_inter += inter
        self.total_union += union

    def get(self):
        IoU = self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return  mIoU
    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

    
def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = torch.unsqueeze(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())
    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union