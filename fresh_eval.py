import torch 
from torch import nn
import os
from launch import parse_args
from exp import MyExp
import shutil
if __name__ == '__main__':
    args = parse_args()
    model_name = "RFR+MSHNet"
    args.model = model_name
    myexp = MyExp(args)
    evaluator = myexp.evaluator
    result_dir = './results'
    dataset_name = 'NUDT-MIRSDT'
    pth_file_path = os.path.join(f'./logs/{dataset_name}',model_name, 'best.pth')
    txt_file_path = os.path.join(f'./logs/{dataset_name}',model_name, 'log.txt')
    mIoU,Auc,Pd,Fa,Pds,Fas=evaluator.refresh_result(model_name=model_name,pth_path=pth_file_path,SpatialDeepSup=args.SpatialDeepSup)
    with open(txt_file_path,'a') as f:
        f.write(f'\nFinal Epoch:mIou:{mIoU*100:.2f},Pd:{Pd*100:.2f},Fa:{Fa*1000000:.2f},AUC:{Auc*100:.2f}')
        