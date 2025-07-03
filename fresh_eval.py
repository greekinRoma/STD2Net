import torch 
from torch import nn
import os
from launch import parse_args,setloader
from exp import MyExp
import shutil
if __name__ == '__main__':
    args = parse_args()
    train_dataset,val_dataset = setloader(args)
    myexp = MyExp(args,train_dataset=train_dataset,val_dataset=val_dataset)
    evaluator = myexp.evaluator
    result_dir = './results'
    for file_name in os.listdir(result_dir):
        model_names = ["res_UNet"]
        model_name = "_".join(model_names)
        pth_file_path = os.path.join('./logs',model_name)
        txt_file_path = os.path.join(result_dir,file_name)
        # print(len(os.listdir(file_path)))
        if (len(os.listdir(pth_file_path))==0): 
            continue
        sav_file_path = os.path.join(txt_file_path,'best.pth')
        txt_file_path = os.path.join(txt_file_path,'log.txt')
        pth_file_path = os.path.join(pth_file_path,'best.pth')
        shutil.copy(pth_file_path,sav_file_path)
        mIoU,Auc,Pd,Fa,Pds,Fas=evaluator.refresh_result(model_name=model_name,pth_path=pth_file_path,SpatialDeepSup=args.SpatialDeepSup)
        with open(txt_file_path,'a') as f:
            f.write(f'\nFinal Epoch:mIou:{mIoU*100:.2f},Pd:{Pd*100:.2f},Fa:{Fa*1000000:.2f},AUC:{Auc*100:.2f}')