import torch 
from torch import nn
from evaluator.seqevaluator import SeqEvaluator
import os
from launch import parse_args,setloader
from exp import MyExp
if __name__ == '__main__':
    args = parse_args()
    train_dataset,val_dataset = setloader(args)
    myexp = MyExp(args,train_dataset=train_dataset,val_dataset=val_dataset)
    evaluator = myexp.evaluator
    result_dir = './results'
    for file_name in os.listdir(result_dir):
        model_name = file_name.split('_')[0]
        file_path = os.path.join(result_dir,file_name)
        if (len(os.listdir(file_path))==0): 
            continue
        txt_file_path = os.path.join(file_path,'log.txt')
        pth_file_path = os.path.join(file_path,'best.pth')
        mIoU,Auc,Pd,Fa,Pds,Fas=evaluator.refresh_result(model_name=model_name,pth_path=pth_file_path,SpatialDeepSup=args.SpatialDeepSup)
        with open(txt_file_path,'a') as f:
            f.write(f'\nFinal Epoch:Pd:{Pd},Fa:{Fa},AUC:{Auc},mIou:{mIoU}')