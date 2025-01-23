from exp import MyExp
from trainer import Trainer
import torch
import argparse
import os
import time
from setting.read_setting import generate_args,read_excel,begin_excel,finish_excel
from DataLoaders.MIRSDTDataLoader import TrainSetLoader, TestSetLoader
def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Infrared_target_detection_overall')
    parser.add_argument('--DataPath',  type=str, default='./dataset/', help='Dataset path [default: ./dataset/]')
    parser.add_argument('--dataset',   type=str, default='NUDT-MIRSDT', help='Dataset name [dafult: NUDT-MIRSDT]')
    parser.add_argument('--align',  default='False', action='store_true', help='align input frames')
    parser.add_argument('--training_rate', type=int, default=1, help='Rate of samples in training (1/n) [default: 1]')
    parser.add_argument('--saveDir',   type=str, default='./results/',
                            help='Save path [defaule: ./results/]')
    parser.add_argument('--logsDir',   type=str, default='./logs/',
                            help='Save path [defaule: ./results/]')
    parser.add_argument('--train',    type=int, default=1)
    parser.add_argument('--test',     type=int, default=1)
    parser.add_argument('--pth_path', type=str, default='.', help='Trained model path')

    # train
    parser.add_argument('--model',     type=str, default='SDiffTransNet',
                        help='ResUNet_DTUM, DNANet_DTUM, ACM, ALCNet, ResUNet, DNANet, ISNet, UIU')
    parser.add_argument('--loss_func', type=str, default='fullySup',
                        help='HPM, FocalLoss, OHEM, fullySup, fullySup1(ISNet), fullySup2(UIU)')
    parser.add_argument('--fullySupervised', default=True)
    parser.add_argument('--SpatialDeepSup',  default=False)
    parser.add_argument('--batchsize', type=int,   default=1)
    parser.add_argument('--epochs',    type=int,   default=20)
    parser.add_argument('--evalepoch',type=int, default=1)
    parser.add_argument('--lrate',     type=float, default=0.001)
    # parser.add_argument('--lrate_min', type=float, default=1e-5)

    # loss
    parser.add_argument('--MyWgt',     default=[0.1667, 0.8333], help='Weights of positive and negative samples')
    parser.add_argument('--MaxClutterNum', type=int, default=39, help='Clutter samples in loss [default: 39]')
    parser.add_argument('--ProtectedArea', type=int, default=2,  help='1,2,3...')
    # GPU
    parser.add_argument('--DataParallel',     default=False,    help='Use one gpu or more')
    parser.add_argument('--availble_devices',type=str,default='0,1',help='availble devices')
    parser.add_argument('--device', type=str, default="cuda:0", help='use comma for multiple gpus')
    # Excel 
    parser.add_argument('--useExcel', default=True, help='Do we use excel setting?')
    args = parser.parse_args()

    # the parser
    return args

def setloader(args):
    train_path =args.DataPath + args.dataset + '/'
    test_path = train_path
    if args.dataset == 'NUDT-MIRSDT':
        train_dataset = TrainSetLoader(train_path, fullSupervision=args.fullySupervised)
        val_dataset = TestSetLoader(test_path)
    elif args.dataset == 'IRDST':
        train_dataset = IRDST_TrainSetLoader(train_path, fullSupervision=args.fullySupervised, align=args.align)
        val_dataset = IRDST_TestSetLoader(test_path, align=args.align)
    else:
        raise
    return train_dataset,val_dataset

if __name__ == '__main__':
    args = parse_args()
    begin_excel(r'input.xlsx', 'input')
    train_dataset,val_dataset = setloader(args)
    while(True):
        try:
            main_dir=r'../'
            set_dict=read_excel(os.path.join(main_dir,'input.xlsx'),'input')
        except:
            main_dir = r'./'
            set_dict = read_excel(os.path.join(main_dir, 'input.xlsx'), 'input')
        args=generate_args(args=args,set_dict=set_dict,is_read_excel=args.useExcel)
        torch.cuda.set_device(0)
        myexp = MyExp(args,train_dataset=train_dataset,val_dataset=val_dataset)
        trainer = Trainer(myexp)
        trainer.launch()
        finish_excel(r'./input.xlsx','input')