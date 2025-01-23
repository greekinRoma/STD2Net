from exp import MyExp
from trainer import Trainer
import torch
import argparse
import os
import time
def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Infrared_target_detection_overall')
    parser.add_argument('--DataPath',  type=str, default='./dataset/', help='Dataset path [default: ./dataset/]')
    parser.add_argument('--dataset',   type=str, default='NUDT-MIRSDT', help='Dataset name [dafult: NUDT-MIRSDT]')
    parser.add_argument('--align',  default='False', action='store_true', help='align input frames')
    parser.add_argument('--training_rate', type=int, default=1, help='Rate of samples in training (1/n) [default: 1]')
    parser.add_argument('--saveDir',   type=str, default='./results/', help='Save path [defaule: ./results/]')
    parser.add_argument('--logsDir', type=str, default='./logs',help='Logs path')
    parser.add_argument('--train',    type=int, default=1)
    parser.add_argument('--test',     type=int, default=0)
    parser.add_argument('--pth_path', type=str, default='.', help='Trained model path')

    # train
    parser.add_argument('--model',     type=str, default='ResUNet_DTUM',
                        help='ResUNet_DTUM, DNANet_DTUM, ACM, ALCNet, ResUNet, DNANet, ISNet, UIU')
    parser.add_argument('--loss_func', type=str, default='fullySup',
                        help='HPM, FocalLoss, OHEM, fullySup, fullySup1(ISNet), fullySup2(UIU)')
    parser.add_argument('--fullySupervised', default=True)
    parser.add_argument('--SpatialDeepSup',  default=False)
    parser.add_argument('--batchsize', type=int,   default=1)
    parser.add_argument('--epochs',    type=int,   default=1)
    parser.add_argument('--evalepoch',type=int, default=1)
    parser.add_argument('--lrate',     type=float, default=0.001)
    # parser.add_argument('--lrate_min', type=float, default=1e-5)

    # loss
    parser.add_argument('--MyWgt',     default=[0.1667, 0.8333], help='Weights of positive and negative samples')
    parser.add_argument('--MaxClutterNum', type=int, default=39, help='Clutter samples in loss [default: 39]')
    parser.add_argument('--ProtectedArea', type=int, default=2,  help='1,2,3...')

    # GPU
    parser.add_argument('--DataParallel',     default=False,    help='Use one gpu or more')
    parser.add_argument('--device', type=str, default="cuda:0", help='use comma for multiple gpus')
    # Write and Save
    parser.add_argument('--writeflag',type=bool,default=True)
    parser.add_argument('--saveflag',type=bool, default=True)
    args = parser.parse_args()

    # the parser
    return args
if __name__ == '__main__':
    args = parse_args()
    StartTime = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    # torch.cuda.set_device(0)
    myexp = MyExp(args)
    trainer = Trainer(myexp)
    trainer.launch()