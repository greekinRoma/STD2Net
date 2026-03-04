from exp import MyExp
from trainer import Trainer
import torch
import argparse
import os
import numpy as np
from models.model_config import run_model
from launch import parse_args
from setting.read_setting import generate_args,read_excel,begin_excel,finish_excel
from tqdm import tqdm
from torch.autograd import Variable
import cv2
def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Infrared_target_detection_overall')
    parser.add_argument('--DataPath',  type=str, default='./dataset/', help='Dataset path [default: ./dataset/]')
    parser.add_argument('--dataset',   type=str, default='IRDST', help='Dataset name [dafult: NUDT-MIRSDT]')
    parser.add_argument('--align',  default='False', action='store_true', help='align input frames')
    parser.add_argument('--training_rate', type=int, default=1, help='Rate of samples in training (1/n) [default: 1]')
    parser.add_argument('--saveDir',   type=str, default='./results/',
                            help='Save path [defaule: ./results/]')
    parser.add_argument('--logsDir',   type=str, default='./logs/',
                            help='Save path [defaule: ./results/]')
    parser.add_argument('--train',    type=int, default=1)
    parser.add_argument('--test',     type=int, default=1)
    parser.add_argument('--pth_path', type=str, default='.', help='Trained model path')
    # model parameters
    parser.add_argument('--model',     type=str, default='SDecNet',
                        help='ResUNet_DTUM, DNANet_DTUM, ACM, ALCNet, ResUNet, DNANet, ISNet, UIU')
    parser.add_argument('--num_frames', type=int, default=5)
    parser.add_argument('--in_channel', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=1)
    # training parameters
    parser.add_argument('--loss_func', type=str, default='fullySup',
                        help='HPM, FocalLoss, OHEM, fullySup, fullySup1(ISNet), fullySup2(UIU)')
    parser.add_argument('--fullySupervised', default=True)
    parser.add_argument('--SpatialDeepSup',  default=False)
    parser.add_argument('--batchsize', type=int,   default=1)
    parser.add_argument('--epochs',    type=int,   default=20)
    parser.add_argument('--evalepoch',type=int, default=1)
    parser.add_argument('--lrate',     type=float, default=0.001)
    parser.add_argument('--lrate_min', type=float, default=1e-5)
    # loss
    parser.add_argument('--MyWgt',     default=[0.1667, 0.8333], help='Weights of positive and negative samples')
    parser.add_argument('--MaxClutterNum', type=int, default=39, help='Clutter samples in loss [default: 39]')
    parser.add_argument('--ProtectedArea', type=int, default=2,  help='1,2,3...')
    # GPU
    parser.add_argument('--DataParallel',     default=False,    help='Use one gpu or more')
    parser.add_argument('--availble_devices',type=str,default='0',help='availble devices')
    parser.add_argument('--device', type=str, default="cuda:0", help='use comma for multiple gpus')
    parser.add_argument('--device_id', type=list[int], default=[0], help='which gpu to use')
    # Excel 
    parser.add_argument('--useExcel', default=True, help='Do we use excel setting?')
    args = parser.parse_args()

    # the parser
    return args


if __name__ == '__main__':
    args = parse_args()
    result_dir = 'result_images'
    os.makedirs(result_dir,exist_ok=True)
    log_dir = 'logs'
    output = []
    for data_name in os.listdir(log_dir):
        data_dir = os.path.join(log_dir,data_name)
        img_dir = os.path.join(result_dir,data_name)
        for model_name in os.listdir(data_dir):
            if "SDecNet" in model_name:
                continue
            pth_path = os.path.join(data_dir,model_name,'best.pth')
            args.model = model_name
            args.dataset = data_name
            myexp = MyExp(args)
            model_weights = torch.load(pth_path,weights_only=True)
            model = myexp.get_net().eval()
            model.load_state_dict(model_weights)
            val_loader = myexp.get_valloader()
            device = myexp.get_device()
            Old_Feat = torch.zeros([1,32,4,myexp.img_size[0],myexp.img_size[1]]).to(device)
            os.makedirs(os.path.join(img_dir,model_name),exist_ok=True)
            for i,data in enumerate(tqdm(val_loader)):
                save_path = os.path.join(img_dir,model_name,f"{i}.png")
                with torch.no_grad():
                    SeqData_t, TgtData_t, m, n = data
                    SeqData = Variable(SeqData_t).to(device)
                    outputs = run_model(model, model_name, SeqData, Old_Feat)
                    if model_name == "DQAligner":
                        outputs = outputs[1]
                    else:
                        if isinstance(outputs, list):
                            outputs = outputs[0]
                        if isinstance(outputs, tuple):
                            Old_Feat = outputs[1]
                            outputs = outputs[0]
                    outputs = torch.squeeze(outputs, 1)
                    output=torch.sigmoid(outputs[0,-1,:m,:n])
                    output = np.array(output.detach().cpu())>0.5
                    cv2.imwrite(save_path,(output*255.).astype(np.uint8))
