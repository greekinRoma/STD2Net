import torch
from torch import nn
from models.model_config import model_chose
from torch.utils.data import DataLoader
from DataLoaders.MIRSDTDataLoader import MIRSDTDataLoader
from DataLoaders.IRDSTDataLoader import IRDSTDataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from losses import loss_chose
from ShootingRules import ShootingRules
import os
import time
from logger.logger import Logger
from evaluator.seqevaluator import SeqEvaluator
class MyExp():
    def __init__(self,args):
        self.args = args
        # GPU
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        # dataloader
        self.dataset_name = self.args.dataset.strip()
        self.DataPath = self.args.DataPath
        self.train_path = os.path.join(self.DataPath,self.dataset_name)
        self.test_path = self.train_path
        self.resetloader()
        self.img_size = self.train_loader.dataset.img_size
        # model
        self.num_frames = args.num_frames
        self.loss_func = self.args.loss_func.strip()
        self.SpatialDeepSup = self.args.SpatialDeepSup
        self.training_rate = self.args.training_rate
        self.net_name = args.model.strip()
        self.in_channel = args.in_channel
        self.num_classes = args.num_classes
        self.net = model_chose(self.net_name, self.loss_func, self.SpatialDeepSup,in_channel=self.in_channel,num_classes=self.num_classes,num_frame=self.num_frames,img_size=self.img_size)
        torch.cuda.set_device(args.device)
        self.net = self.net.to(self.device)
        if args.DataParallel:
            self.net = nn.DataParallel(self.net,device_ids=args.device_id)
        # Optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lrate, betas=(0.9, 0.99))
        self.scheduler = StepLR(self.optimizer, step_size=3, gamma=0.5, last_epoch=-1)
        self.criterion = loss_chose(args)
        self.criterion2 = nn.BCELoss()
        self.loss_list = []
        self.Gain = 100
        self.epoch_loss = 0

        # logger     
        self.save_dir = os.path.join(self.args.saveDir,self.dataset_name)
        self.log_dir = os.path.join(self.args.logsDir,self.dataset_name,self.args.model)
        ########### save ############
        self.ModelPath, self.ParameterPath, self.SavePath = self.generate_savepath( 0, 0)
        self.test_save = self.SavePath[0:-1] + '_visualization/'
        self.writeflag = 0
        self.save_flag = 0
         # Logger
        self.logger = self.resetloger()
        # Eval
        self.eval_metrics = ShootingRules()
        self.evaluator = SeqEvaluator(self.args.model,epochs=self.args.epochs,val_loader=self.val_loader,device=self.device,eval_metrics=self.eval_metrics)
        if self.save_flag == 1 and not os.path.exists(self.test_save):
            os.mkdir(self.test_save)
    def generate_savepath(self,epoch, epoch_loss):
        timestamp = time.time()
        CurTime = time.strftime("%Y_%m_%d__%H_%M", time.localtime(timestamp))
        Save_dir_name = self.net_name + '_SpatialDeepSup' + str(self.SpatialDeepSup) + '_' + self.loss_func + '/'
        Weight_name = 'net_' + str(epoch+1) + '_epoch_' + str(epoch_loss) + '_loss_' + CurTime + '.pth'
        SavePath = os.path.join(self.save_dir, Save_dir_name)
        ModelPath = os.path.join(SavePath, Weight_name)
        ParameterPath = os.path.join(SavePath, 'net_para_' + CurTime + '.pth')
        if not os.path.exists(self.args.saveDir):
            os.makedirs(self.args.saveDir)
        if not os.path.exists(SavePath):
            os.makedirs(SavePath)

        return ModelPath, ParameterPath, SavePath
    
    def setloader(self):
        if self.args.dataset == 'NUDT-MIRSDT':
            train_dataset = MIRSDTDataLoader(self.train_path, fullSupervision=self.args.fullySupervised,mode='train',cache_type='disk',data_dir="dataset",cache_dir_name="MIRSDT",path_filename="train",use_cache=False)
            val_dataset = MIRSDTDataLoader(self.test_path, fullSupervision=self.args.fullySupervised,mode='test',cache_type='disk',data_dir="dataset",cache_dir_name="MIRSDT",path_filename="test",use_cache=False)
        elif self.args.dataset == 'IRDST':
            train_dataset = IRDSTDataLoader(self.train_path, fullSupervision=self.args.fullySupervised,mode='train')
            val_dataset = IRDSTDataLoader(self.test_path, fullSupervision=self.args.fullySupervised,mode='test')
        else:
            raise Exception("Dataset not implemented!")
        train_loader = DataLoader(train_dataset, batch_size=self.args.batchsize, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        return train_loader,val_loader
    def resetloader(self):
        self.train_loader,self.val_loader = self.setloader()
    def resetloger(self):
        return Logger(self.log_dir,self.SavePath)