import torch
from torch import nn
from models.model_config import model_chose
from torch.utils.data import DataLoader
from DataLoaders.MIRSDTDataLoader import TrainSetLoader, TestSetLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from losses import loss_chose
from ShootingRules import ShootingRules
import os
import time
from logger.logger import Logger
class MyExp():
    def __init__(self,args,train_dataset,val_dataset):
        self.args = args
        # GPU
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        # Logger
        self.logger = self.resetloger()
        # path 
        self.save_dir = self.args.saveDir
        self.log_dir = os.path.join(self.args.logsDir,self.args.model)
        # model
        self.loss_func = self.args.loss_func
        self.training_rate = self.args.training_rate
        self.net = model_chose(args.model, args.loss_func, args.SpatialDeepSup)
        self.net_name = args.model
        if args.DataParallel:
            self.net = nn.DataParallel(self.net)  #, device_ids=[0,1,2]).cuda()
        self.net = self.net.to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lrate, betas=(0.9, 0.99))
        self.scheduler = StepLR(self.optimizer, step_size=3, gamma=0.5, last_epoch=-1)

        self.criterion = loss_chose(args)
        self.criterion2 = nn.BCELoss()
        self.eval_metrics = ShootingRules()

        self.loss_list = []
        self.Gain = 100
        self.epoch_loss = 0
        ########### data ############
        self.train_path = self.args.DataPath + self.args.dataset + '/'
        self.test_path = self.train_path
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batchsize, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, )
        ########### save ############
        self.ModelPath, self.ParameterPath, self.SavePath = self.generate_savepath( 0, 0)
        self.test_save = self.SavePath[0:-1] + '_visualization/'
        self.writeflag = 1
        self.save_flag = 1
        if self.save_flag == 1 and not os.path.exists(self.test_save):
            os.mkdir(self.test_save)
    def generate_savepath(self,epoch, epoch_loss):

        timestamp = time.time()
        CurTime = time.strftime("%Y_%m_%d__%H_%M", time.localtime(timestamp))

        SavePath = self.args.saveDir + self.args.model + '_SpatialDeepSup' + str(self.args.SpatialDeepSup) + '_' + self.args.loss_func + '/'
        ModelPath = SavePath + 'net_' + str(epoch+1) + '_epoch_' + str(epoch_loss) + '_loss_' + CurTime + '.pth'
        ParameterPath = SavePath + 'net_para_' + CurTime + '.pth'

        if not os.path.exists(self.args.saveDir):
            os.mkdir(self.args.saveDir)
        if not os.path.exists(SavePath):
            os.mkdir(SavePath)

        return ModelPath, ParameterPath, SavePath
    def setloader(self):
        if self.args.dataset == 'NUDT-MIRSDT':
            train_dataset = TrainSetLoader(self.train_path, fullSupervision=self.args.fullySupervised)
            val_dataset = TestSetLoader(self.test_path)
        elif self.args.dataset == 'IRDST':
            train_dataset = IRDST_TrainSetLoader(self.train_path, fullSupervision=self.args.fullySupervised, align=self.args.align)
            val_dataset = IRDST_TestSetLoader(self.test_path, align=self.args.align)
        else:
            raise
        train_loader = DataLoader(train_dataset, batch_size=self.args.batchsize, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        return train_loader,val_loader
    def resetloader(self):
        self.train_loader,self.val_loader = self.setloader()
    def resetloger(self):
        return Logger(self.log_dir)