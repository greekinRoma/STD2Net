from models.model_config import run_model
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import scipy.io as scio
import time
import osa
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import auc
import os
from models.model_ISNet.train_ISNet import Get_gradientmask_nopadding, Get_gradient_nopadding
from write_results import writeNUDTMIRSDT_ROC, writeIRSeq_ROC
from exp import MyExp
import copy 
from torch.autograd import Variable
class Trainer(object):
    def __init__(self, exp:MyExp):
        self.args = exp.args
        self.exp = exp
        # evaluator
        self.evaluator = self.exp.evaluator
        # logger
        self.logger = self.exp.logger
        # path 
        self.save_dir = self.exp.save_dir
        self.log_dir = self.exp.log_dir
        # model
        self.net = self.exp.net
        self.test_path = self.exp.test_path
        self.net_name = self.exp.net_name
        self.device = self.exp.device
        # dataloader
        self.train_loader = self.exp.train_loader
        self.val_loader = self.exp.val_loader
        # Optimizer
        self.optimizer = self.exp.optimizer
        self.scheduler = self.exp.scheduler
        self.training_rate = self.exp.training_rate
        # Criterion Function
        self.loss_func = self.exp.loss_func
        self.criterion = self.exp.criterion
        self.criterion2 = self.exp.criterion2
        self.eval_metrics = self.exp.eval_metrics
        self.loss_list = copy.copy(self.exp.loss_list)
        self.Gain = copy.copy(self.exp.Gain)
        self.epoch_loss = copy.copy(self.exp.epoch_loss)
        ########### save ############
        self.ModelPath, self.ParameterPath, self.SavePath = self.exp.ModelPath,self.exp.ParameterPath, self.exp.SavePath
        self.test_save = self.SavePath[0:-1] + '_visualization/'
        self.writeflag = self.exp.writeflag
        self.save_flag = self.exp.save_flag
        if self.save_flag == 1 and not os.path.exists(self.test_save):
            os.mkdir(self.test_save)


    def training(self, epoch):
        args = self.args
        running_loss = 0.0
        loss_last = 0.0
        self.net.train()
        for i, data in enumerate(tqdm(self.train_loader), 0):
            if i % args.training_rate != 0:
                continue
            SeqData_t, TgtData_t, m, n = data
            SeqData, TgtData = Variable(SeqData_t).to(self.device), Variable(TgtData_t).to(self.device)  # b,t,m,n  // b,1,m.n
            self.optimizer.zero_grad()
            outputs = run_model(self.net, args.model, SeqData, 0, 0)
            if isinstance(outputs, list):
                if isinstance(outputs[0], tuple):
                    outputs[0] = outputs[0][0]
            elif isinstance(outputs, tuple):
                outputs = outputs[0]

            if 'DNANet' in args.model:
                loss = 0
                if isinstance(outputs, list):
                    for output in outputs:
                        loss += self.criterion(output, TgtData.float())
                    loss /= len(outputs)
                else:
                    loss = self.criterion(outputs, TgtData.float())
            elif 'ISNet' in args.model and args.loss_func == 'fullySup1':   ## and 'ISNet_woTFD' not in args.model
                edge = torch.cat([TgtData, TgtData, TgtData], dim=1).float()  # b, 3, m, n
                gradmask = Get_gradientmask_nopadding()
                edge_gt = gradmask(edge)
                loss_io = self.criterion(outputs[0], TgtData.float())
                if args.fullySupervised:
                    outputs[1] = torch.sigmoid(outputs[1])
                    loss_edge = 10 * self.criterion2(outputs[1], edge_gt) + self.criterion(outputs[1], edge_gt)
                else:
                    loss_edge = 10 * self.criterion2(torch.sigmoid(outputs[1]), edge_gt) + self.criterion(outputs[1], edge_gt.float())
                if 'DTUM' in args.model or not args.fullySupervised:
                    alpha = 0.1
                else:
                    alpha = 1
                loss = loss_io + alpha * loss_edge
            elif 'UIU' in args.model:
                if 'fullySup2' in args.loss_func:
                    loss0, loss = self.criterion(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6], TgtData.float())
                    if not args.SpatialDeepSup:
                        loss = loss0   ## without SDS
                else:
                    loss = 0
                    if not args.SpatialDeepSup:
                        loss = self.criterion(outputs[0], TgtData.float())
                    else:
                        for output in outputs:
                            loss += self.criterion(output, TgtData.float())
            else:
                loss = self.criterion(outputs, TgtData.float())

            '''
            LogSoftmax = nn.Softmax(dim=1)
            outputs=torch.squeeze(outputs, 2)
            Outputs_Max = LogSoftmax(outputs)
            fig=plt.figure()
            ShowInd=0
            plt.subplot(221); plt.imshow(SeqData.data.cpu().numpy()[ShowInd,0,4,:,:], cmap='gray')
            plt.subplot(222); plt.imshow(TgtData.data.cpu().numpy()[ShowInd,0,:,:], cmap='gray')
            plt.subplot(223); plt.imshow(outputs.data.cpu().numpy()[ShowInd,1,:,:], cmap='gray')
            plt.subplot(224); plt.imshow(Outputs_Max.data.cpu().numpy()[ShowInd,1,:,:], cmap='gray')
            plt.show()
            '''

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if epoch == 0 and (i + 1) % 50 == 0:
                loss_50 = running_loss - loss_last
                loss_last = running_loss
                print('model: %s, epoch=%d, i=%d, loss.item=%.10f' % (args.model + args.loss_func, epoch, i, loss_50))

        self.epoch_loss = running_loss / i * self.Gain
        print('model: %s, epoch: %d, loss: %.10f' % (args.model + args.loss_func, epoch + 1, self.epoch_loss))
        ########################################
        self.scheduler.step()
        # if optimizer.state_dict()['param_groups'][0]['lr'] < args.lrate_min:
        #     optimizer.state_dict()['param_groups'][0]['lr'] = args.lrate_min

        self.loss_list.append(self.epoch_loss)


    def validation(self, epoch):
        self.net.eval()
        mIoU,Auc,Pd,Fa=self.evaluator.get_results(self.net)
        self.logger.write_epoch(model=self.net,epoch=epoch,Fd=Pd,Fa=Fa,AUC=Auc,mIou=mIoU)

        


    def savemodel(self, epoch):
        self.ModelPath, self.ParameterPath, self.SavePath = self.generate_savepath(self.args, epoch, self.epoch_loss)
        torch.save(self.net, self.ModelPath)
        torch.save(self.net.state_dict(), self.ParameterPath)
        print('save net OK in %s' % self.ModelPath)


    def saveloss(self):
        CurTime = time.strftime("%Y_%m_%d__%H_%M", time.localtime())
        print(CurTime)

        ###########save lost_list
        LossMatSavePath = self.SavePath + 'loss_list_' + CurTime + '.mat'
        scio.savemat(LossMatSavePath, mdict={'loss_list': self.loss_list})

        ############plot
        x1 = range(self.args.epochs)
        y1 = self.loss_list
        fig = plt.figure()
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch')
        plt.ylabel('train loss')
        LossJPGSavePath = self.SavePath + 'train_loss_' + CurTime + '.jpg'
        plt.savefig(LossJPGSavePath)
        # plt.show()
        print('finished Show!')
        
    def generate_savepath(self,args, epoch, epoch_loss):

        timestamp = time.time()
        CurTime = time.strftime("%Y_%m_%d__%H_%M", time.localtime(timestamp))

        SavePath = args.saveDir + args.model + '_SpatialDeepSup' + str(args.SpatialDeepSup) + '_' + args.loss_func + '/'
        ModelPath = SavePath + 'net_' + str(epoch+1) + '_epoch_' + str(epoch_loss) + '_loss_' + CurTime + '.pth'
        ParameterPath = SavePath + 'net_para_' + CurTime + '.pth'

        if not os.path.exists(args.saveDir):
            os.mkdir(args.saveDir)
        if not os.path.exists(SavePath):
            os.mkdir(SavePath)

        return ModelPath, ParameterPath, SavePath
        
    def launch(self):
        if self.args.train == 1:
            for epoch in range(self.args.epochs):
                self.training(epoch)
                if (epoch+1)%self.args.evalepoch == 0:
                    self.validation(epoch)
        # self..savemodel()
        self.saveloss()
        print('finished training!')
        if self.args.test == 1:
            print(self.logger.get_best_path())
            model = torch.load(self.logger.get_best_path(), map_location=self.device)
            self.net.load_state_dict(model)
            mIoU,Auc,Pd,Fa,Pds,Fas = self.evaluator.get_final_result(self.net)
            self.logger.write_final(Pd,Fa,Auc,mIoU,Pds,Fas)
            
    












