from models.model_config import run_model
import torch
import torch.nn as nn
from torch.autograd import Variable
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
class Trainer(object):
    def __init__(self, exp:MyExp):
        self.args = exp.args
        self.exp = exp
        # path 
        self.save_dir = self.exp.save_dir
        self.log_dir = self.exp.log_dir
        # model
        self.net = self.exp.net
        self.test_path = self.exp.test_path
        self.net_name = self.exp.net_name
        self.device = self.exp.device
        # dataloader
        self.train_dataset = self.exp.train_dataset
        self.val_dataset = self.exp.val_dataset
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
        running_loss = 0.0
        loss_last = 0.0
        self.net.train()
        for i, data in enumerate(tqdm(self.train_loader), 0):
            if i % self.training_rate != 0:
                continue

            SeqData_t, TgtData_t, m, n = data
            SeqData, TgtData = Variable(SeqData_t).to(self.device), Variable(TgtData_t).to(self.device)  # b,t,m,n  // b,1,m.n
            self.optimizer.zero_grad()

            outputs = run_model(self.net, self.net_name, SeqData, 0, 0)
            if isinstance(outputs, list):
                if isinstance(outputs[0], tuple):
                    outputs[0] = outputs[0][0]
            elif isinstance(outputs, tuple):
                outputs = outputs[0]

            if 'DNANet' in self.net_name:
                loss = 0
                if isinstance(outputs, list):
                    for output in outputs:
                        loss += self.criterion(output, TgtData.float())
                    loss /= len(outputs)
                else:
                    loss = self.criterion(outputs, TgtData.float())
            elif 'ISNet' in self.net_name and self.loss_func == 'fullySup1':   ## and 'ISNet_woTFD' not in args.model
                edge = torch.cat([TgtData, TgtData, TgtData], dim=1).float()  # b, 3, m, n
                gradmask = Get_gradientmask_nopadding()
                edge_gt = gradmask(edge)
                loss_io = self.criterion(outputs[0], TgtData.float())
                if self.args.fullySupervised:
                    outputs[1] = torch.sigmoid(outputs[1])
                    loss_edge = 10 * self.criterion2(outputs[1], edge_gt) + self.criterion(outputs[1], edge_gt)
                else:
                    loss_edge = 10 * self.criterion2(torch.sigmoid(outputs[1]), edge_gt) + self.criterion(outputs[1], edge_gt.float())
                if 'DTUM' in self.args.model or not self.args.fullySupervised:
                    alpha = 0.1
                else:
                    alpha = 1
                loss = loss_io + alpha * loss_edge
            elif 'UIU' in self.args.model:
                if 'fullySup2' in self.args.loss_func:
                    loss0, loss = self.criterion(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6], TgtData.float())
                    if not self.args.SpatialDeepSup:
                        loss = loss0   ## without SDS
                else:
                    loss = 0
                    if not self.args.SpatialDeepSup:
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
                print('model: %s, epoch=%d, i=%d, loss.item=%.10f' % (self.args.model + self.args.loss_func, epoch, i, loss_50))

        self.epoch_loss = running_loss / i * self.Gain
        print('model: %s, epoch: %d, loss: %.10f' % (self.args.model + self.args.loss_func, epoch + 1, self.epoch_loss))
        ########################################
        self.scheduler.step()
        # if optimizer.state_dict()['param_groups'][0]['lr'] < args.lrate_min:
        #     optimizer.state_dict()['param_groups'][0]['lr'] = args.lrate_min

        self.loss_list.append(self.epoch_loss)


    def validation(self, epoch):
        args = self.args
        txt = np.loadtxt(self.test_path + 'test.txt', dtype=bytes).astype(str)
        self.net.eval()

        Th_Seg = np.array(
            [0, 1e-30, 1e-20, 1e-19, 1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7,
             1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, .15, 0.2, .25, 0.3, .35, 0.4, .45, 0.5, .55, 0.6, .65, 0.7, .75,
             0.8, .85, 0.9, 0.95, 0.975, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99999, 0.999999, 0.9999999, 1])
        if epoch < args.epochs-1:
            Th_Seg = np.array([0, 1e-1, 0.2, 0.3, .35, 0.4, .45, 0.5, .55, 0.6, .65, 0.7, 0.8, 0.9, 0.95, 1])

        OldFlag = 0
        Old_Feat = torch.zeros([1,32,4,512,512]).to(self.device)  # interface for iteration input
        FalseNumBatch, TrueNumBatch, TgtNumBatch, pixelsNumBatch = [], [], [], []
        time_start = time.time()
        for i, data in enumerate(tqdm(self.val_loader), 0):
            # if i > 5: break
            if i % 100 == 0:
                OldFlag = 0
            else:
                OldFlag = 1

            with torch.no_grad():
                SeqData_t, TgtData_t, m, n = data
                SeqData, TgtData = Variable(SeqData_t).to(self.device), Variable(TgtData_t).to(self.device)

                outputs = run_model(self.net, args.model, SeqData, Old_Feat, OldFlag)
                if 'ISNet' in args.model:   ## and args.model != 'ISNet_woTFD'
                    edge_out = torch.sigmoid(outputs[1]).data.cpu().numpy()[0, 0, 0:m, 0:n]

                if isinstance(outputs, list):
                    outputs = outputs[0]
                if isinstance(outputs, tuple):
                    Old_Feat = outputs[1]
                    outputs = outputs[0]
                outputs = torch.squeeze(outputs, 2)

                Outputs_Max = torch.sigmoid(outputs)
                TestOut = Outputs_Max.data.cpu().numpy()[0, 0, 0:m, 0:n]

                pixelsNumBatch.append(np.array(m*n))
                if self.save_flag:
                    img = Image.fromarray(uint8(TestOut * 255))
                    folder_name = "%s%s/" % (self.test_save, txt[i].split('/')[0])
                    if not os.path.exists(folder_name):
                        os.mkdir(folder_name)
                    name = folder_name + txt[i].split('/')[-1].split('.')[0] + '.png'
                    img.save(name)
                    save_name = folder_name + txt[i].split('/')[-1].split('.')[0] + '.mat'
                    scio.savemat(save_name, {'TestOut': TestOut})

                    if 'ISNet' in args.model:   ## and args.model != 'ISNet_woTFD'
                        edge_out = Image.fromarray(uint8(edge_out * 255))
                        edge_name = folder_name + txt[i].split('/')[-1].split('.')[0] + '_EdgeOut.png'
                        edge_out.save(edge_name)

                # the statistics for detection result
                if self.writeflag:
                    for th_i in range(len(Th_Seg)):
                        FalseNum, TrueNum, TgtNum = self.eval_metrics(Outputs_Max[:,:,:m,:n], TgtData[:,:,:m,:n], Th_Seg[th_i])

                        FalseNumBatch.append(FalseNum)
                        TrueNumBatch.append(TrueNum)
                        TgtNumBatch.append(TgtNum)

        time_end = time.time()
        print('FPS=%.3f' % ((i+1)/(time_end-time_start)))

        if self.writeflag:
            if 'NUDT-MIRSDT' in args.dataset:
                writeNUDTMIRSDT_ROC(FalseNumBatch, TrueNumBatch, TgtNumBatch, pixelsNumBatch, Th_Seg, txt, self.SavePath, args, epoch)
            else:
                writeIRSeq_ROC(FalseNumBatch, TrueNumBatch, TgtNumBatch, pixelsNumBatch, Th_Seg, self.SavePath, args, epoch)


    def savemodel(self, epoch):
        self.ModelPath, self.ParameterPath, self.SavePath = self.exp.generate_savepath(epoch, self.epoch_loss)
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
    def launch(self):
        if self.args.train == 1:
            for epoch in range(self.args.epochs):
                self.training(epoch)
                if (epoch+1)%self.args.evalepoch == 0:
                    self.savemodel(epoch)
                    self.validation(epoch)
        # self..savemodel()
        self.saveloss()
        print('finished training!')
        if self.args.test == 1:
            #####################################################
            self.ModelPath = self.args.pth_path
            self.test_save = self.SavePath[0:-1] + '_visualization/'
            self.net = torch.load(self.ModelPath, map_location=self.device)
            print('load OK!')
            epoch = self.args.epochs
            #####################################################
            self.validation(epoch)
    












