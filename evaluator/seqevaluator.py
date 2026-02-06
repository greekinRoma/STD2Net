import torch
import numpy as np
from models.model_config import run_model
import time
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.metrics import auc
from evaluator.mIoU import mIoU
from models.model_config import model_chose
class SeqEvaluator():
    def __init__(self,model_name,epochs,val_loader,device,eval_metrics):
        self.model_name = model_name
        self.epochs = epochs
        self.val_loader = val_loader
        self.img_size = val_loader.dataset.img_size
        self.device = device
        self.eval_metrics = eval_metrics
        self.mIou = mIoU()
    def get_AUC(self,Fa_all,Pd_all):
        auc_all = auc(Fa_all, Pd_all)
        return auc_all
    def get_Pd(self,TrueNumAll,TgtNumAll):
        Pd_all = np.sum(TrueNumAll[:, :], axis=0) / np.sum(TgtNumAll[:, :], axis=0)
        return Pd_all
    def get_Fa(self,FalseNumAll,pixelsNumber):
        Fa_all = np.sum(FalseNumAll[:, :], axis=0) / pixelsNumber.sum()
        return Fa_all
    def get_mIou(self):
        return self.mIou.get()
    def get_results(self,model):
        self.mIou.reset()
        Th_Seg = np.array(
            [0, 1e-1, 0.2, 0.3, .35, 0.4, .45, 0.5, .55, 0.6, .65, 0.7, 0.8, 0.9, 0.95, 1])
        OldFlag = 0
        Old_Feat = torch.zeros([1,32,4,self.img_size[0],self.img_size[1]]).to(self.device)  # interface for iteration input
        FalseNumBatch, TrueNumBatch, TgtNumBatch, pixelsNumBatch = [], [], [], []
        seg_index = list(Th_Seg).index(0.5)
        for i, data in enumerate(tqdm(self.val_loader), 0):
            # if i > 5: break
            if i % 100 == 0:
                OldFlag = 0
            else:
                OldFlag = 1

            with torch.no_grad():
                SeqData_t, TgtData_t, m, n = data
                SeqData, TgtData = Variable(SeqData_t).to(self.device), Variable(TgtData_t).to(self.device)
                outputs = run_model(model, self.model_name, SeqData, Old_Feat, OldFlag)
                if isinstance(outputs, list):
                    outputs = outputs[0]
                if isinstance(outputs, tuple):
                    Old_Feat = outputs[1]
                    outputs = outputs[0]
                outputs = torch.squeeze(outputs, 1)
                TgtData = torch.squeeze(TgtData,1)
                output=outputs[:,-1,:m,:n]
                target=TgtData[:,-1,:m,:n]
                self.mIou.update(preds=output,labels=target)
                Outputs_Max = torch.sigmoid(outputs)
                pixelsNumBatch.append(np.array(m*n))
                
                for th_i in range(len(Th_Seg)):
                        FalseNum, TrueNum, TgtNum = self.eval_metrics(Outputs_Max[:,:,:m,:n], TgtData[:,:,:m,:n], Th_Seg[th_i])
                        FalseNumBatch.append(FalseNum)
                        TrueNumBatch.append(TrueNum)
                        TgtNumBatch.append(TgtNum)
                            
        # 计算方法
        #############################################
        print(np.array(FalseNumBatch).shape)
        FalseNumAll = np.array(FalseNumBatch).reshape((20, -1, len(Th_Seg))).sum(axis=1)
        print(FalseNumAll.shape)
        TrueNumAll = np.array(TrueNumBatch).reshape((20, -1, len(Th_Seg))).sum(axis=1)
        TgtNumAll = np.array(TgtNumBatch).reshape((20, -1, len(Th_Seg))).sum(axis=1)
        pixelsNumber = np.array(pixelsNumBatch).reshape(20, -1).sum(axis=1)

        Pds = self.get_Pd(TrueNumAll,TgtNumAll)
        Fas = self.get_Fa(FalseNumAll,pixelsNumber)
        Auc = self.get_AUC(Fas,Pds)
        
        Pd = Pds[seg_index]
        Fa = Fas[seg_index]
        mIoU = self.get_mIou()
        return mIoU,Auc,Pd,Fa
    def get_final_result(self,model):
        model.eval()
        self.mIou.reset()
        Th_Seg = np.array(
            [0, 1e-30, 1e-20, 1e-19, 1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7,
             1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, .15, 0.2, .25, 0.3, .35, 0.4, .45, 0.5, .55, 0.6, .65, 0.7, .75,
             0.8, .85, 0.9, 0.95, 0.975, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99999, 0.999999, 0.9999999, 1])
        OldFlag = 0
        Old_Feat = torch.zeros([1,32,self.img_size[0],self.img_size[1]]).to(self.device)  # interface for iteration input
        FalseNumBatch, TrueNumBatch, TgtNumBatch, pixelsNumBatch = [], [], [], []
        seg_index = list(Th_Seg).index(0.5)
        for i, data in enumerate(tqdm(self.val_loader), 0):
            # if i > 5: break
            if i % 100 == 0:
                OldFlag = 0
            else:
                OldFlag = 1

            with torch.no_grad():
                SeqData_t, TgtData_t, m, n = data
                SeqData, TgtData = Variable(SeqData_t).to(self.device), Variable(TgtData_t).to(self.device)
                outputs = run_model(model, self.model_name, SeqData, Old_Feat, OldFlag)
                if isinstance(outputs, list):
                    outputs = outputs[0]
                if isinstance(outputs, tuple):
                    Old_Feat = outputs[1]
                    outputs = outputs[0]
                outputs = torch.squeeze(outputs, 1)
                TgtData = torch.squeeze(TgtData,1)
                output=outputs[:,-1,:m,:n]
                target=TgtData[:,-1,:m,:n]

                self.mIou.update(preds=output,labels=target)
                Outputs_Max = torch.sigmoid(outputs)
                pixelsNumBatch.append(np.array(m*n))
                for th_i in range(len(Th_Seg)):
                        FalseNum, TrueNum, TgtNum = self.eval_metrics(Outputs_Max[:,:,:m,:n], TgtData[:,:,:m,:n], Th_Seg[th_i])
                        FalseNumBatch.append(FalseNum)
                        TrueNumBatch.append(TrueNum)
                        TgtNumBatch.append(TgtNum)
        # 计算方法
        #############################################
        FalseNumAll = np.array(FalseNumBatch).reshape((20, -1, len(Th_Seg))).sum(axis=1)
        TrueNumAll = np.array(TrueNumBatch).reshape((20, -1, len(Th_Seg))).sum(axis=1)
        TgtNumAll = np.array(TgtNumBatch).reshape((20, -1, len(Th_Seg))).sum(axis=1)
        pixelsNumber = np.array(pixelsNumBatch).reshape(20, -1).sum(axis=1)

        Pds = self.get_Pd(TrueNumAll,TgtNumAll)
        Fas = self.get_Fa(FalseNumAll,pixelsNumber)
        Auc = self.get_AUC(Fas,Pds)
        
        Pd = Pds[seg_index]
        Fa = Fas[seg_index]
        mIoU = self.get_mIou()
        return mIoU,Auc,Pd,Fa,Pds,Fas
    def refresh_result(self,model_name,pth_path,SpatialDeepSup=True):
        model_weights = torch.load(pth_path,self.device,weights_only=True)
        model = model_chose(model_name,loss_func=None,SpatialDeepSup=SpatialDeepSup)
        model = torch.nn.DataParallel(model,device_ids=[1,2])
        model.to(self.device)
        model.load_state_dict(model_weights)
        mIoU,Auc,Pd,Fa,Pds,Fas = self.get_final_result(model)
        return mIoU,Auc,Pd,Fa,Pds,Fas