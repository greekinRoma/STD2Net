import os
import torch
import shutil 
import matplotlib.pyplot as plt
class Logger():
    def __init__(self,logdir,savedir):
        self.logdir = logdir
        self.savedir = savedir
        try:
            shutil.rmtree(self.savedir)
        except:
            pass
        os.makedirs(self.logdir,exist_ok=True)
        os.makedirs(self.savedir,exist_ok=True)
        self.best_miou_epoch = 0
        self.best_miou = 0
        self.best_Fd_epoch = 0
        self.best_Fd=0
        self.best_Fa_epoch = 0
        self.best_Fa = 0
        self.best_AUC_epoch =0
        self.best_AUC = 0
        self.best_model = None
        self.log_txt = os.path.join(self.logdir,'log.txt')
        self.sav_txt = os.path.join(self.savedir,'log.txt')
        f = open(self.log_txt,'w')
        f.close()
    def write_line(self,inp_str):
        with open(self.log_txt,'a') as f:
            f.write(inp_str)
            f.write('\n')
    def save_model(self,model,path):
        torch.save(model.state_dict(), path)
    def write_epoch(self,model,epoch,Fd,Fa,AUC,mIou):
        model_path = os.path.join(self.logdir,f'{epoch}.pth')
        self.save_model(model=model,path=model_path)
        log_text = f'{epoch}:Pd:{Fd},Fa:{Fa},AUC:{AUC},mIou:{mIou}'
        print(log_text)
        self.write_line(log_text)
        if (Fd>self.best_Fd):
            self.best_Fd = Fd
            self.best_Fd_epoch = epoch
        if (Fa>self.best_Fa):
            self.best_Fa = Fa
            self.best_Fa_epoch = epoch
        if (AUC>self.best_AUC):
            self.best_AUC = AUC
            self.best_AUC_epoch = epoch
            self.best_model = model
        if (mIou>self.best_miou):
            self.best_miou = mIou
            self.best_miou_epoch = epoch
        save_log = f'Best Pd:({self.best_Fd_epoch},{self.best_Fd}),Best Fa:({self.best_Fa_epoch},{self.best_Fa}),Best AUC:({self.best_AUC_epoch},{self.best_AUC}), Best mIou:({self.best_miou_epoch},{self.best_miou})'
        print(save_log)
        print('-'*len(save_log))
        self.write_line(save_log)
        self.write_line('-'*len(save_log))
    def get_best_path(self):
        return os.path.join(self.logdir,f'{self.best_AUC_epoch}.pth')
    def write_final(self,Pd,Fa,AUC,mIou,Pds,Fas):
        log_text = f'Final Epoch:mIou:{mIou*100:.2f},Pd:{Pd*100:.2f},Fa:{Fa*100000:.2f},AUC:{AUC*100:.2f}'
        self.write_line(log_text)
        self.save_model(model=self.best_model,path=os.path.join(self.savedir,"best.pth"))
        plt.plot(Pds,Fas)
        plt.savefig(os.path.join(self.savedir,'ROC.png'))
        shutil.copy(self.log_txt,self.sav_txt)