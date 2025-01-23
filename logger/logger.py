import os
import torch
class Logger():
    def __init__(self,logdir):
        self.logdir = logdir
        os.makedirs(self.logdir,exist_ok=True)
        self.best_miou_epoch = 0
        self.best_miou = 0
        self.best_Fd_epoch = 0
        self.best_Fd=0
        self.best_Fa_epoch = 0
        self.best_Fa = 0
        self.best_AUC_epoch =0
        self.best_AUC = 0
        self.log_txt = os.path.join(self.logdir,'log.txt')
        f = open(self.log_txt,'w')
        f.close()
    def write_line(self,inp_str):
        with open(self.log_txt,'a') as f:
            f.write(inp_str)
        f.write('\n')
    def save_model(self,model,path):
        torch.save(model.state_dict(), path)
    def write_epoch(self,model,epoch,Fd,Fa,AUC,mIou):
        model_path = os.path.join(self.logdir,f'{epoch}')
        self.save_model(model=model,path=model_path)
        log_text = f'{epoch}:Fd:{Fd},Fa:{Fa},AUC:{AUC},mIou:{mIou}'
        print(log_text)
        if (Fd>self.best_Fd):
            self.best_Fd = Fd
            self.best_Fd_epoch = epoch
        if (Fa>self.best_Fa):
            self.best_Fa = Fa
            self.best_Fa_epoch = epoch
        if (AUC>self.best_AUC):
            self.best_AUC = AUC
            self.best_AUC_epoch = epoch
        if (mIou>self.best_miou):
            self.best_miou = mIou
            self.best_miou_epoch = epoch
        save_log = f'Fd:({self.best_Fd_epoch},{self.best_Fd}),Fa:({self.best_Fa_epoch},{self.best_Fa}),AUC:({self.best_AUC_epoch},{self.best_AUC}),mIou:({self.best_miou_epoch},{self.best_miou})'
        print(save_log)
        self.write_line(save_log)
    def get_best_path(self):
        return os.path.join(self.logdir,f'{self.best_Fd_epoch}')