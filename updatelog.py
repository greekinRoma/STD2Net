import torch 
from torch import nn
import os
from launch import parse_args,setloader
from logger.logger import Logger
from exp import MyExp
from models.model_config import model_chose
from evaluator.seqevaluator import SeqEvaluator
from ShootingRules import ShootingRules
from torch.utils.data import DataLoader
import shutil
class UpdateLogger(object):
    def __init__(self,exp:MyExp):
        pass
if __name__ == '__main__':
    main_dir  = 'logs'
    for model_name in os.listdir(main_dir):
        print(model_name)
        args = parse_args()
        args.model = model_name
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model_path = os.path.join(main_dir,model_name)
        net = model_chose(model=model_name,loss_func= args.loss_func, SpatialDeepSup=args.SpatialDeepSup)
        net = net.to(device)
        train_dataset,val_dataset = setloader(args)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, )
        evaluator = SeqEvaluator(args.model,epochs=args.epochs,val_loader=val_loader,device=args.device,eval_metrics=ShootingRules())
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        log_dir = os.path.join(args.logsDir,args.model)
        SavePath = args.saveDir + args.model + '_SpatialDeepSup' + str(args.SpatialDeepSup) + '_' + args.loss_func + '/'
        logger = Logger(log_dir,SavePath)
        if args.DataParallel:
            net = nn.DataParallel(net,device_ids=[1,2])
            pass
        for epoch,pth_name in enumerate(os.listdir(model_path)):
            if pth_name.endswith('pth'):
                pth_path = os.path.join(model_path,pth_name)
                weights = torch.load(pth_path)
                net.load_state_dict(weights)
                net.eval()
                mIoU,Auc,Pd,Fa=evaluator.get_results(net)
                logger.write_epoch(model=net,epoch=epoch,Fd=Pd,Fa=Fa,AUC=Auc,mIou=mIoU)
        model = torch.load(logger.get_best_path(), map_location=device)
        net.load_state_dict(model)
        net.eval()
        mIoU,Auc,Pd,Fa,Pds,Fas = evaluator.get_final_result(net)
        logger.write_final(Pd,Fa,Auc,mIoU,Pds,Fas)
        shutil.copy(logger.get_best_path(),os.path.join(os.path.dirname(logger.get_best_path()),'best.pth'))
        

                
                
                
        