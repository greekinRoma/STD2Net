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
import time
try:
    from thop import profile
except ImportError:
    print("正在尝试自动安装 thop 库以计算复杂度...")
    os.system('pip install thop')
    from thop import profile

def measure_performance(model, model_name, input_shape, device):
    """
    自动化性能测试：计算 Params, GFLOPS 和 FPS
    input_shape: (Batch, Channel, Frames, H, W)
    """
    dummy_input = torch.randn(input_shape).to(device)
    # 根据你的代码逻辑，Old_Feat 的通道通常是固定的 32
    dummy_feat = torch.zeros([1, 32, 4, input_shape[-2], input_shape[-1]]).to(device)
    
    # 1. 计算 Params 和 GFLOPS
    # 注意：profile 默认追踪 forward。如果 run_model 有复杂逻辑，结果为近似值
    model = model.eval()
    print(model_name)
    if 'DTUM' in model_name:
        flops, params = profile(model, inputs=(dummy_input, dummy_feat, False), verbose=False)
    elif 'RFR' in model_name:
        dummy_input = dummy_input.transpose(2,1)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    elif model_name == "STDecNet":
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    elif model_name == "DQAligner":
        flops, params = profile(model, inputs=(dummy_input,None, 0, False), verbose=False)
    else:
        dummy_input = dummy_input[:, :, -1, :, :]
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    gflops = flops / 1e9
    params_m = params / 1e6
    dummy_input = torch.randn(input_shape).to(device)
    # 2. 计算 FPS
    # 预热 GPU
    for _ in range(10):
        _ = run_model(model, model_name, dummy_input, dummy_feat)
    
    torch.cuda.synchronize()
    start_time = time.time()
    iters = 100
    for _ in range(iters):
        _ = run_model(model, model_name, dummy_input, dummy_feat)
    torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    fps = iters / total_time
    
    return params_m, gflops, fps

if __name__ == '__main__':
    args = parse_args()
    result_dir = 'result_images'
    os.makedirs(result_dir, exist_ok=True)
    log_dir = 'logs'
    parameter_path = f'{result_dir}/parameter.txt'
    f = open(parameter_path,'w')
    # 存储所有模型的性能数据
    performance_registry = []

    for data_name in os.listdir(log_dir):
        data_dir = os.path.join(log_dir, data_name)
        if not os.path.isdir(data_dir): continue
        
        img_dir = os.path.join(result_dir, data_name)
        
        for model_name in os.listdir(data_dir):
            if "SDecNet" in model_name:
                continue
            
            pth_path = os.path.join(data_dir, model_name, 'best.pth')
            if not os.path.exists(pth_path):
                continue

            # 初始化模型
            args.model = model_name
            args.dataset = data_name
            myexp = MyExp(args)
            device = myexp.get_device()
            
            # 加载权重
            model_weights = torch.load(pth_path, map_location=device, weights_only=True)
            model = myexp.get_net().to(device).eval()
            model.load_state_dict(model_weights)

            # --- 自动性能评估 ---
            # 动态获取输入尺寸
            H, W = myexp.img_size if hasattr(myexp, 'img_size') else (256, 256)
            input_shape = (1, args.in_channel, args.num_frames, H, W)
            
            p, g, f = measure_performance(model, model_name, input_shape, device)
            
            print(f"\n{'='*40}")
            print(f"Model: {model_name} | Dataset: {data_name}")
            print(f"Params: {p:.3f} M | GFLOPS: {g:.3f} | FPS: {f:.2f}")
            print(f"{'='*40}")
            file = open(parameter_path,'a')
            file.write(f"{model_name} {data_name} {(H,W)} & {p:.2f} & {g:.2f} & {f:.2f}\n")
            performance_registry.append({
                "Model": model_name,
                "size":(H,W),
                "Params(M)": p,
                "GFLOPS": g,
                "FPS": f
            })
            file.close()
    