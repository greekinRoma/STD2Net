import os
import numpy as np
from PIL import Image
from scipy.io import savemat
import shutil
def load_and_normalize_image(path, size=None):
    img = Image.open(path).convert('L')  # 灰度图
    return np.array(img)

def create_right_aligned_stacks(folder_path, output_folder, stack_size=5):
    os.makedirs(output_folder, exist_ok=True)

    # 获取并排序所有图像路径
    image_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])

    total = len(image_files)
    if total == 0:
        print("No images found.")
        return

    # 加载所有图像（统一尺寸）
    base_img = load_and_normalize_image(image_files[0])
    H, W = base_img.shape
    all_images = [load_and_normalize_image(p, size=(W, H)) for p in image_files]

    # 构建堆栈
    for i in range(total):
        stack = []
        for j in range(i - (stack_size - 1), i + 1):  # 包括当前图像
            if j < 0:
                img = all_images[0]  # 前面不足补第一张图
            else:
                img = all_images[j]
            stack.append(img)
        stack_array = np.stack(stack, axis=0)  
        save_path = os.path.join(output_folder, os.path.basename(image_files[i]).replace('.png','.mat'))
        savemat(save_path, {'data': stack_array})
        print(f"Saved: {save_path}")
import os

def get_immediate_subfolders(folder):
    return [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, name))
    ]

subfolders = get_immediate_subfolders('dataset/IRDST')
for folder in subfolders:
    if os.path.exists(os.path.join(folder,'Mix')):
        shutil.rmtree(os.path.join(folder,'Mix'))
    if os.path.exists(os.path.join(folder,'Mix_masks')):
        shutil.rmtree(os.path.join(folder,'Mix_masks'))
    if os.path.exists(os.path.join(folder,'Mix_masks_centroid')):
        shutil.rmtree(os.path.join(folder,'Mix_masks_centroid'))
    create_right_aligned_stacks(folder_path=os.path.join(folder,'images'),output_folder=os.path.join(folder,'Mix'))
    create_right_aligned_stacks(folder_path=os.path.join(folder,'masks'),output_folder=os.path.join(folder,'Mix_masks'))
    # create_right_aligned_stacks(folder_path=os.path.join(folder,'masks_centroid'),output_folder=os.path.join(folder,'Mix_masks_centroid'))

