import os
import numpy as np
import scipy.io as scio
from numpy import *
from wrapper import CacheDataset,cache_read_img
class SeqSource(CacheDataset):
    def __init__(
        self,
        root,
        imgs_arr,
        img_size=(256,256),
        frame_num=5,
        use_cache=True,
        cache_type="ram",
    ):
        self.frame_num = frame_num
        self.root = root
        self.imgs_arr = imgs_arr
        self.cache = use_cache
        self.cache_type = cache_type
        self.num_imgs = len(self.imgs_arr)
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            cache=use_cache,
            cache_type=cache_type
        )
    def __len__(self):
        return self.num_imgs
    @cache_read_img(use_cache=True)
    def read_img(self, index):
        file_path = os.path.join(self.root,self.imgs_arr[index])
        MixData_mat = scio.loadmat(file_path)
        MixData_Img = MixData_mat.get('data')
        MixData_Img = MixData_Img.astype(np.float32)
        MixData = MixData_Img[-self.frame_num:,:,:]
        MixData_out = np.expand_dims(MixData, 0)
        return MixData_out
    def _input_dim(self):
        return self.img_size
    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        return self.read_img(index)

if __name__ == '__main__':
    def are_all_images_same(imgs):
        """
        判断 imgs 中所有图片是否相同
        
        参数:
            imgs: numpy.ndarray, 形状为 (b, h, w) 的图片数组
            
        返回:
            bool: 如果所有图片相同返回 True，否则返回 False
        """
        if len(imgs) == 0:
            return True  # 空数组视为所有图片相同
        
        first_img = imgs[0]
        for img in imgs[1:]:
            if not np.array_equal(first_img, img):
                return False
        return True
    root = '/home/greek/files/Video_structure/dataset/NUDT-MIRSDT'
    txtpath = os.path.join(root , 'train.txt')
    txts = np.loadtxt(txtpath, dtype=bytes).astype(str)
    img_source = SeqSource(root=root,imgs_arr=txts,use_cache=True,cache_type="ram")
    import cv2
    for source in img_source:
        print(are_all_images_same(source[0]))
        for img in source[0]:
            cv2.imshow("test",img/255.)
            cv2.waitKey(10)
        break
        