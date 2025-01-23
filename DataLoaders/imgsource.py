from wrapper import CacheDataset,cache_read_img
import os
import numpy as np
from PIL import Image
class ImgSource(CacheDataset):
    def __init__(
        self,
        root,
        imgs_arr,
        cache=False,
        cache_type="ram",
    ):
        
        self.root = root
        self.imgs_arr = imgs_arr
        self.cache = cache
        self.cache_type = cache_type
        self.num_imgs = len(self.imgs_arr)
        super().__init__(
            input_dimension=(512,512),
            num_imgs=self.num_imgs,
            cache=cache,
            cache_type=cache_type
        )
    def __len__(self):
        return self.num_imgs
    @cache_read_img(use_cache=True)
    def read_img(self, index):
        file_path = os.path.join(self.root,self.imgs_arr[index])
        img = Image.open(file_path)
        img = np.array(img, dtype=np.float32)
        return img
    def __len__(self):
        return len(self.imgs_arr)
    def _input_dim(self):
        return self.img_size
    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        return self.read_img(index)

if __name__ == '__main__':
    root = '/home/greek/files/Video_structure/dataset/NUDT-MIRSDT'
    txtpath = os.path.join(root , 'train.txt')
    txts = np.loadtxt(txtpath, dtype=bytes).astype(str)
    txts = [txt.replace('.mat', '.png') for txt in txts]
    txts = [txt.replace('Mix', 'masks') for txt in txts]
    img_source = ImgSource(root=root,imgs_arr=txts,img_size=(512,512),cache=True,cache_type="ram")
    