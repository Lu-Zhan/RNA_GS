import os
import torch

from pathlib import Path
from torch.utils.data import Dataset

from datasets.read_images import read_images_single_round, read_images_from_rounds, read_dapi_image_outside


class RNADataset3D(Dataset):
    def __init__(self, hparams, mode='train'):
        self.color_bias = hparams['train']['color_bias']
        self.data_dir = hparams['data']['data_path']
        self.num_cams = len(hparams['camera']['cam_ids'])
        self.num_dims = hparams['camera']['num_dims']
        self.mode = mode

        self.num_per_epoch = hparams['train']['num_per_epoch']

        # [(k, h, w, c)], (vmax, vmin)
        self.gt_images, self.cam_ids, self.slice_indexs, self.range = read_images_from_rounds(
            image_folder=self.data_dir,
            num_rounds=hparams['camera']['max_num_cams'],
        ) # (m, h, w, c)
        self.dapi_images = read_dapi_image_outside(image_folder=os.path.join(self.data_dir, 'dapi_image')) # (m, h, w, 1)
        
        # preprocessing
        self.gt_images = self.gt_images[..., :hparams['camera']['num_dims']] * (1 - self.color_bias * 2) + self.color_bias

        # select gt_images, cam_ids, slice_indexs, dapi_images by cam_ids
        self.select_camera(cam_ids=hparams['camera']['cam_ids'])

    def __len__(self):
        if self.mode == 'train':
            return self.num_per_epoch
        else:
            return self.gt_images.shape[0]
        
    def __getitem__(self, index):
        index = index % self.gt_images.shape[0]
        return self.gt_images[index], self.cam_ids[index], self.slice_indexs[index]    # (h, w, c)

    @property
    def size(self):
        return self.gt_images.shape[1:]
    
    def select_camera(self, cam_ids):
        selected_index = [i for i, x in enumerate(self.cam_ids) if x in cam_ids]
        self.gt_images = self.gt_images[selected_index]
        self.cam_ids = [self.cam_ids[i] for i in selected_index]
        self.slice_indexs = [self.slice_indexs[i] for i in selected_index]
    

class RNADataset3DSingleRound(Dataset):
    def __init__(self, hparams, mode='train'):  
        path = Path(hparams['data']['data_path'])

        self.gt_images, self.range = read_images_single_round(path)

        self.dapi_images =self.gt_images[..., :1]   # (n, h, w, 1)
        self.gt_images = self.gt_images[..., 1:1+hparams['model']['num_dims']]    # (n, h, w, c)

        self.color_bias = hparams['train']['color_bias']

        self.gt_images = self.gt_images * (1 - self.color_bias * 2) + self.color_bias

        self.mode = mode
        
    def __len__(self):
        if self.mode == 'train':
            return 100000
        else:
            return self.gt_images.shape[0]
        
    def __getitem__(self, index):
        index = index % self.gt_images.shape[0]
        return self.gt_images[index], index    # (h, w, c)

    @property
    def size(self):
        return self.gt_images.shape[1:]