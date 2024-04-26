import os
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset

from datasets.read_images import read_images_single_round, read_images_multi_round, read_dapi_image_outside


class RNADataset3D(Dataset):
    def __init__(self, hparams, mode='train'):
        self.color_bias = hparams['train']['color_bias']
        self.data_dir = hparams['data']['data_path']
        self.num_cams = len(hparams['camera']['cam_ids'])
        self.num_dims = hparams['model']['num_dims']

        # [(k, h, w, c)], (vmax, vmin)
        self.gt_images, self.cam_indexs, self.slice_indexs, self.range = read_images_multi_round(image_folder=self.data_dir)
        self.dapi_images = read_dapi_image_outside(image_folder=os.path.join(self.data_dir, 'dapi_image')) # (m, h, w, 1)
        # self.dapi_images = torch.relu(self.dapi_images - self.range[0]) / (self.range[1] - self.range[0])

        self.gt_images = self.gt_images[..., :hparams['model']['num_dims']] * (1 - self.color_bias * 2) + self.color_bias 
        self.mode = mode

        # select cameras
        self.select_camera(cam_ids=hparams['camera']['cam_ids'])
        # num_slices = [n0, n1, n2, ...]
        self.num_slices = [self.cam_indexs.count(idx) for idx in hparams['camera']['cam_ids']]

    def __len__(self):
        if self.mode == 'train':
            return 6000 * self.num_cams * self.num_dims
        else:
            return self.gt_images.shape[0]
        
    def __getitem__(self, index):
        index = index % self.gt_images.shape[0]
        return self.gt_images[index], self.cam_indexs[index], self.slice_indexs[index]    # (h, w, c)

    @property
    def size(self):
        return self.gt_images.shape[1:]
    
    def select_camera(self, cam_ids):
        selected_index = [i for i, x in enumerate(self.cam_indexs) if x in cam_ids]
        self.gt_images = self.gt_images[selected_index]
        self.cam_indexs = [self.cam_indexs[i] for i in selected_index]
        self.slice_indexs = [self.slice_indexs[i] for i in selected_index]
    

class RNADataset3DPerCam(RNADataset3D):
    def __init__(self, hparams, mode='train'):
        super(RNADataset3DPerCam, self).__init__(hparams, mode)
        self.cam_indexs = torch.tensor(self.cam_indexs, dtype=torch.long)
        self.slice_indexs = torch.tensor(self.slice_indexs, dtype=torch.long)
        self.batch_size = hparams['train']['batch_size']
        self.create_cam_dict(cam_ids=hparams['camera']['cam_ids'])

    def create_cam_dict(self, cam_ids):
        self.cam_dict = {
            cam_id: self.slice_indexs[self.cam_indexs == cam_id] for cam_id in cam_ids
        }

    def __len__(self):
        if self.mode == 'train':
            return 6000 // self.batch_size * self.num_cams * self.num_dims
        else:
            return self.gt_images.shape[0]
    
    def __getitem__(self, index):
        index = index % self.gt_images.shape[0]

        if self.mode == 'train':
            cam_id = int(self.cam_indexs[index])
            slices = self.cam_dict[cam_id]  # [0, 1, ..., m-1]

            rand_slice_idx = torch.randint(0, len(slices), (self.batch_size,))
            rand_idx = self.num_slices[cam_id] - self.num_slices[0] + slices[rand_slice_idx]

            # (b, h, w, c), (b), (b)
            return self.gt_images[rand_idx], self.cam_indexs[rand_idx], self.slice_indexs[rand_idx]    # (h, w, c)
        else:
            return self.gt_images[index], self.cam_indexs[index], self.slice_indexs[index]    # (h, w, c)


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