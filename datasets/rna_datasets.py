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
        self.batch_size = hparams['train']['batch_size'] if mode == 'train' else 1
        self.mode = mode

        self.num_per_epoch = hparams['train']['num_per_epoch']

        # [(k, h, w, c)], (vmax, vmin)
        self.gt_images, self.cam_ids, self.slice_indexs, self.range = read_images_from_rounds(
            image_folder=self.data_dir,
            num_rounds=hparams['camera']['max_num_cams'],
        ) # (m, h, w, c)
        self.cam_ids = torch.tensor(self.cam_ids, dtype=torch.long)
        self.slice_indexs = torch.tensor(self.slice_indexs, dtype=torch.long)
        self.dapi_images = read_dapi_image_outside(image_folder=os.path.join(self.data_dir, 'dapi_image')) # (m, h, w, 1)
        
        # preprocessing
        self.gt_images = self.gt_images[..., :hparams['camera']['num_dims']] * (1 - self.color_bias * 2) + self.color_bias
        self.reg_masks = self.get_valid_reg_mask(num_rounds=hparams['camera']['max_num_cams'])

        # select gt_images, cam_ids, slice_indexs, dapi_images by cam_ids
        self.select_camera(cam_ids=hparams['camera']['cam_ids'])
        self.count = 0
        
    def __len__(self):
        if self.mode == 'train':
            return self.num_per_epoch
        else:
            return self.gt_images.shape[0]
        
    def __getitem__(self, index):
        if self.count == 0:
            self.cam_id = self.cam_ids[index % self.gt_images.shape[0]]

        self.count = (self.count + 1) % self.batch_size
        
        select_cam_index = self.cam_ids == self.cam_id
        select_sample_index = index % select_cam_index.sum()
        return self.gt_images[select_cam_index][select_sample_index], \
            self.cam_ids[select_cam_index][select_sample_index], \
            self.slice_indexs[select_cam_index][select_sample_index]    # (h, w, c)

    @property
    def size(self):
        return self.gt_images.shape[1:]
    
    def select_camera(self, cam_ids):
        selected_index = [i for i, x in enumerate(self.cam_ids) if x in cam_ids]
        self.gt_images = self.gt_images[selected_index]
        self.cam_ids = self.cam_ids[selected_index]
        self.slice_indexs = self.slice_indexs[selected_index]
    
    def get_valid_reg_mask(self, num_rounds, th=0.71):
        r0 = self.gt_images[self.cam_ids == 0].max(dim=0)[0]    # (h, w, c)

        masks = [torch.ones_like(r0)]
        for i in range(1, num_rounds):
            r = self.gt_images[self.cam_ids == i].max(dim=0)[0]    # (h, w, c)
            mask = (r0 > th) & (r > th)
            masks.append(mask)

            import numpy as np
            from PIL import Image
            Image.fromarray(mask.numpy().astype(np.uint8) * 255).save(f'mask_{i}.png')
        
        return torch.stack(masks, dim=0)    # (k, h, w, c)
    

class RNADatasetReg(RNADataset3D):
    def __getitem__(self, index):
        # index = index % self.gt_images.shape[0]
        # return self.gt_images[index], self.cam_ids[index], self.slice_indexs[index], self.reg_masks[self.cam_ids[index]]   # (h, w, c)
        if self.count == 0:
            self.cam_id = self.cam_ids[index % self.gt_images.shape[0]]

        self.count = (self.count + 1) % self.batch_size
        
        select_cam_index = self.cam_ids == self.cam_id
        select_sample_index = index % select_cam_index.sum()
        return self.gt_images[select_cam_index][select_sample_index], \
            self.cam_ids[select_cam_index][select_sample_index], \
            self.slice_indexs[select_cam_index][select_sample_index], \
            self.reg_masks[self.cam_id]    # (h, w, c)

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