import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from skimage import io
from glob import glob
from pathlib import Path
from torch.utils.data import Dataset


class RNADataset3D(Dataset):
    def __init__(self, hparams, mode='train'):  
        path = Path(hparams['data']['data_path'])

        self.gt_images, self.range = read_images(path)

        self.dapi_images =self.gt_images[..., :1]   # (n, h, w, 1)
        self.gt_images = self.gt_images[..., 1:]    # (n, h, w, c)

        self.color_bias = hparams['train']['color_bias']

        self.gt_images = self.gt_images * (1 - self.color_bias * 2) + self.color_bias

        # self.num_iters = hparams['train']['iterations']
        self.mode = mode
        
    def __len__(self):
        if self.mode == 'train':
            return self.num_iters
        else:
            return 1

    def __getitem__(self, index):
        return self.gt_images

    @property
    def size(self):
        return self.gt_images.shape[1:]
    

class RNADataset3DRand(RNADataset3D):
    def __len__(self):
        if self.mode == 'train':
            return 10000
        else:
            return self.gt_images.shape[0]

    def __getitem__(self, index):
        index = index % self.gt_images.shape[0]

        return self.gt_images[index], index    # (h, w, c)


def read_images(image_path: Path):
    image_paths = glob(str(image_path / Path('*.tif')))

    # sftp://10.39.21.61//home/luzhan/Projects/rna/data/IM41236/rawtiff/1_Z0046_C0004.tif
    # sort image_paths by the number in the file name
    image_paths = sorted(image_paths)

    image_dic = {f'C{i+1:04d}': [] for i in range(4)}
    image_dic = {k: [x for x in image_paths if k in x] for k in image_dic.keys()}

    images = {k: [] for k in image_dic.keys()}
    transform = transforms.ToTensor()

    for key in image_dic.keys():
        for image_path in image_dic[key]:
            img = io.imread(image_path)
            img = img.astype(np.float32)
            img_tensor = transform(img).permute(1, 2, 0)[None, ..., :1] # (1, h, w, 1)
        
            images[key].append(img_tensor)
        
        images[key] = torch.cat(images[key], dim=0) # (n, h, w, 1)
    
    imgs_tensor = torch.cat([images[f'C{i+1:04d}'] for i in range(4)], dim=-1) / 1.0 # (n, h, w, 4)
    # imgs_tensor[imgs_tensor < 10] = 0
    imgs_tensor = torch.log10(imgs_tensor + 1)

    dapi_tensor = imgs_tensor[..., :1]
    imgs_tensor = imgs_tensor[..., 1:2]  # (n, h, w, k)

    all_pixels = imgs_tensor.reshape(-1, imgs_tensor.shape[-1]) # (m, k)
    top_min_values = torch.topk(-all_pixels, int(0.01 * all_pixels.shape[0]), dim=0)[0] # (0.01 * m, k)
    min_values = - top_min_values.min(dim=0)[0].mean(dim=0)

    # min_values = imgs_tensor.min()
    max_values = imgs_tensor.max()

    imgs_tensor = torch.relu(imgs_tensor - min_values) / (max_values - min_values)
    dapi_tensor = torch.relu(dapi_tensor - min_values) / (max_values - min_values)

    return torch.cat((dapi_tensor, imgs_tensor), dim=-1), (min_values, max_values)

    # min_value = imgs_tensor.min()

    # # min_value = 0.01
    # max_value = imgs_tensor.mean() + 3 * imgs_tensor.std()
    # # max_value = imgs_tensor.max()

    # imgs_tensor = (torch.relu(torch.clamp(imgs_tensor, max=max_value) - min_value)) / (max_value - min_value)

    # return imgs_tensor ** 0.5, (min_value, max_value)
    

# draw histogram of the image, image: [h, w]
def draw_histogram(image, name):
    # remove zero values, 100 bins, title is the percentage of the non-zero values
    pos_image = image[image < 1.9]
    plt.hist(pos_image.ravel(), 100, [1.6, 1.95])
    plt.title(f'P: {(pos_image.ravel().shape[0] / image.ravel().shape[0]) * 100:.2f}% mean: {pos_image.mean():.2f} std: {pos_image.std():.2f}' + \
              f' O: mean: {image.mean():.2f} std: {image.std():.2f}')

    plt.savefig(name)


if __name__ == "__main__":
    images, _ = read_images('/home/luzhan/Projects/rna/data/rna_data_0407/IM41236/rawtiff')
    draw_histogram(images[..., 1:].numpy(), name=f'histogram.png')

    # val_dataset = RNA3DDataset(
    #     hparams={
    #         'data': {'data_path': '/home/luzhan/Projects/rna/data/IM41340_full/IM41340_raw'}, 
    #         'train': {'iterations': 1},
    #     }, 
    #     mode='val',
    # )

    # image_idx = 0
    # print(val_dataset.gt_images[..., image_idx].numpy().mean())

    # draw_histogram(val_dataset.gt_images[..., image_idx].numpy(), name=f'histogram_{image_idx}.png')