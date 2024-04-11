import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class RNADataset(Dataset):
    def __init__(self, hparams, mode='train'):  
        path = Path(hparams['data']['data_path'])

        self.gt_images, self.range = images_to_tensor(path)
        self.dapi_images = read_dapi_image(path)

        self.color_bias = hparams['train']['color_bias']

        self.gt_images = self.gt_images * (1 - self.color_bias * 2) + self.color_bias

        self.num_iters = hparams['train']['iterations']
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
        return self.gt_images.shape


def images_to_tensor(image_path: Path):
    image_paths = [image_path / f'F1R{r}Ch{c}.tif' for r in range(1, 6) for c in range(2, 5)]

    images = []
    transform = transforms.ToTensor()

    for image_path in image_paths:
        img = Image.open(image_path)
        img_tensor = transform(img).permute(1, 2, 0)[..., :3] #[h,w,1]
        images.append(img_tensor)

    imgs_tensor = torch.cat(images, dim=2) / 1.0 # [h, w, 15]
    imgs_tensor = torch.log10(imgs_tensor + 1)

    all_pixels = imgs_tensor[..., :-1].reshape(-1)
    top_min_values = torch.topk(-all_pixels, int(0.001 * all_pixels.shape[0])).values

    min_value = -top_min_values.min()
    max_value = all_pixels.max()

    imgs_tensor = torch.relu(imgs_tensor - min_value) / (max_value - min_value)

    # min_value, max_value = imgs_tensor.min(), imgs_tensor.max()
    imgs_tensor = (imgs_tensor - min_value) / (max_value - min_value)

    return imgs_tensor ** 0.5, (0, max_value)


# def images_to_tensor_cropped(image_path: Path):
#     image_paths = [image_path / f'{i}.png' for i in range(1, 16)]

#     images = []

#     for image_path in image_paths:
#         img = Image.open(image_path)
#         transform = transforms.ToTensor()
#         img_tensor = transform(img).permute(1, 2, 0)[..., :3] #[h,w,1]
#         images.append(img_tensor)

#     imgs_tensor = torch.cat(images, dim=2) / 255. # [h, w, 15]
#     min_value, max_value = imgs_tensor.min(), imgs_tensor.max()
#     imgs_tensor = (imgs_tensor - min_value) / (max_value - min_value)

#     return imgs_tensor, (min_value, max_value)


def read_dapi_image(image_path: Path):
    image_paths = [image_path / f'F1R{r}Ch1.tif' for r in range(1, 6)]

    images = []
    transform = transforms.ToTensor()

    for image_path in image_paths:
        img = Image.open(image_path)
        img_tensor = transform(img).permute(1, 2, 0)[..., :3] #[h,w,1]
        images.append(img_tensor)

    imgs_tensor = torch.cat(images, dim=2) / 1.0 # [h, w, 15]
    imgs_tensor = torch.log10(imgs_tensor + 1) # / torch.log10(torch.tensor([2801])) # [h,w,15]
    # imgs_tensor = torch.clamp(imgs_tensor, 0, 1)

    min_value, max_value = imgs_tensor.min(), imgs_tensor.max()
    imgs_tensor = (imgs_tensor - min_value) / (max_value - min_value)

    return imgs_tensor


# draw histogram of the image, image: [h, w]
def draw_histogram(image, name):
    # remove zero values, 100 bins, title is the percentage of the non-zero values
    pos_image = image[image > 0]
    # plt.hist(pos_image.ravel(), 1000, [0, 1])
    plt.hist(image.ravel(), 100, [0, 3])
    plt.title(f'P: {(pos_image.ravel().shape[0] / image.ravel().shape[0]) * 100:.2f}% mean: {pos_image.mean():.2f} std: {pos_image.std():.2f}' + \
              f' O: mean: {image.mean():.2f} std: {image.std():.2f}')

    plt.savefig(name)


class RNARawDataset(Dataset):
    def __init__(self, hparams, mode='train'):  
        path = Path(hparams['data']['data_path'])

        self.gt_images, self.range = read_raw_image(path)

        self.num_iters = hparams['train']['iterations']
        self.mode = mode


def read_raw_image(image_path: Path):
    image_paths = [image_path / f'F2R{r}Ch{c}.png' for r in range(1, 6) for c in range(2, 5)]

    images = []
    transform = transforms.ToTensor()

    for image_path in image_paths:
        img = Image.open(image_path)
        img_tensor = transform(img).permute(1, 2, 0)[..., :3] #[h,w,1]
        images.append(img_tensor)

    imgs_tensor = torch.cat(images, dim=2) / 1.0 # [h, w, 15]
    imgs_tensor = torch.log10(imgs_tensor + 1)
    
    # / torch.log10(torch.tensor([2000])) # [h,w,15]

    # imgs_tensor = torch.clamp(imgs_tensor, 0, 1)
    # imgs_tensor = torch.cat(images, dim=2) / 255. # [h, w, 15]

    # min_value, max_value = imgs_tensor.min(), imgs_tensor.max()
    # imgs_tensor = (imgs_tensor - min_value) / (max_value - min_value)

    return imgs_tensor ** 0.5, (0, 65536)


if __name__ == "__main__":
    val_dataset = RNARawDataset(
        hparams={
            'data': {'data_path': '/home/luzhan/Projects/rna/data/IM41340_full/IM41340_raw'}, 
            'train': {'iterations': 1},
        }, 
        mode='val',
    )

    image_idx = 0
    print(val_dataset.gt_images[..., image_idx].numpy().mean())

    draw_histogram(val_dataset.gt_images[..., image_idx].numpy(), name=f'histogram_{image_idx}.png')