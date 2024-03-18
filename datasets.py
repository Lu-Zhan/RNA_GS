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

        try:
            self.gt_images, self.range = images_to_tensor(path)
            self.dapi_images = read_dapi_image(path)
        except:
            self.gt_images, self.range = images_to_tensor_cropped(path)
        
        # self.gt_image = torch.repeat_interleave(self.gt_image[..., 0:1], 15, dim=2)

        self.gt_images = self.gt_images ** 0.5

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
    image_paths = [image_path / f'F1R{r}Ch{c}.png' for r in range(1, 6) for c in range(2, 5)]

    images = []
    transform = transforms.ToTensor()

    for image_path in image_paths:
        img = Image.open(image_path)
        img_tensor = transform(img).permute(1, 2, 0)[..., :3] #[h,w,1]
        images.append(img_tensor)

    imgs_tensor = torch.cat(images, dim=2) / 1.0 # [h, w, 15]
    imgs_tensor = torch.log10(imgs_tensor + 1) / torch.log10(torch.tensor([2801])) # [h,w,15]
    # imgs_tensor = torch.clamp(imgs_tensor, 0, 1)
    # imgs_tensor = torch.cat(images, dim=2) / 255. # [h, w, 15]

    min_value, max_value = imgs_tensor.min(), imgs_tensor.max()
    imgs_tensor = (imgs_tensor - min_value) / (max_value - min_value)

    return imgs_tensor, (min_value, max_value)


def images_to_tensor_cropped(image_path: Path):
    image_paths = [image_path / f'{i}.png' for i in range(1, 16)]

    images = []

    for image_path in image_paths:
        img = Image.open(image_path)
        transform = transforms.ToTensor()
        img_tensor = transform(img).permute(1, 2, 0)[..., :3] #[h,w,1]
        images.append(img_tensor)

    imgs_tensor = torch.cat(images, dim=2) / 255. # [h, w, 15]
    min_value, max_value = imgs_tensor.min(), imgs_tensor.max()
    imgs_tensor = (imgs_tensor - min_value) / (max_value - min_value)

    return imgs_tensor, (min_value, max_value)


def read_dapi_image(image_path: Path):
    image_paths = [image_path / f'F1R{r}Ch1.png' for r in range(1, 6)]

    images = []
    transform = transforms.ToTensor()

    for image_path in image_paths:
        img = Image.open(image_path)
        img_tensor = transform(img).permute(1, 2, 0)[..., :3] #[h,w,1]
        images.append(img_tensor)

    imgs_tensor = torch.cat(images, dim=2) / 1.0 # [h, w, 15]
    imgs_tensor = torch.log10(imgs_tensor + 1) / torch.log10(torch.tensor([2801])) # [h,w,15]
    imgs_tensor = torch.clamp(imgs_tensor, 0, 1)

    # min_value, max_value = imgs_tensor.min(), imgs_tensor.max()
    # imgs_tensor = (imgs_tensor - min_value) / (max_value - min_value)

    return imgs_tensor


# draw histogram of the image, image: [h, w]
def draw_histogram(image, name):
    # remove zero values, 100 bins, title is the percentage of the non-zero values
    pos_image = image[image != 0]
    plt.hist(pos_image.ravel(), 100, [0, 1])
    plt.title(f'P: {(pos_image.ravel().shape[0] / image.ravel().shape[0]) * 100:.2f}% mean: {pos_image.mean():.2f} std: {pos_image.std():.2f}' + \
              f' O: mean: {image.mean():.2f} std: {image.std():.2f}')

    plt.savefig(name)


if __name__ == "__main__":
    val_dataset = RNADataset(
        hparams={
            'data': {'data_path': '../data/IM41340_tiled/1_36'}, 
            'train': {'iterations': 1},
        }, 
        mode='val',
    )

    draw_histogram(val_dataset.gt_images[..., 0].numpy(), name='histogram_ori.png')