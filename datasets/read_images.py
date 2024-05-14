import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from pathlib import Path
from skimage import io
from glob import glob


def read_images_from_rounds(image_folder, rounds=[0]):
    # /home/luzhan/Projects/rna/data/IM41236/IM41441/Rx/1_Z0046_C0004.tif

    round_images = []
    for round in rounds:
        round = f'R{round+1}'

        image_paths = glob(os.path.join(image_folder, round, '1', '*.tif'))
        image_paths = sorted(image_paths)

        if len(image_paths) == 0:
            raise ValueError(f'No images found in {os.path.join(image_folder, round, "1")}')

        image_dic = {f'C{i+2:04d}': [] for i in range(3)}
        image_dic = {k: [x for x in image_paths if k in x] for k in image_dic.keys()}

        images = {k: [] for k in image_dic.keys()}

        for key in image_dic.keys():
            for image_path in image_dic[key]:
                img = io.imread(image_path)
                img = img.astype(np.float32)
                if len(img.shape) == 2:
                    img = img[..., None]
                img_tensor = torch.tensor(img)[None, ..., :1] # (1, h, w, 1)
            
                images[key].append(img_tensor)
            
            images[key] = torch.cat(images[key], dim=0) # (n, h, w, 1)
    
        imgs_tensor = torch.cat([images[f'C{i+2:04d}'] for i in range(3)], dim=-1) / 1.0 # (n, h, w, 3)
        round_images.append(imgs_tensor)
    
    cam_indexs = [n for n, x in enumerate(round_images) for _ in range(x.shape[0])] # [0, ..., 0, 1, ...]
    slice_indexs = [i for x in round_images for i in range(x.shape[0])]   # [0, ..., n0, 0, ..., n1, ...]
    
    all_images = torch.cat(round_images, dim=0) # (5n, h, w, 3)
    all_images = torch.log10(all_images + 1)

    # all_pixels = all_images.reshape(-1, imgs_tensor.shape[-1]) # (m, k)
    # top_min_values = torch.topk(-all_pixels, int(0.01 * all_pixels.shape[0]), dim=0)[0] # (0.01 * m, k)
    # min_values = - top_min_values.min(dim=0)[0].mean(dim=0)

    min_values = all_images.min()
    max_values = all_images.max()

    all_images = torch.relu(all_images - min_values) / (max_values - min_values)
    
    return all_images, cam_indexs, slice_indexs, (min_values, max_values)


def read_dapi_image_outside(image_folder):
    # /home/luzhan/Projects/rna/data/rna_data_0407/IM41441/dapi_image/F1R1Ch1.tif
    image_paths = glob(os.path.join(image_folder, '*.tif'))
    image_paths = sorted(image_paths)

    image_paths = [x for x in image_paths if 'Ch1' in x]    # [5]

    images = [io.imread(x).astype(np.float32) for x in image_paths]
    images = [torch.tensor(x)[None, ..., None] for x in images]
    images = torch.cat(images, dim=0) / 1.0 # (m, h, w, 1)

    # images = torch.log10(images + 1)
    images = (images - images.min()) / (images.max() - images.min())

    return images


def read_images_single_round(image_path: Path):
    image_paths = glob(str(image_path / Path('*.tif')))

    #/home/luzhan/Projects/rna/data/IM41236/rawtiff/1_Z0046_C0004.tif
    # sort image_paths by the number in the file name
    image_paths = sorted(image_paths)

    image_dic = {f'C{i+1:04d}': [] for i in range(1)}
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
    
    imgs_tensor = torch.cat([images[f'C{i+1:04d}'] for i in range(1)], dim=-1) / 1.0 # (n, h, w, 4)
    # imgs_tensor[imgs_tensor < 10] = 0
    imgs_tensor = torch.log10(imgs_tensor + 1)

    dapi_tensor = imgs_tensor[..., :1]
    imgs_tensor = imgs_tensor[..., 1:]  # (n, h, w, k)

    all_pixels = imgs_tensor.reshape(-1, imgs_tensor.shape[-1]) # (m, k)
    top_min_values = torch.topk(-all_pixels, int(0.01 * all_pixels.shape[0]), dim=0)[0] # (0.01 * m, k)
    min_values = - top_min_values.min(dim=0)[0].mean(dim=0)

    # min_values = imgs_tensor.min()
    max_values = imgs_tensor.max()

    imgs_tensor = torch.relu(imgs_tensor - min_values) / (max_values - min_values)
    dapi_tensor = torch.relu(dapi_tensor - min_values) / (max_values - min_values)

    return torch.cat((dapi_tensor, imgs_tensor), dim=-1), (min_values, max_values)


# draw histogram of the image, image: [h, w]
def draw_histogram(image, name):
    # remove zero values, 100 bins, title is the percentage of the non-zero values
    pos_image = image[image < 1.9]
    plt.hist(pos_image.ravel(), 100, [1.6, 1.95])
    plt.title(f'P: {(pos_image.ravel().shape[0] / image.ravel().shape[0]) * 100:.2f}% mean: {pos_image.mean():.2f} std: {pos_image.std():.2f}' + \
              f' O: mean: {image.mean():.2f} std: {image.std():.2f}')

    plt.savefig(name)


if __name__ == "__main__":
    images, _ = read_images_single_round('/home/luzhan/Projects/rna/data/rna_data_0407/IM41236/rawtiff')
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