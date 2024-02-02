import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# (zwx)
def preprocess_data(dir_path: Path):
    image_paths = [dir_path / f"{i}.png" for i in range(1, 16)]
    ths = 13  # 更改二值化阈值
    pp = []
    for full_path in image_paths:
        img = cv2.imread(str(full_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((2, 2), dtype=np.uint8)
        img_normalized = cv2.convertScaleAbs(img)
        usm = cv2.erode(img_normalized, kernel, iterations=1)
        # (gzx):用腐蚀之后的图像
        thresh = cv2.threshold(usm, ths, 255, cv2.THRESH_BINARY)[1]

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh.copy(), connectivity=8, ltype=cv2.CV_32S
        )
        # 加入连通块外接矩阵几何中心
        for i in range(1, num_labels):  # 从1开始，跳过背景区域
            x, y, width, height, area = stats[i]
            centre_x = np.int64(x + width // 2)
            centre_y = np.int64(y + height // 2)
            pp.append([centre_x, centre_y])
        for pi in centroids[1:]:
            px, py = np.int64(pi[0]), np.int64(pi[1])
            pp.append([px, py])
    pp = np.array(pp)
    # 去重
    unique_pp = np.unique(pp, axis=0)
    return len(unique_pp), unique_pp


def give_required_data(input_coords, image_size, image_array, device):
    # normalising pixel coordinates [-1,1]
    coords = torch.tensor(
        input_coords / [image_size[0], image_size[1]], device=device
    ).float()  # , dtype=torch.float32
    center_coords_normalized = torch.tensor(
        [0.5, 0.5], device=device
    ).float()  # , dtype=torch.float32
    coords = - (center_coords_normalized - coords) * 2.0
    #(gzx):
    # center_coords_normalized = torch.tensor(
    #     [0, 0], device=device
    # ).float()  # , dtype=torch.float32
    # coords = - (coords - center_coords_normalized)
    # Fetching the colour of the pixels in each coordinates
    colour_values = [image_array[coord[1], coord[0]] for coord in input_coords]
    colour_values_on_cpu = [
        tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor
        for tensor in colour_values
    ]
    colour_values_np = np.array(colour_values_on_cpu)
    colour_values_tensor = torch.tensor(
        colour_values_np, device=device
    ).float()  # , dtype=torch.float32

    return colour_values_tensor, coords


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]

    # if img_tensor.shape[-1] == 1:
    #     img_tensor = img_tensor.repeat(1, 1, 3)

    return img_tensor


def images_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    try:
        image_paths = [image_path / f'F1R{r}Ch{c}.png' for r in range(1, 6) for c in range(2, 5)]
    except:
        image_paths = [image_path / f'{i}.png' for i in range(1, 16)]
        

    images = []

    for image_path in image_paths:
        img = Image.open(image_path)
        transform = transforms.ToTensor()
        img_tensor = transform(img).permute(1, 2, 0)[..., :3] #[h,w,1]
        images.append(img_tensor)

    imgs_tensor = torch.cat(images, dim=2) / 1.0 # [h, w, 15]
    print("Max image value:", imgs_tensor.max())
    imgs_tensor = torch.log10(imgs_tensor + 1) / torch.log10(torch.tensor([2801])) # [h,w,15]
    imgs_tensor = torch.clamp(imgs_tensor, 0, 1)

    return imgs_tensor


if __name__ == '__main__':
    image = images_to_tensor(Path('../data/IM41340_0124'))
    print('done')