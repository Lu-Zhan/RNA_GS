import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np

from PIL import Image
# 读取图像
path = 'data/IM41340regi'
path = Path(path)
image_paths = [path / f'F1R{r}Ch{c}.png' for r in range(1, 6) for c in range(2, 5)]
for image_path in image_paths:
    image = Image.open(image_path)

    # 展平图像数组，使其成为一维数组
    pixels = np.array(image).flatten()

    # 绘制直方图
    plt.hist(pixels, bins=1000, range=(1, 1000), color='blue', alpha=0.9)

    # 设置标题和标签
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Normalized Frequency')

    # 保存直方图为图像文件
    save_name = f'histogram_plot_{image_path.stem}.png'
    plt.savefig(save_name)

    # 显示直方图
    plt.show()
