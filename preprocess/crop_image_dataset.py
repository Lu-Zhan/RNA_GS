# crop image from the original image by input param '--hw 512 512', the crop window is centeralized in the original image, image is stored in a format of .png and int16, create new folder to store the cropped images
import glob
import os
import cv2
import shutil
import argparse
import numpy as np

from skimage import io



def crop_images_by_tile_number(img_folder, dst_folder, i, j, tile_number):
    images = glob.glob(os.path.join(img_folder, "*.tif"))

    for path in images:
        # img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = io.imread(path)

        h, w = img.shape
        h_start = h // tile_number * i
        w_start = w // tile_number * j
        img = img[h_start:h_start+h//tile_number, w_start:w_start+w//tile_number]

        new_folder_name = j + i * tile_number

        output_path = os.path.join(dst_folder, '1', os.path.basename(path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        io.imsave(output_path, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--hw", type=int, nargs=2, default=[2304, 2304])
    parser.add_argument("--n_tile", type=int, default=6)
    args = parser.parse_args()

    src_folder = args.src
    hw = args.hw
    tile_number = args.n_tile
    
    for i in range(tile_number):
        for j in range(tile_number):
            new_folder_name = j + i * tile_number
            dst_folder = os.path.join(src_folder, f'cropped/tile_{tile_number ** 2}/{new_folder_name}')
            
            for idx in range(5):
                round = f'R{idx+1}'

                img_folder = os.path.join(src_folder, round, '1')
                dst_round_folder = os.path.join(dst_folder, round)
                crop_images_by_tile_number(img_folder, dst_round_folder, i, j, tile_number)

                # crop dapi image
                dapi_image_path = os.path.join(src_folder, f'dapi_image/{tile_number}/{new_folder_name}_{tile_number ** 2}', f'F1R{idx+1}Ch1.tif')
                save_image_path = os.path.join(dst_folder, f'dapi_image', f'F1R{idx}Ch1.tif')

                os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
                shutil.copy(dapi_image_path, save_image_path)
            
            if j + i * tile_number == 1:
                break
        break