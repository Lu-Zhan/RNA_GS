# crop image from the original image by input param '--hw 512 512', the crop window is centeralized in the original image, image is stored in a format of .png and int16, create new folder to store the cropped images
import glob
import os
import cv2
import argparse
import numpy as np


def crop_images(src_folder, hw):
    images = glob.glob(os.path.join(src_folder, "*.png"))
    print(f"Found {len(images)} images")

    for path in images:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        h, w = img.shape
        h_start = (h - hw[0]) // 3
        w_start = (w - hw[1]) // 3
        img = img[h_start:h_start+hw[0], w_start:w_start+hw[1]]

        new_folder_name = str(hw[0]) + "x" + str(hw[1])

        output_path = os.path.join(src_folder, new_folder_name, os.path.basename(path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cv2.imwrite(output_path, img)


def crop_images_by_tile_number(src_folder, tile_number):
    images = glob.glob(os.path.join(src_folder, "*.png"))
    print(f"Found {len(images)} images")

    for i in range(tile_number):
        for j in range(tile_number):
            for path in images:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

                h, w = img.shape
                h_start = h // tile_number * i
                w_start = w // tile_number * j
                img = img[h_start:h_start+h//tile_number, w_start:w_start+w//tile_number]

                new_folder_name = j + i * tile_number

                output_path = os.path.join(src_folder, f'{new_folder_name}_{tile_number * tile_number}', os.path.basename(path))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                cv2.imwrite(output_path, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--hw", type=int, nargs=2, required=True)
    parser.add_argument("--tile_number", type=int, default=4)
    args = parser.parse_args()

    src_folder = args.src
    hw = args.hw
    tile_number = args.tile_number

    # crop_images(src_folder, hw)
    # print(f"Image cropped and saved to {args.src} / {args.hw}")

    crop_images_by_tile_number(src_folder, tile_number)
    print(f"Image cropped and saved to {args.src} / {args.tile_number}")