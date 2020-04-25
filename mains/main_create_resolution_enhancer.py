__author__="cstur"

import os
import numpy as np
import tifffile
import scipy
from PIL import Image

import settings
from pipeline.networks.super_resolution import Super_resolution

if __name__ == '__main__':
    input_dir=os.path.join(settings.dataset_dir, "pix2pix", "maps", "train")
    processed_data_set_dir=os.path.join(settings.dataset_dir, "super_resolution", "pix2pix", "maps")
    os.makedirs(processed_data_set_dir, exist_ok=True)

    input_dim=(128, 128, 3)
    output_dim=(512, 512, 3)
    SR_model=Super_resolution(input_dim=input_dim, output_dim=output_dim)

    min_w = 1000
    min_h = 1000
    for input_file in os.listdir(input_dir):
        img = np.asarray(Image.open(os.path.join(input_dir, input_file))).astype("float32")
        h, w, _ = img.shape
        half_w = int(w / 2)

        left_part = img[:, :half_w, :]
        right_part = img[:, half_w:, :]

        right_pic = np.interp(right_part, (0, 255), (0, 1))

        if right_pic.shape[0] < min_w:
            min_w= right_pic.shape[0]
        if right_pic.shape[1] < min_h:
            min_h= right_pic.shape[1]

        tifffile.imsave(os.path.join(processed_data_set_dir, input_file), right_pic)

    if min_w<output_dim[0] or min_h<output_dim[1]:
        resolution_enhancer = Super_resolution(input_dim=input_dim, output_dim=(min_w, min_h, 3))
    else:
        resolution_enhancer = Super_resolution(input_dim=input_dim, output_dim=output_dim)

