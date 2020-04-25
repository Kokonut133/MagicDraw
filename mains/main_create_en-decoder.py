from pathlib import Path

from pipeline.hardware_handler.gpu_handler import list_available_gpus
from pipeline.networks.pix2pix import Pix2Pix
import os
from tensorflow.python.client import device_lib

import settings

from pipeline.processors.preprocessors import dataset_normalizer

__author__="cstur"


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # Titan X Pascal
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # GTX 980 Ti
    # os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # CPU
    # running Generator on Titan and Discriminator on 980 Ti is about 30-50% slower than CUDA_visibility to Titan only
    # visibility to all GPUs results in the same performance as forcing CUDA_visibility to Titan only
    list_available_gpus()

    data_dir = os.path.join(settings.dataset_dir, "pix2pix", "maps", "train")
    processed_data_dir = os.path.join(Path(data_dir).parent, "train_processed")
    dataset_normalizer.convert_to_tif(input_dir=data_dir, output_dir=processed_data_dir)    # kicked out normalization

    # batch size 1 is recommended in the paper
    pix2pix = Pix2Pix(image_shape=(128, 128, 3), gpu_memory_friendly=True)
    pix2pix.train(epochs=1000000, batch_size=1, log_interval=100, sample_interval=1000, data_dir=processed_data_dir, generate_right=False, load_last_chkpt=True)
