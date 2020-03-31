from pathlib import Path
from pipeline.networks.pix2pix import Pix2Pix
import os

import settings

from pipeline.processors.preprocessors import dataset_normalizer

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # to train on cpu
    data_dir = os.path.join(settings.dataset_dir, "pix2pix", "facades", "train")
    processed_data_dir = os.path.join(Path(data_dir).parent, "train_processed")

    # 100 epochs 64x64x3 loss for D and G gpu_memory_friendly=True->9 sec; =False->9sec
    pix2pix = Pix2Pix(image_shape=(64, 64, 3), gpu_memory_friendly=True)

    dataset_normalizer.process(input_dir=data_dir, output_dir=processed_data_dir)
    # batch size 1 is recommended in the paper; maybe abstraction is already too hard for this structure?
    pix2pix.train(epochs=100000, batch_size=1, sample_interval=100, data_dir=processed_data_dir, generate_right=False, load_last_chkpt=False)
