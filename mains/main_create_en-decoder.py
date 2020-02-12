from pathlib import Path

from pipeline.data_gatherers.coco_grabber import create_coco_dataset
from pipeline.networks.pix2pix import Pix2Pix
import settings
import os

from pipeline.processors.preprocessors import dataset_normalizer

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # to train on cpu
    data_dir = os.path.join(settings.dataset_dir, "pix2pix", "maps", "train")
    processed_data_dir = os.path.join(Path(data_dir).parent, "train_processed")

    pix2pix = Pix2Pix(image_shape=(128, 128, 3), light_w=False)

    dataset_normalizer.process(input_dir=data_dir, output_dir=processed_data_dir)
    pix2pix.train(epochs=100000, batch_size=3, sample_interval=100, data_dir=processed_data_dir, load_last_chkpt=False)
