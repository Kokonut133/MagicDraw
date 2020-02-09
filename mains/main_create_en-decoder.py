from pipeline.data_gatherers.coco_grabber import create_coco_dataset
from pipeline.networks.pix2pix import Pix2Pix
import settings
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # to train on cpu
    data_dir = os.path.join(settings.dataset_dir, "pix2pix", "maps", "train")

    pix2pix = Pix2Pix(image_shape=(256, 256, 3), light_w=True)
    pix2pix.train(epochs=100000, batch_size=2, sample_interval=100, data_dir=data_dir, load_last_chkpt=False) # batch size 2 seems to fit still on GPU
