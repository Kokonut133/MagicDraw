from pipeline.data_gatherers.coco_grabber import create_coco_dataset
from pipeline.networks.pix2pix import Pix2Pix
import settings

if __name__ == '__main__':
    data_dir = "/home/cstur/projects/frame2frame/resources/datasets/pix2pix/cityscapes/train/"

    pix2pix = Pix2Pix(image_shape=(256, 256, 3))
    pix2pix.train(epochs=200, batch_size=5, sample_interval=200, data_dir=data_dir)
