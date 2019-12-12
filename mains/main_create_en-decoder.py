from pipeline.data_gatherers.coco_grabber import getCats
from pipeline.networks.pix2pix import Pix2Pix

if __name__ == '__main__':
    getCats()

    pix2pix = Pix2Pix(image_shape=(256, 256, 3))
    pix2pix.train(epochs=200, batch_size=1, sample_interval=200)
