from pipeline.networks.pix2pix import Pix2Pix



if __name__ == '__main__':
    pix2pix = Pix2Pix()
    pix2pix.train(epochs=200, batch_size=1, sample_interval=200)
