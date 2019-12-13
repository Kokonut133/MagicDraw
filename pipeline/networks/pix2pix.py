import datetime
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy
from keras.layers import BatchNormalization
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model

import settings


class Pix2Pix:
    def __init__(self, image_shape):
        self.img_shape = image_shape
        if len(self.img_shape) != 3:
            print("Image should be w, h, c but is " + str(self.img_shape))

        patchsize = int(self.img_shape[0]/16)   # 16 parts of the image are evaluated if real
        self.disc_patch = (patchsize, patchsize, 1)

        self.discriminator = self.build_discriminator(input_shape=self.img_shape)
        self.discriminator.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
        self.generator = self.build_generator(input_shape=self.img_shape)

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        fake_A = self.generator(img_B)

        valid = self.discriminator([fake_A, img_B])
        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=["mse", "mae"], loss_weights=[1, 100], optimizer="adam")

    def build_generator(self, input_shape):
        def conv2d(input, filters, batch_norm, k_size=4):
            d = Conv2D(filters, kernel_size=k_size, strides=2, padding="same")(input)
            d = LeakyReLU(alpha=0.2)(d)
            if batch_norm:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(input, filters, skip_input, k_size=4):
            u = UpSampling2D(size=2)(input)
            u = Conv2D(filters, kernel_size=k_size, strides=1, padding="same", activation="relu")(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        input = Input(shape=input_shape)
        n = input_shape[0]

        d1 = conv2d(input=input, filters=n, batch_norm=False)
        d2 = conv2d(input=d1, filters=n*2, batch_norm=True)
        d3 = conv2d(input=d2, filters=n*4, batch_norm=True)
        d4 = conv2d(input=d3, filters=n*8, batch_norm=True)
        d5 = conv2d(input=d4, filters=n*8, batch_norm=True)
        d6 = conv2d(input=d5, filters=n*8, batch_norm=True)

        d7 = conv2d(input=d6, filters=n*8, batch_norm=True)

        u1 = deconv2d(input=d7, filters=n*8, skip_input=d6)
        u2 = deconv2d(input=u1, filters=n*8, skip_input=d5)
        u3 = deconv2d(input=u2, filters=n*8, skip_input=d4)
        u4 = deconv2d(input=u3, filters=n*4, skip_input=d3)
        u5 = deconv2d(input=u4, filters=n*2, skip_input=d2)
        u6 = deconv2d(input=u5, filters=n, skip_input=d1)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(input_shape[2], kernel_size=4, strides=1, padding="same", activation="tanh")(u7)

        return Model(input, output_img)

    def build_discriminator(self, input_shape):
        def discriminator_layer(input, filters, batch_norm, f_size=4):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding="same")(input)
            d = LeakyReLU(alpha=0.2)(d)
            if batch_norm:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=input_shape)
        img_B = Input(shape=input_shape)
        combined_imgs = Concatenate(axis=-1)([img_A, img_B]) # Concatenate image and conditioning image by channels to produce input

        n = input_shape[0]

        d1 = discriminator_layer(input=combined_imgs, filters=n, batch_norm=False)
        d2 = discriminator_layer(input=d1, filters=n * 2, batch_norm=True)
        d3 = discriminator_layer(input=d2, filters=n * 4, batch_norm=True)
        d4 = discriminator_layer(input=d3, filters=n * 8, batch_norm=True)

        validity = Conv2D(1, kernel_size=4, strides=1, padding="same")(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, data_dir, batch_size=1, sample_interval=50):
        os.makedirs(settings.root_dir + "resources/results/pix2pix/", exist_ok=True)

        start_time = datetime.datetime.now()

        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for count, (imgs_A, imgs_B) in enumerate(self.load_batch(data_dir=data_dir, batch_size=batch_size)):
                fake_As = self.generator.predict(imgs_B)

                discriminator_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                discriminator_loss_fake = self.discriminator.train_on_batch([fake_As, imgs_B], fake)
                discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
                generator_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                print("[Epoch %d/%d] [D loss real: %f; fake: %f] [G loss: %f] time: %s" %
                      (epoch, epochs, discriminator_loss[0], discriminator_loss[1], generator_loss[0], elapsed_time))

                if count % sample_interval == 0:
                    gen_imgs = np.concatenate([imgs_B, imgs_A, fake_As])

                    titles = ["Condition", "Original", "Generated"]
                    rows, cols = 3, 3
                    fig, axs = plt.subplots(rows, cols)
                    for i in range(rows):
                        for j in range(cols):
                            axs[i, j].imshow(gen_imgs[rows][cols])
                            axs[i, j].set_title(titles[i])
                            axs[i, j].axis("off")
                    fig.savefig(settings.root_dir + "resources/results/pix2pix/"+str(epochs))
                    plt.close()

    def load_batch(self, data_dir, batch_size):
        paths = glob(data_dir+"*")

        for i in range(len(paths)-1):
            batch = paths[i:i+batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img = self.load_img_as_np(img)
                h, w, _ = img.shape
                half_w = int(w/2)
                
                img_A = img[:, :half_w, :]
                img_B = img[:, half_w:, :]
                img_A = scipy.misc.imresize(img_A, self.img_shape)
                img_B = scipy.misc.imresize(img_B, self.img_shape)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)

            yield imgs_A, imgs_B

    def load_img_as_np(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
