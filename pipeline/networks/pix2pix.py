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


class Pix2Pix:
    def __init__(self, image_shape):
        self.imgenerator_shape = image_shape
        if len(self.imgenerator_shape) != 3:
            print("Image should be w, h, c but is " + str(self.imgenerator_shape))

        patchsize = int(self.imgenerator_shape[0]/16)   # 16 parts of the image are evaluated if real
        self.disc_patch = (patchsize, patchsize, 1)

        self.discriminator = self.buildiscriminator_discriminator(input_shape=self.imgenerator_shape)
        self.discriminator.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
        self.generator = self.buildiscriminator_generator(input_shape=self.imgenerator_shape)

        img_A = Input(shape=self.imgenerator_shape)
        img_B = Input(shape=self.imgenerator_shape)
        fake_A = self.generator(img_B)

        valid = self.discriminator([fake_A, img_B])
        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=["mse", "mae"], loss_weights=[1, 100], optimizer="adam")

    def buildiscriminator_generator(self, input_shape):
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
        start_time = datetime.datetime.now()

        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for count, (imgs_A, imgs_B) in enumerate(self.load_batch(data_dir=data_dir, batch_size=batch_size)):
                fake_A = self.generator.predict(imgs_B)

                discriminator_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                discriminator_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

                generator_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                print("[Epoch %d/%d] [D loss real: %f; fake: %f] [G loss: %f] time: %s" %
                      (epoch, epochs, discriminator_loss[0], discriminator_loss[1], generator_loss[0], elapsed_time))

                # If at save interval => save generated image samples
                if count % sample_interval == 0:
                    self.show_results(epoch, count)

    def show_results(self, epoch, batch_i):
        os.makedirs("images/%s" % self.dataset_name, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ["Condition", "Generated", "Original"]
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis("off")
                cnt += 1
        fig.savefig("images/%s/%discriminator_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()

    def load_data(self, batch_size=1, is_testing=False):
        path = glob(self.data_dir+"*")
        batch_images = np.random.choice(path, size=batch_size)

        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img = self.load_img_as_np(img_path)

            h, w, _ = img.shape
            _w = int(w/2)
            img_A, img_B = img[:, :_w, :], img[:, _w:, :]

            img_A = scipy.misc.imresize(img_A, self.img_res)
            img_B = scipy.misc.imresize(img_B, self.img_res)

            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def load_batch(self, data_dir, batch_size=1, is_testing=False):
        paths = glob(data_dir+"*")

        for i in range(len(paths)-1):
            batch = paths[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img = self.load_img_as_np(img)
                h, w, _ = img.shape
                half_w = int(w/2)
                img_A = img[:, :half_w, :]
                img_B = img[:, half_w:, :]

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def load_img_as_np(self, path):
        return scipy.misc.load_img_as_np(path, mode='RGB').astype(np.float)
