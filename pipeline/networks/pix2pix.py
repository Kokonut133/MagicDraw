from __future__ import print_function, division

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import BatchNormalization
from keras.layers import Input, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam


class Pix2Pix:
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.generator = self.build_generator(input_shape=self.img_shape)


        # # Build and compile the discriminator
        # self.discriminator = self.build_discriminator()
        # self.discriminator.compile(loss='mse',
        #                            optimizer=optimizer,
        #                            metrics=['accuracy'])

        # # Input images and their conditioning images
        # img_A = Input(shape=self.img_shape)
        # img_B = Input(shape=self.img_shape)
        #
        # # By conditioning on B generate a fake version of A
        # fake_A = self.generator(img_B)
        #
        # # For the combined model we will only train the generator
        # self.discriminator.trainable = False
        #
        # # Discriminators determines validity of translated images / condition pairs
        # valid = self.discriminator([fake_A, img_B])
        #
        # self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        # self.combined.compile(loss=['mse', 'mae'],
        #                       loss_weights=[1, 100],
        #                       optimizer=optimizer)

    def build_generator(self, input_shape):
        def conv2d(input, filters, batch_norm, k_size=4):
            d = Conv2D(filters, kernel_size=k_size, strides=2, padding='same')(input)
            d = LeakyReLU(alpha=0.2)(d)
            if batch_norm:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(input, filters, skip_input, k_size=4):
            u = UpSampling2D(size=2)(input)
            u = Conv2D(filters, kernel_size=k_size, strides=1, padding='same', activation='relu')(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        input = Input(shape=input_shape)
        # filter base number -> 256
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
        output_img = Conv2D(input_shape[2], kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(input, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                                                      batch_i,
                                                                                                      self.data_loader.n_batches,
                                                                                                      d_loss[0],
                                                                                                      100 * d_loss[1],
                                                                                                      g_loss[0],
                                                                                                      elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()