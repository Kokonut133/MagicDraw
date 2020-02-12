import datetime
import gc
import logging
import os
import random
import re
from glob import glob
from pathlib import Path

import tensorflow as tf
from keras_contrib.callbacks import tensorboard

import settings
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import keras
from keras.layers import BatchNormalization
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
import settings

# discriminator outputs 1 if its a real pair
# img A is used to produce img B/fake B

class Pix2Pix:
    def __init__(self, image_shape, light_w=False):
        logging.getLogger("matplotlib").setLevel(logging.ERROR)

        self.img_shape = image_shape
        if len(self.img_shape) != 3:
            print("Image should be w, h, c but is " + str(self.img_shape))

        self.discriminator = self.build_discriminator(input_shape=self.img_shape)
        self.discriminator.trainable = False
        self.discriminator.compile(loss="mse", optimizer="adam")
        if light_w:
            self.generator = self.build_lightw_generator(input_shape=self.img_shape)
        else:
            self.generator = self.build_generator(input_shape=self.img_shape)
        self.generator.compile(loss="mse", optimizer="adam")

        self.disc_patch = (int(self.img_shape[0]/16), int(self.img_shape[0]/16), 1)

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

    def build_lightw_generator(self, input_shape):
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

        d7 = conv2d(input=d3, filters=n*8, batch_norm=True)

        u4 = deconv2d(input=d7, filters=n*4, skip_input=d3)
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

    def train(self, epochs, data_dir, load_last_chkpt=False, batch_size=5, sample_interval=1000):
        result_dir = os.path.join(settings.result_dir, "pix2pix", Path(data_dir).parent.stem, str(datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")))
        os.makedirs(result_dir, exist_ok=True)
        checkpoint_path = os.path.join(result_dir, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)
        logging.basicConfig(filename=os.path.join(result_dir, "log.txt"), level=logging.INFO, filemode="w")

        start_iter = 0
        snapshot_count = 1

        if load_last_chkpt:
            list_of_folders = os.listdir(os.path.join(settings.result_dir, "pix2pix", Path(data_dir).parent.stem))
            potential_folders = []
            for folder in list_of_folders:  # remove with no checkpoints
                if len(os.listdir(os.path.join(settings.result_dir, "pix2pix", Path(data_dir).parent.stem, folder, "checkpoints"))) != 0:
                    potential_folders.append(folder)
            if not potential_folders:
                print("No previous weights found!")
            latest_folder = max([datetime.datetime.strptime(i, "%Y-%m-%d-%H-%M-%S") for i in potential_folders])
            goal_folder = os.path.join(settings.result_dir, "pix2pix", Path(data_dir).parent.stem, latest_folder.strftime("%Y-%m-%d-%H-%M-%S"), "checkpoints")
            discriminator_path = max([os.path.join(goal_folder, d) for d in os.listdir(goal_folder) if "discriminator" in d], key=os.path.getctime)
            generator_path = max([os.path.join(goal_folder, d) for d in os.listdir(goal_folder) if "generator" in d], key=os.path.getctime)
            self.discriminator.load_weights(discriminator_path)
            self.generator.load_weights(generator_path)
            start_iter = int(re.sub('[^0-9]', '', discriminator_path[-5:]))
            print("Loaded weights " + os.path.basename(discriminator_path) + " and " + os.path.basename(generator_path) +
                  " from " + os.path.basename(os.path.dirname(goal_folder)))

        real = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        data_generator = self.load_batch(data_dir=data_dir, batch_size=batch_size)
        start_time = datetime.datetime.now()
        for epoch in range(start_iter, epochs):
            imgs_A, imgs_B = next(data_generator)   # abstraction A, realistic B
            fake_Bs = self.generator.predict(imgs_A)

            discriminator_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], real)
            discriminator_loss_fake = self.discriminator.train_on_batch([imgs_A, fake_Bs], fake)
            # discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
            generator_loss = self.generator.train_on_batch(x=imgs_A, y=imgs_B)

            # visualization
            if epoch % sample_interval == 0:
                elapsed_time = datetime.datetime.now() - start_time

                if elapsed_time > datetime.timedelta(hours=0, minutes=30)*snapshot_count or epoch == epochs:
                    snapshot_count += 1
                    self.discriminator.save(os.path.join(checkpoint_path, "discriminator" + str(epoch)))
                    self.generator.save(os.path.join(checkpoint_path, "generator" + str(epoch)))
                    print("Saved model at " + str(epoch) + " epochs model.")

                logging.info("[Epoch %d/%d] [D loss real: %f; fake: %f] [G loss: %f] time: %s",
                             epoch, epochs, discriminator_loss_real, discriminator_loss_fake, generator_loss, str(elapsed_time))
                print("[Epoch %d/%d] [D loss real: %f; fake: %f] [G loss: %f] time: %s" %
                      (epoch, epochs, discriminator_loss_real, discriminator_loss_fake, generator_loss, str(elapsed_time)))

                gen_imgs = [imgs_B, imgs_A, fake_Bs]

                titles = ["Condition", "Original", "Generated"]
                if batch_size <= 1:
                    fig, ax = plt.subplots(nrows=1, ncols=len(gen_imgs))
                    for j in range(len(gen_imgs)):
                        ax[j].imshow(gen_imgs[j][0])
                        ax[j].set_title(titles[j])
                        ax[j].axis("off")
                    fig.savefig(os.path.join(result_dir, str(epoch)))
                    plt.close()
                else:
                    if batch_size > 3:
                        rows = 3
                    else:
                        rows = batch_size
                    fig, axs = plt.subplots(nrows=rows, ncols=len(gen_imgs))
                    for i in range(rows):
                        for j in range(len(gen_imgs)):
                            axs[i][j].imshow(gen_imgs[j][i])
                            axs[i][j].set_title(titles[j])
                            axs[i][j].axis("off")
                    fig.savefig(os.path.join(result_dir, str(epoch)))
                    plt.close()

    def load_batch(self, data_dir, batch_size):
        paths = os.listdir(data_dir)
        paths = [os.path.join(data_dir, i) for i in paths]

        while True:
            random_start = random.randint(0, len(paths)-batch_size-1)
            batch = paths[random_start:random_start+batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img = self.load_img_as_np(img)
                h, w, _ = img.shape
                half_w = int(w/2)

                img_A = img[:, :half_w, :]
                img_B = img[:, half_w:, :]

                img_A = scipy.misc.imresize(img_A, self.img_shape[0:2])
                img_B = scipy.misc.imresize(img_B, self.img_shape[0:2])


                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)

            yield imgs_A, imgs_B

    def load_img_as_np(self, path):
        return scipy.misc.imread(path, mode="RGB").astype(np.float)
