import datetime
import logging
import os
import random
import re
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import tifffile
from PIL import Image

import tensorflow as tf
from keras.layers import BatchNormalization
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Model
from pipeline.processors.trainers.memory_saving_checkpoints_tf2 import checkpointable
from keras.backend import tensorflow_backend, set_session

import settings


__author__="cstur"


# discriminator outputs 1 if its a real pair
# img A is used to produce img B/fake B

# tinkered version of Pix2Pix model
class Pix2Pix:
    def __init__(self, image_shape, gpu_memory_friendly=False):
        logging.getLogger("matplotlib").setLevel(logging.ERROR)

        self.img_shape = image_shape
        self.gpu_memory_friendly = gpu_memory_friendly
        if self.gpu_memory_friendly:
            gpu_devices = tf.config.experimental.list_physical_devices('GPU')
            for device in gpu_devices:
                tf.config.experimental.set_memory_growth(device, True)
        if len(self.img_shape) != 3:
            print("Image should be width, height, channel but is " + str(self.img_shape))

        self.discriminator = self.build_discriminator(input_shape=self.img_shape)
        self.discriminator.compile(loss="mse", optimizer="adam")
        self.generator = self.build_generator(input_shape=self.img_shape)
        self.generator.compile(loss="mae", optimizer="adam")

        # 16 to 64 seem to be a good patch split
        # consider sliding window patches to get rid of the sudden changes in pics
        self.disc_patch = (int(self.img_shape[0] / 16), int(self.img_shape[0] / 16), 1)

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        fake_A = self.generator(img_B)
        valid = self.discriminator([fake_A, img_B])
        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'], optimizer="adam")

    # using only the generator with "mae" as loss seems to work better for segmenting pictures instead of the combination with the discriminator
    def build_generator(self, input_shape):
        if self.gpu_memory_friendly:
            @checkpointable
            def conv2d(input, filters, batch_norm, k_size=4, _checkpoint=True):
                d = Conv2D(filters, kernel_size=k_size, strides=2, padding="same", activation="relu")(input)
                d = LeakyReLU(alpha=0.2)(d)
                if batch_norm:
                    d = BatchNormalization(momentum=0.8)(d)
                return d

            @checkpointable
            def deconv2d(input, filters, skip_input, k_size=4, _checkpoint=True):
                u = UpSampling2D(size=2)(input)
                u = Conv2D(filters, kernel_size=k_size, strides=1, padding="same", activation="relu")(u)
                u = BatchNormalization(momentum=0.8)(u)
                u = Concatenate()([u, skip_input])
                return u
        else:
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
        d2 = conv2d(input=d1, filters=n * 2, batch_norm=True)
        d3 = conv2d(input=d2, filters=n * 4, batch_norm=True)
        d4 = conv2d(input=d3, filters=n * 8, batch_norm=True)
        d5 = conv2d(input=d4, filters=n * 8, batch_norm=True)
        # d6 = conv2d(input=d5, filters=n * 8, batch_norm=True)     # memory issues

        d7 = conv2d(input=d5, filters=n * 8, batch_norm=True)

        # u1 = deconv2d(input=d7, filters=n * 8, skip_input=d6)
        u2 = deconv2d(input=d7, filters=n * 8, skip_input=d5)
        u3 = deconv2d(input=u2, filters=n * 8, skip_input=d4)
        u4 = deconv2d(input=u3, filters=n * 4, skip_input=d3)
        u5 = deconv2d(input=u4, filters=n * 2, skip_input=d2)
        u6 = deconv2d(input=u5, filters=n, skip_input=d1)

        output_img = Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='same', activation="sigmoid")(u6)

        layers = [d1, d2, d3, d4, d5, d7, u2, u3, u4, u5, u6]
        watch = [v for layer in layers for v in layer.trainable_variables]  # delete if want to run
        return Model(input, output_img, _watch_vars=watch)

    def build_discriminator(self, input_shape):
        if self.gpu_memory_friendly:
            @checkpointable
            def discriminator_layer(input, filters, batch_norm, f_size=4, _checkpoint=True):
                d = Conv2D(filters, kernel_size=f_size, strides=2, padding="same", activation="relu")(input)
                d = LeakyReLU(alpha=0.2)(d)
                if batch_norm:
                    d = BatchNormalization(momentum=0.8)(d)
                return d
        else:
            def discriminator_layer(input, filters, batch_norm, f_size=4):
                d = Conv2D(filters, kernel_size=f_size, strides=2, padding="same", activation="relu")(input)
                d = LeakyReLU(alpha=0.2)(d)
                if batch_norm:
                    d = BatchNormalization(momentum=0.8)(d)
                return d

        img_A = Input(shape=input_shape)
        img_B = Input(shape=input_shape)
        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        n = input_shape[0]

        d1 = discriminator_layer(input=combined_imgs, filters=n, batch_norm=False)
        d2 = discriminator_layer(input=d1, filters=n * 2, batch_norm=True)
        d3 = discriminator_layer(input=d2, filters=n * 4, batch_norm=True)
        d4 = discriminator_layer(input=d3, filters=n * 8, batch_norm=True)

        validity = Conv2D(1, kernel_size=4, strides=1, padding="same")(d4)

        layers = [d1, d2, d3, d4]
        watch = [v for layer in layers for v in layer.graph.trainable_variables]
        return Model([img_A, img_B], validity)

    def train(self, epochs, data_dir, load_last_chkpt=False, generate_right=False, batch_size=5, sample_interval=1000):
        result_dir = os.path.join(settings.result_dir, "pix2pix", Path(data_dir).parent.stem,
                                  str(datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")))
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
                if len(os.listdir(os.path.join(settings.result_dir, "pix2pix", Path(data_dir).parent.stem, folder,
                                               "checkpoints"))) != 0:
                    potential_folders.append(folder)
            if not potential_folders:
                print("No previous weights found!")
            latest_folder = max([datetime.datetime.strptime(i, "%Y-%m-%d-%H-%M-%S") for i in potential_folders])
            goal_folder = os.path.join(settings.result_dir, "pix2pix", Path(data_dir).parent.stem,
                                       latest_folder.strftime("%Y-%m-%d-%H-%M-%S"), "checkpoints")
            discriminator_path = max(
                [os.path.join(goal_folder, d) for d in os.listdir(goal_folder) if "discriminator" in d],
                key=os.path.getctime)
            generator_path = max([os.path.join(goal_folder, d) for d in os.listdir(goal_folder) if "generator" in d],
                                 key=os.path.getctime)
            self.discriminator.load_weights(discriminator_path)
            self.generator.load_weights(generator_path)
            start_iter = int(re.sub('[^0-9]', '', discriminator_path[-5:]))
            print(
                "Loaded weights " + os.path.basename(discriminator_path) + " and " + os.path.basename(generator_path) +
                " from " + os.path.basename(os.path.dirname(goal_folder)))

        real = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        data_generator = self.load_batch(data_dir=data_dir, batch_size=batch_size, generate_right=generate_right)
        start_time = datetime.datetime.now()
        for epoch in range(start_iter, epochs):
            imgs_A, imgs_B = next(data_generator)  # abstraction A, realistic B
            fake_Bs = np.float64(self.generator.predict(imgs_A))

            discriminator_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], real)
            discriminator_loss_fake = self.discriminator.train_on_batch([imgs_A, fake_Bs], fake)
            discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

            elapsed_time = datetime.datetime.now() - start_time

            # if the discriminator learns more from real examples, improve the generator\
            # trains both the first 30 mins and then only trains the generator when it learned more from the real images
            if discriminator_loss_real > discriminator_loss_fake or elapsed_time < datetime.timedelta(minutes=30):
                # generator_loss = self.generator.train_on_batch(x=imgs_A, y=imgs_B)
                generator_loss = self.combined.train_on_batch([imgs_A, imgs_B], [real, imgs_A])
                generator_loss = np.average(generator_loss)
            else:
                generator_loss = 0

            # visualization
            if epoch % sample_interval == 0:
                if elapsed_time > datetime.timedelta(hours=0, minutes=30) * snapshot_count or epoch == epochs:
                    snapshot_count += 1
                    self.discriminator.save(os.path.join(checkpoint_path, "discriminator" + str(epoch)))
                    self.generator.save(os.path.join(checkpoint_path, "generator" + str(epoch)))
                    print("Saved model at " + str(epoch) + " epochs model.")

                # logging.info("[Epoch %d/%d] [G loss: %f] time: %s", epoch, epochs, generator_loss, str(elapsed_time))
                # print("[Epoch %d/%d] [G loss: %f] time: %s" % (epoch, epochs, generator_loss, str(elapsed_time)))
                elapsed_time = datetime.datetime.now() - start_time
                logging.info("[Epoch %d/%d] [D loss real: %f; fake: %f] [G loss: %f] time: %s",
                             epoch, epochs, discriminator_loss_real, discriminator_loss_fake, generator_loss, str(elapsed_time))
                print("[Epoch %d/%d] [D loss real: %f; fake: %f; total: %f] [G loss: %f] time: %s" %
                    (epoch, epochs, discriminator_loss_real, discriminator_loss_fake, discriminator_loss, generator_loss, str(elapsed_time)))

                gen_imgs = [imgs_B, imgs_A, fake_Bs]

                titles = ["Condition", "Original", "Generated"]
                if batch_size == 1:
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

    def load_batch(self, data_dir, batch_size, generate_right):
        paths = os.listdir(data_dir)
        paths = [os.path.join(data_dir, i) for i in paths]

        while True:
            random_start = random.randint(0, len(paths) - batch_size - 1)
            batch = paths[random_start:random_start + batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img = self.load_img_as_np(img)
                h, w, _ = img.shape
                half_w = int(w / 2)

                img_A = img[:, :half_w, :]
                img_B = img[:, half_w:, :]

                img_A = np.interp(scipy.misc.imresize(img_A, self.img_shape[0:2]), (0, 255), (0, 1))
                img_B = np.interp(scipy.misc.imresize(img_B, self.img_shape[0:2]), (0, 255), (0, 1))

                # switches the to generate picture to be from the right to the left
                if generate_right:
                    imgs_A.append(img_B)
                    imgs_B.append(img_A)
                else:
                    imgs_A.append(img_A)
                    imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)

            yield [imgs_A, imgs_B]

    def load_img_as_np(self, path):
        return tifffile.imread(path)
