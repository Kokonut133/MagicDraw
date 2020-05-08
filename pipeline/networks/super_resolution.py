from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.vgg19 import VGG19
import tensorflow as tf
import numpy as np

__author__="cstur"

class Super_resolution:
    def __init__(self, input_dim, output_dim):
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

    # This is the generator
    def sr_resnet(self, num_filters=64, num_res_blocks=16):
        def res_block(x_in, num_filters, momentum=0.8):
            x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
            x = BatchNormalization(momentum=momentum)(x)
            x = PReLU(shared_axes=[1, 2])(x)
            x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
            x = BatchNormalization(momentum=momentum)(x)
            x = Add()([x_in, x])
            return x

        def upsample(x_in, num_filters):
            x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
            x = Lambda(lambda x: tf.nn.depth_to_space(x, 2))
            return PReLU(shared_axes=[1, 2])(x)

        x_in = Input(shape=(None, None, 3))
        x = Lambda((x_in - self.DIV2K_RGB_MEAN) / 127.5)(x_in)

        x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
        x = x_1 = PReLU(shared_axes=[1, 2])(x)

        for _ in range(num_res_blocks):
            x = res_block(x, num_filters)

        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x_1, x])

        x = upsample(x, num_filters * 4)
        x = upsample(x, num_filters * 4)

        x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
        x = Lambda((x + 1) * 127.5)

        return Model(x_in, x)

    def discriminator(self, num_filters=64):
        def discriminator_block(x_in, num_filters, strides=1, batchnorm=True, momentum=0.8):
            x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
            if batchnorm:
                x = BatchNormalization(momentum=momentum)(x)
            return LeakyReLU(alpha=0.2)(x)

        x_in = Input(shape=(self.output_dim.shape[0], self.output_dim.shape[1], 3))
        x = Lambda(x_in/127.5-1)

        x = discriminator_block(x, num_filters, batchnorm=False)
        x = discriminator_block(x, num_filters, strides=2)
        x = discriminator_block(x, num_filters * 2)
        x = discriminator_block(x, num_filters * 2, strides=2)
        x = discriminator_block(x, num_filters * 4)
        x = discriminator_block(x, num_filters * 4, strides=2)
        x = discriminator_block(x, num_filters * 8)
        x = discriminator_block(x, num_filters * 8, strides=2)

        x = Flatten()(x)

        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1, activation='sigmoid')(x)

        return Model(x_in, x)

    def _vgg(self, output_layer):
        vgg = VGG19(input_shape=(None, None, 3), include_top=False)
        return Model(vgg.input, vgg.layers[output_layer].output)



    def resolve_single(self, model, lr):
        return self.resolve(model, tf.expand_dims(lr, axis=0))[0]

    def resolve(self, model, lr_batch):
        lr_batch = tf.cast(lr_batch, tf.float32)
        sr_batch = model(lr_batch)
        sr_batch = tf.clip_by_value(sr_batch, 0, 255)
        sr_batch = tf.round(sr_batch)
        sr_batch = tf.cast(sr_batch, tf.uint8)
        return sr_batch


    def evaluate(self, model, dataset):
        psnr_values = []
        for lr, hr in dataset:
            sr = self.resolve(model, lr)
            psnr_value = tf.image.psnr(hr, sr, max_val=255)[0]
            psnr_values.append(psnr_value)
        return tf.reduce_mean(psnr_values)
