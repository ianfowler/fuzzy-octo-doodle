import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.io
import os
from data import utils as CTRUtil
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
import keras
import data_preparation


IMG_HW = 512

def make_segmentor(kernel_size=3):
    inputs = keras.Input(shape=(IMG_HW, IMG_HW, 1))

    x = layers.Conv2D(64, kernel_size, strides=4, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, kernel_size, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(512, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(4, 1, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Conv2DTranspose(4, kernel_size,
                                     strides=16,
                                     activation="softmax",
                                     padding="same")(x)

    model = keras.Model(inputs, outputs)

    return model


def make_discriminator(kernel_size=3):
    inputs = keras.Input(shape=(IMG_HW, IMG_HW, 4))

    x = layers.Conv2D(64, kernel_size, strides=4, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, kernel_size, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, kernel_size, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(512, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    output = layers.Dense(1, activation="softmax")(x)

    model = keras.Model(inputs, output)

    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


def make_adversial_network(segmentor, discriminator):
    # This will only be used for training the segmentor.
    # Note, the weights in the discriminator and generator are shared.
    discriminator.trainable = False
    adv = Sequential([segmentor, discriminator])
    adv.compile(loss='binary_crossentropy', optimizer='adam')
    return adv 


def train(imgs, segs, epochs=1, batch_size=32, path=''):

    segmentor = make_segmentor()
    discriminator = make_discriminator()
    adversial_net = make_adversial_network(segmentor, discriminator)

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}')

        discr_loss = 0
        gen_loss = 0
        for _ in tqdm(range(batch_size)):
            rand_choice = np.random.choice(segs.shape[0], batch_size,
                                           replace=False)

            real_images = imgs[rand_choice]
            real_segments = segs[rand_choice]

            generated_segments = segmentor.predict(real_images)
            discrimination_data = np.concatenate(
                [real_segments, generated_segments])

            # Labels for generated and real data, uses soft label trick
            discrimination_labels = 0.1 * np.ones(2 * batch_size)
            discrimination_labels[:batch_size] = 0.9

            # To train, we alternate between training just the discriminator
            # and just the generator.
            discriminator.trainable = True
            discr_loss += discriminator.train_on_batch(discrimination_data,
                                                       discrimination_labels)

            # Trick to 'freeze' discriminator weights in adversial_net. Only
            # the generator weights will be changed, which are shared with
            # the generator.
            discriminator.trainable = False
            # N.B, changing the labels because now we want to 'fool' the
            # discriminator.
            gen_loss += adversial_net.train_on_batch(
                real_images, np.ones(batch_size))

        print(f'Discriminator Loss: {discr_loss/batch_size}')
        print(f'Generator Loss:     {gen_loss/batch_size}')

        segmentor.save("{}seg_{}.h5".format(path, epoch))
        discriminator.save("{}disc_{}.h5".format(path, epoch))
        adversial_net.save("{}adv_{}.h5".format(path, epoch))

imgs, segs = data_preparation.load_data("JSRT_imgs", "JSRT_segs", relpath="./prepared_data/")
train(imgs, segs, epochs=100, path="./savesMay2/")
