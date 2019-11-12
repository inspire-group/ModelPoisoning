# -*- coding:utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
from tensorflow import keras


def cifar10_model(name='cifar10', img_shape=(32, 32, 3)):
    cnn = keras.Sequential(name=name)

    cnn.add(keras.layers.Conv2D(64, kernel_size=5, strides=2, padding='same',
                                input_shape=img_shape
                                ))
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.LeakyReLU(alpha=0.2))

    cnn.add(keras.layers.Conv2D(128, kernel_size=5, strides=2, padding='same', use_bias=False
                                ))
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.LeakyReLU(alpha=0.2))

    cnn.add(keras.layers.Conv2D(256, kernel_size=5, strides=2, padding='same', use_bias=False))
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.LeakyReLU(alpha=0.2))

    cnn.add(keras.layers.Conv2D(512, kernel_size=5, strides=2, padding='same', use_bias=False))
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.LeakyReLU(alpha=0.2))


    cnn.add(keras.layers.Flatten())
    cnn.add(keras.layers.Dense(10))

    return cnn

