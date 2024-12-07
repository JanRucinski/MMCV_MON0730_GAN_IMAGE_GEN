import keras
import tensorflow as tf
import numpy as np

from constants import BATCH_SIZE, IMAGE_SIZE, LATENT_DIM

# Define the generator model
def create_generator():
    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(LATENT_DIM,)),
        keras.layers.Dense(LATENT_DIM * 5),
        keras.layers.LeakyReLU(),
        keras.layers.BatchNormalization(momentum=0.8),
        keras.layers.Dense(LATENT_DIM * 10),
        keras.layers.LeakyReLU(),
        keras.layers.BatchNormalization(momentum=0.8),
        keras.layers.Dense(np.prod((IMAGE_SIZE, IMAGE_SIZE, 3)), activation='tanh'),
        keras.layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 3)),
    ], name='generator')
    return model

# Define the discriminator model
def create_discriminator():
    model = keras.models.Sequential([
        keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(LATENT_DIM * 10),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(LATENT_DIM * 5),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(1, activation='sigmoid'),
    ], name='discriminator')
    return model

# Instantiate the models
generator = create_generator()
discriminator = create_discriminator()

