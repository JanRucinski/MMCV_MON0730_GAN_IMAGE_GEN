import keras
import tensorflow as tf

IMAGE_SIZE = 64

generator = keras.models.Sequential([
    keras.layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3)),
    keras.layers.Dense(IMAGE_SIZE * IMAGE_SIZE * 3, activation='relu'),
    keras.layers.Dense(IMAGE_SIZE * IMAGE_SIZE * 3, activation='relu'),
    keras.layers.Dense(IMAGE_SIZE * IMAGE_SIZE * 3, activation='relu'),
], name='generator')

