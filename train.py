import keras
import tensorflow as tf


from generator.basic import generator as BasicGenerator
from discriminator.basic import discriminator as BasicDiscriminator

IMAGE_SIZE = 64


optimizer = keras.optimizers.Adam(0.0002, 0.5)


