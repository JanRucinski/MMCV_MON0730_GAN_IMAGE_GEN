
import tensorflow as tf
import keras

from constants import BATCH_SIZE, IMAGE_SIZE

image_path = ".\\images\\input\\data"

def make_dataset():
    dataset = keras.utils.image_dataset_from_directory(
        image_path,
        labels=None,
        crop_to_aspect_ratio=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        
    )
    # Repeat the dataset indefinitely
    dataset = dataset.map(lambda x: x / 255.0)
    dataset = dataset.repeat()  
    return dataset