
import tensorflow as tf
import keras



image_path = ".\\images\\input\\data"

dataset = keras.utils.image_dataset_from_directory(
    image_path,
    labels=None,
    crop_to_aspect_ratio=True,
    image_size=(256, 256),
)

dataset.save("dataset", compression="gzip")
