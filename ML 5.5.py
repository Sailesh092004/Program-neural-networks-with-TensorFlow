import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Specify the directory where your dataset is located
train_dir = 'D:\\venv\\horse-or-human'

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1. / 255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,            # Update to the correct path
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary'
)

