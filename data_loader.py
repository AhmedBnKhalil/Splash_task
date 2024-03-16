from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Define your dataset path
dataset_path = '/Users/ahmed/PycharmProjects/Splash_task/Spalsh_Data'

# Initialize the ImageDataGenerator with a validation split for training data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # IMPORTANT: Define the validation split here
)

# Initialize the ImageDataGenerator for validation data (no data augmentation, just rescaling)
validation_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2  # Use the same validation split ratio as the training
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=5,
    class_mode='categorical',
    subset='training'  # Specify this is training data
)

# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=5,
    class_mode='categorical',
    subset='validation'  # Specify this is validation data
)

# Iterate over class_indices and print class names with the number of images
for class_name, class_index in train_generator.class_indices.items():
    # Count the number of occurrences of each class index in train_generator.classes
    num_images = sum(train_generator.classes == class_index)
    print(f"Class '{class_name}' has {num_images} images")