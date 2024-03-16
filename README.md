# **Product Image Classifier**

## Overview

This project aims to enhance the intelligence of an e-commerce platform by automatically categorizing product images into predefined groups such as fashion, nutrition, or accessories. Utilizing the power of Convolutional Neural Networks (CNNs) and transfer learning, specifically the ResNet50 model pre-trained on ImageNet, we can accurately classify images into their respective categories, streamlining the product upload process and improving user experience on the platform.

## Approach

### Data Preparation

The dataset, sourced from the platform's product images, is organized into directories named after each category, facilitating easy loading and automatic labeling. We augmented the dataset using TensorFlow's ImageDataGenerator to include image transformations such as rotations, zooming, and flipping, enhancing the model's robustness.

### Model Building

We employed the ResNet50 model, leveraging its deep architecture and pre-trained weights for feature extraction. Custom dense layers were added on top of the base model to tailor it to our specific classification task, followed by dropout layers to prevent overfitting.

### Training and Validation

The model was trained on the augmented dataset, using class weights to handle class imbalance effectively. Various callbacks were implemented, including Early Stopping, ReduceLROnPlateau, and ModelCheckpoint, to optimize training and save the best-performing model.

### Fine-Tuning

After the initial training, top layers of the ResNet50 model were unfrozen, and the model underwent further training to fine-tune the weights for our specific dataset, significantly improving the accuracy.

### Functionalities

* **Automatic Image Classification:** Automatically categorizes product images into predefined groups, facilitating product management and discovery.
* **Model Monitoring and Evaluation**: Utilizes TensorBoard for real-time monitoring of training metrics and performances.
* **Scalability**: The model structure and training process are designed to be scalable, easily adapting to new categories or larger datasets.

### Setup and Usage

#### Prerequisites
* Python 3.8+
* TensorFlow 2.x
# Splash_task
