import tensorflow as tf

from data_loader import train_generator


def predict_image_class(image_path, model_path='product_image_classifier_model'):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array /= 255.  # Rescale the image

    # Make predictions
    predictions = model.predict(img_array)

    # Return the class with the highest probability
    class_index = tf.argmax(predictions, axis=1).numpy()[0]
    class_label = list(train_generator.class_indices.keys())[class_index]
    print( class_label)


predict_image_class("Spalsh_Data/Fashion/IMG_7951.PNG", './models/best_model.keras' )
