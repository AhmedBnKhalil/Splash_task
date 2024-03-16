import datetime
import io
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from data_loader import train_generator, validation_generator
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1: INFO, 2: WARNING, 3: ERROR)
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (alternative method)


def get_model_summary(model):
    stream = io.StringIO()
    # Modify the lambda function to accept arbitrary keyword arguments
    model.summary(print_fn=lambda x, **kwargs: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def compare_model_summaries(summary1, summary2):
    def extract_parameters(summary):
        lines = summary.split('\n')
        total_params = trainable_params = non_trainable_params = 0
        for line in lines:
            if 'Total params:' in line:
                total_params_str = line.split('Total params:')[1].split('(')[0].replace(',', '').strip()
                total_params = int(total_params_str)
            elif 'Trainable params:' in line:
                trainable_params_str = line.split('Trainable params:')[1].split('(')[0].replace(',', '').strip()
                trainable_params = int(trainable_params_str)
            elif 'Non-trainable params:' in line:
                non_trainable_params_str = line.split('Non-trainable params:')[1].split('(')[0].replace(',', '').strip()
                non_trainable_params = int(non_trainable_params_str)
        return total_params, trainable_params, non_trainable_params

    total_params1, trainable_params1, non_trainable_params1 = extract_parameters(summary1)
    total_params2, trainable_params2, non_trainable_params2 = extract_parameters(summary2)

    print("Differences in model parameters:")
    print(f"Total Parameters: From {total_params1} to {total_params2}")
    print(f"Trainable Parameters: From {trainable_params1} to {trainable_params2}")
    print(f"Non-trainable Parameters: From {non_trainable_params1} to {non_trainable_params2}")


warnings.filterwarnings('ignore')

# Initialize the base model without the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Construct the top layers for your specific classification task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', name='FC_1024')(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
predictions = Dense(len(train_generator.class_indices), activation='softmax', name='Output_Layer')(x)

# Instantiate the model with the base model's input and the new output layer
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
initial_summary = get_model_summary(model)

# Callbacks for early stopping, learning rate adjustment, model checkpointing, and TensorBoard logging
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
model_checkpoint = ModelCheckpoint(filepath='./models/best_model.keras', save_best_only=True, monitor='val_accuracy')
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Train the model with class weights
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=30,
    callbacks=[early_stopping, lr_scheduler, model_checkpoint, tensorboard_callback],
    class_weight=class_weights_dict
)
post_training_summary = get_model_summary(model)

# Plotting the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plotting the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Fine-tuning: Unfreeze some top layers of the base model and recompile
base_model.trainable = True

model.compile(optimizer=Adam(1e-5),  # Lower learning rate for fine-tuning
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tuning training
history_fine = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator,
    callbacks=[early_stopping, lr_scheduler, model_checkpoint, tensorboard_callback]  # Reuse the same callbacks
)
fine_tuned_summary = get_model_summary(model)

# Comparing before and after initial training
print("Comparing before and after initial training:")
compare_model_summaries(initial_summary, post_training_summary)

# Comparing after initial training and after fine-tuning
print("\nComparing after initial training and after fine-tuning:")
compare_model_summaries(post_training_summary, fine_tuned_summary)

model.save('./models/model.keras')
