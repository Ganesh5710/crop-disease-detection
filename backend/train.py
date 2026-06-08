import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Using direct Keras imports completely avoids the tf/tensorflow naming conflict
import keras
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization
)
import tensorflow as tf

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
from keras.callbacks import EarlyStopping


# ======================================
# Dataset Path & Hyperparameters
# ======================================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "Dataset", "PlantVillage")
)


print("Dataset Path:", DATASET_PATH)
print("Exists:", os.path.exists(DATASET_PATH))
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ======================================
# Data Generators (Fixed Validation Split)
# ======================================

# Training generator with heavy data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

# SEPARATE Validation generator (Strictly no augmentations except rescaling!)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_generator.class_indices)
print("Classes identified:", num_classes)

# ======================================
# Custom CNN Architecture (Optimized)
# ======================================

model = Sequential([

    # Block 1
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(224,224,3)
    ),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Block 2
    Conv2D(
        64,
        (3,3),
        activation='relu'
    ),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Block 3
    Conv2D(
        128,
        (3,3),
        activation='relu'
    ),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Block 4 (Added missing BatchNormalization)
    Conv2D(
        256,
        (3,3),
        activation='relu'
    ),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Fully Connected Layer
    Flatten(),
    Dense(
        512,
        activation='relu'
    ),
    Dropout(0.5),
    
    # Output Layer
    Dense(
        num_classes,
        activation='softmax'
    )
])

# ======================================
# Compile Model
# ======================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ======================================
# Early Stopping Callback
# ======================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ======================================
# Model Training Run
# ======================================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=3,
    steps_per_epoch=20,
    validation_steps=5,
    callbacks=[early_stop]
)

# ======================================
# Save Model (Modern Native Format)
# ======================================

os.makedirs("../models", exist_ok=True)
model.save("../models/cnn_model.keras")
print("Model Saved Successfully to '../models/cnn_model.keras'")

# ======================================
# Visual Diagnostics (Plotting Curves)
# ======================================

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()