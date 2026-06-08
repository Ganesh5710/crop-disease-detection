import os
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization
)
from keras.callbacks import EarlyStopping

# ======================================
# Dataset Path
# ======================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = r"Dataset\PlantVillage"

print("Dataset Path:", DATASET_PATH)
print("Exists:", os.path.exists(DATASET_PATH))

# ======================================
# Hyperparameters
# ======================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3

# ======================================
# Data Generators
# ======================================

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
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

# ======================================
# Class Information
# ======================================

num_classes = len(train_generator.class_indices)

print("\nClasses Found:")
print(train_generator.class_indices)

print("\nNumber of Classes:", num_classes)

# ======================================
# CNN Model
# ======================================

model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(224, 224, 3)
    ),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(
        128,
        (3, 3),
        activation='relu'
    ),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(
        256,
        (3, 3),
        activation='relu'
    ),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(
        512,
        activation='relu'
    ),

    Dropout(0.5),

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

print("\nModel Summary:")
model.summary()

# ======================================
# Early Stopping
# ======================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ======================================
# Train Model
# ======================================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=20,
    validation_steps=5,
    callbacks=[early_stop]
)

# ======================================
# Save Model
# ======================================

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "cnn_model.keras"
)

model.save(MODEL_PATH)

print("\nModel Saved Successfully")
print("Location:", MODEL_PATH)

# ======================================
# Accuracy Graph
# ======================================

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.legend()

plt.tight_layout()

GRAPH_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(GRAPH_DIR, exist_ok=True)

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "training_results.png"
    )
)

plt.show()

print("\nTraining Completed Successfully")