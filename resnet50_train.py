import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
GlobalAveragePooling2D,
Dense,
Dropout
)
from tensorflow.keras.callbacks import (
EarlyStopping,
ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

# ======================================

# Configuration

# ======================================

DATASET_PATH = r"Dataset\PlantVillage"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 2

# ======================================

# Create Folders

# ======================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/training", exist_ok=True)

# ======================================

# Dataset Check

# ======================================

print("\nDataset Path:", DATASET_PATH)
print("Exists:", os.path.exists(DATASET_PATH))

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
f"Dataset not found: {DATASET_PATH}"
)

# ======================================

# Data Generators

# ======================================

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
rescale=1.0 / 255,
validation_split=0.2,
rotation_range=20,
zoom_range=0.2,
horizontal_flip=True,
brightness_range=[0.8, 1.2]
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
rescale=1.0 / 255,
validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
DATASET_PATH,
target_size=IMAGE_SIZE,
batch_size=BATCH_SIZE,
class_mode="categorical",
subset="training"
)

val_generator = val_datagen.flow_from_directory(
DATASET_PATH,
target_size=IMAGE_SIZE,
batch_size=BATCH_SIZE,
class_mode="categorical",
subset="validation"
)

# ======================================

# Classes

# ======================================

num_classes = len(
train_generator.class_indices
)

print("\nClasses Found:")

for cls, idx in train_generator.class_indices.items():
    print(f"{idx} : {cls}")

print(
f"\nTotal Classes: {num_classes}"
)

# ======================================

# ResNet50 Base Model

# ======================================

print("\nLoading ResNet50...")

base_model = ResNet50(
weights="imagenet",
include_top=False,
input_shape=(224, 224, 3)
)

base_model.trainable = False

# ======================================

# Build Model

# ======================================

model = Sequential ( [
base_model,


GlobalAveragePooling2D(),

Dense(
    512,
    activation="relu"
),

Dropout(0.5),

Dense(
    256,
    activation="relu"
),

Dropout(0.3),

Dense(
    num_classes,
    activation="softmax"
)


])

# ======================================

# Compile

# ======================================

model.compile(
optimizer=Adam(
learning_rate=0.0001
),
loss="categorical_crossentropy",
metrics=["accuracy"]
)

print("\nModel Summary")
model.summary()

# ======================================

# Callbacks

# ======================================

early_stop = EarlyStopping(
monitor="val_loss",
patience=3,
restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
monitor="val_loss",
factor=0.5,
patience=2,
verbose=1
)

# ======================================

# Train

# ======================================

print("\nStarting Training...\n")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=2,
    steps_per_epoch=20,
    validation_steps=5,
    callbacks=[
        early_stop,
        reduce_lr
    ]
)

# ======================================

# Save Model

# ======================================

MODEL_PATH = (
"models/resnet50_model.keras"
)

model.save(
MODEL_PATH
)

print("\nModel Saved Successfully")
print(MODEL_PATH)

# ======================================

# Accuracy & Loss Graph

# ======================================

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(
len(acc)
)

plt.figure(figsize=(12, 5))

# Accuracy Plot

plt.subplot(1, 2, 1)

plt.plot(
epochs_range,
acc,
label="Training Accuracy"
)

plt.plot(
epochs_range,
val_acc,
label="Validation Accuracy"
)

plt.title(
"ResNet50 Accuracy"
)

plt.legend()

# Loss Plot

plt.subplot(1, 2, 2)

plt.plot(
epochs_range,
loss,
label="Training Loss"
)

plt.plot(
epochs_range,
val_loss,
label="Validation Loss"
)

plt.title(
"ResNet50 Loss"
)

plt.legend()

plt.tight_layout()

GRAPH_PATH = (
"outputs/training/resnet50_training.png"
)

plt.savefig(
GRAPH_PATH,
dpi=300
)

plt.close()

print("\nGraph Saved:")
print(GRAPH_PATH)

# ======================================

# Final Results

# ======================================

print(
f"\nTraining Accuracy: {acc[-1] * 100:.2f}%"
)

print(
f"Validation Accuracy: {val_acc[-1] * 100:.2f}%"
)

print(
"\nWeek 3 ResNet50 Training Completed Successfully"
)
