import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
confusion_matrix,
classification_report,
accuracy_score
)

# ======================================

# Configuration

# ======================================

DATASET_PATH = r"Dataset\PlantVillage"
MODEL_PATH = r"models\resnet50_model.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ======================================

# Create Output Folder

# ======================================

os.makedirs(
"outputs/evaluation",
exist_ok=True
)

# ======================================

# Load Model

# ======================================

print("\nLoading Model...")

model = tf.keras.models.load_model(
MODEL_PATH
)

print("Model Loaded Successfully")

# ======================================

# Validation Dataset

# ======================================

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
rescale=1.0 / 255,
validation_split=0.2
)

val_generator = datagen.flow_from_directory(
DATASET_PATH,
target_size=IMAGE_SIZE,
batch_size=BATCH_SIZE,
class_mode="categorical",
subset="validation",
shuffle=False
)

# ======================================

# Class Names

# ======================================

class_names = list(
val_generator.class_indices.keys()
)

print("\nClasses:")

for idx, cls in enumerate(class_names):
    print(f"{idx} : {cls}")

# ======================================

# Predictions

# ======================================

print("\nGenerating Predictions...")

predictions = model.predict(
val_generator,
verbose=1
)

y_pred = np.argmax(
predictions,
axis=1
)

y_true = val_generator.classes

# ======================================

# Accuracy

# ======================================

accuracy = accuracy_score(
y_true,
y_pred
)

print(
f"\nValidation Accuracy: {accuracy * 100:.2f}%"
)

# ======================================

# Confusion Matrix

# ======================================

print("\nGenerating Confusion Matrix...")

cm = confusion_matrix(
y_true,
y_pred
)

plt.figure(figsize=(14, 12))

sns.heatmap(
cm,
annot=True,
fmt="d",
cmap="Blues",
xticklabels=class_names,
yticklabels=class_names
)

plt.title("ResNet50 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()

CM_PATH = (
"outputs/evaluation/confusion_matrix.png"
)

plt.savefig(
CM_PATH,
dpi=300
)

plt.close()

print("\nConfusion Matrix Saved:")
print(CM_PATH)

# ======================================

# Classification Report

# ======================================

print("\nGenerating Classification Report...")

report = classification_report(
y_true,
y_pred,
target_names=class_names
)

print("\n")
print(report)

REPORT_PATH = (
"outputs/evaluation/classification_report.txt"
)

with open(
REPORT_PATH,
"w",
encoding="utf-8"
) as file:


    file.write(
    "Crop Disease Detection\n"
)

file.write(
    "ResNet50 Evaluation Report\n\n"
)

file.write(
    f"Validation Accuracy: {accuracy * 100:.2f}%\n\n"
)

file.write(report)

print("\nClassification Report Saved:")
print(REPORT_PATH)

# ======================================

# Summary

# ======================================

print("\n================================")
print("Evaluation Completed")
print("================================")

print(
f"Accuracy : {accuracy * 100:.2f}%"
)

print(
f"Confusion Matrix : {CM_PATH}"
)

print(
f"Classification Report : {REPORT_PATH}"
)
