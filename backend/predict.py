import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ======================================
# Model Path
# ======================================

MODEL_PATH = os.path.join(
    "..",
    "models",
    "resnet50_model.keras"
)

# ======================================
# Class Names
# ======================================

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",

    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",

    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# ======================================
# Load Model
# ======================================

print("Loading ResNet50 Model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model Loaded Successfully")

# ======================================
# Image Preprocessing
# ======================================

def preprocess_image(image_path):

    image = Image.open(image_path)

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image = np.array(image)

    image = image.astype("float32") / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image

# ======================================
# Disease Prediction
# ======================================

def predict_disease(image_path):

    image = preprocess_image(
        image_path
    )

    prediction = model.predict(
        image,
        verbose=0
    )

    class_index = np.argmax(
        prediction
    )

    confidence = (
        float(np.max(prediction))
        * 100
    )

    disease_name = CLASS_NAMES[
        class_index
    ]

    return {
        "disease": disease_name,
        "confidence": round(
            confidence,
            2
        )
    }

# ======================================
# Local Testing
# ======================================

if __name__ == "__main__":

    test_image = "sample.jpg"

    if os.path.exists(test_image):

        result = predict_disease(
            test_image
        )

        print("\nPrediction Result:")
        print(result)

    else:

        print(
            "\nPlace sample.jpg inside backend folder for testing."
        )