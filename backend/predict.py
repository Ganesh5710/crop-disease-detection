import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ======================================
# Model Path
# ======================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "models",
        "resnet50_model.keras"
    )
)

print("Model Path:", MODEL_PATH)
print("Model Exists:", os.path.exists(MODEL_PATH))

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
# Load Model Safely
# ======================================

model = None

try:

    if os.path.exists(MODEL_PATH):

        print("Loading Model...")

        model = tf.keras.models.load_model(
            MODEL_PATH
        )

        print("Model Loaded Successfully")

    else:

        print(
            f"WARNING: Model not found: {MODEL_PATH}"
        )

except Exception as e:

    print(
        f"Model Loading Error: {str(e)}"
    )

# ======================================
# Image Preprocessing
# ======================================

def preprocess_image(image_path):

    image = Image.open(image_path)

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image = np.array(image)

    image = image.astype(
        "float32"
    ) / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image

# ======================================
# Disease Prediction
# ======================================

def predict_disease(image_path):

    try:

        if model is None:

            return {
                "disease": "Model Not Available",
                "confidence": 0.0
            }

        image = preprocess_image(
            image_path
        )

        prediction = model.predict(
            image,
            verbose=0
        )

        class_index = int(
            np.argmax(prediction)
        )

        confidence = float(
            np.max(prediction)
        ) * 100

        if class_index >= len(CLASS_NAMES):

            return {
                "disease": "Unknown",
                "confidence": 0.0
            }

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

    except Exception as e:

        print(
            "Prediction Error:",
            str(e)
        )

        return {
            "disease": "Error",
            "confidence": 0.0,
            "error": str(e)
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

        print(result)

    else:

        print(
            "Place sample.jpg inside backend folder"
        )