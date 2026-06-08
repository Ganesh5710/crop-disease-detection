from fastapi import FastAPI, UploadFile, File
import shutil
import os

from predict import predict_disease

# ======================================
# FastAPI App
# ======================================

app = FastAPI(
    title="Crop Disease Detection API",
    description="Predict crop diseases using ResNet50",
    version="1.0"
)

# ======================================
# Upload Directory
# ======================================

UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

# ======================================
# Home Route
# ======================================

@app.get("/")
def home():

    return {
        "message": "Crop Disease Detection API Running"
    }

# ======================================
# Prediction Route
# ======================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    result = predict_disease(
        file_path
    )

    return {
        "filename": file.filename,
        "prediction": result["disease"],
        "confidence": result["confidence"]
    }

# ======================================
# Health Route
# ======================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

# ======================================
# Run Locally
# ======================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )