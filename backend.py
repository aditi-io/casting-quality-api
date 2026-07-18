from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import urllib.request

MODEL_URL = "https://huggingface.co/aditi-io/casting-quality-model/resolve/main/casting_quality_model.keras"
MODEL_PATH = "casting_quality_model.keras"

# Download the model if it isn't already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess(image: Image.Image):
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    processed = preprocess(image)

    prediction = model.predict(processed)
    confidence = float(prediction[0][0])

    is_ok = confidence >= 0.5
    label = "OK Front" if is_ok else "Defective Front"
    final_confidence = confidence if is_ok else 1 - confidence

    return {
        "label": label,
        "confidence": round(final_confidence * 100, 2)
    }