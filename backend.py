from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import io
import os
import urllib.request

WEIGHTS_URL = "https://huggingface.co/aditi-io/casting-quality-model/resolve/main/casting_quality_model.weights.h5"
WEIGHTS_PATH = "casting_quality_model.weights.h5"

if not os.path.exists(WEIGHTS_PATH):
    print("Downloading weights...")
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
    print("Weights downloaded.")

# Rebuild the exact architecture from model.summary()
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid'),
])

model.load_weights(WEIGHTS_PATH)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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
    return {"label": label, "confidence": round(final_confidence * 100, 2)}
