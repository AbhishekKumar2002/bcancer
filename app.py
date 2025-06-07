from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import os
import requests

# Model download details
MODEL_URL = "https://kitish-whatsapp-bot-media.s3.ap-south-1.amazonaws.com/documentMessage_1749283806302.bin"
MODEL_PATH = "model/breast_cancer_model.h5"

# Create model directory and download if not exists
os.makedirs("model", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    with open(MODEL_PATH, "wb") as f:
        f.write(requests.get(MODEL_URL).content)
    print("Model downloaded.")

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# FastAPI app
app = FastAPI()

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request format
class InputData(BaseModel):
    features: list  # list of floats (e.g. 30 values)

@app.get("/")
def root():
    return {"message": "Breast Cancer Prediction API is running."}

@app.post("/predict")
def predict(data: InputData):
    try:
        input_data = np.array([data.features])
        prediction = model.predict(input_data)
        return {"prediction": prediction.tolist()[0]}
    except Exception as e:
        return {"error": str(e)}
