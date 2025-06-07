from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import os
import requests

# Constants
MODEL_URL = "https://kitish-whatsapp-bot-media.s3.ap-south-1.amazonaws.com/documentMessage_1749283806302.bin"
MODEL_PATH = "model/breast_cancer_model.pkl"

# Create model directory & download if not exists
os.makedirs("model", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    with open(MODEL_PATH, "wb") as f:
        f.write(requests.get(MODEL_URL).content)
    print("Model downloaded.")

# Load the pickle model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class InputData(BaseModel):
    features: list  # e.g. [17.99, 10.38, ..., 0.1189]

@app.get("/")
def root():
    return {"message": "Breast Cancer Prediction API is running."}

@app.post("/predict")
def predict(data: InputData):
    try:
        input_array = np.array([data.features])
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0].tolist()
        return {
            "prediction": int(prediction),
            "probability": probability
        }
    except Exception as e:
        return {"error": str(e)}
