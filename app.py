from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle
import os

# Define the app
app = FastAPI(title="Breast Cancer Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema
class InputData(BaseModel):
    features: List[float]

    class Config:
        schema_extra = {
            "example": {
                "features": [0.106, 1001.0, 17.99, 122.8, 0.3001]
            }
        }

# Load the model once at startup
MODEL_PATH = "cancer_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model file 'cancer_model.pkl' not found.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Root route
@app.get("/")
def read_root():
    return {"message": "API is up and running 🎉"}

# Prediction route
@app.post("/predict")
def predict(data: InputData):
    try:
        input_array = np.array([data.features], dtype=float)
        print("📥 Received:", input_array)

        prediction = int(model.predict(input_array)[0])
        probability = model.predict_proba(input_array)[0].tolist()

        return {
            "prediction": prediction,
            "probability": probability
        }

    except Exception as e:
        print("❌ Error:", str(e))
        return {"error": str(e)}
