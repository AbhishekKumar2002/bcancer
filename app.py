from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle
import os

app = FastAPI(title="Breast Cancer Predictor")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic input model
class InputData(BaseModel):
    features: List[float]

    class Config:
        schema_extra = {
            "example": {
                "features": [0.106, 1001.0, 17.99, 122.8, 0.3001]
            }
        }

# Load model
MODEL_PATH = "cancer_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model file 'cancer_model.pkl' not found in working directory.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Routes
@app.get("/")
def read_root():
    return {"message": "✅ Breast Cancer Prediction API is live."}

@app.post("/predict")
def predict(data: InputData):
    try:
        input_array = np.array([data.features], dtype=float)
        print("📥 Received input:", input_array)

        prediction = int(model.predict(input_array)[0])
        probability = model.predict_proba(input_array)[0].tolist()

        print("✅ Prediction:", prediction)
        print("📊 Probabilities:", probability)

        return {
            "prediction": prediction,
            "probability": probability
        }

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return {"error": str(e)}
