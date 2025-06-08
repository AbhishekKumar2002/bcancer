from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle
import os

app = FastAPI(title="Breast Cancer Predictor")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class InputData(BaseModel):
    features: List[float]

    class Config:
        schema_extra = {
            "example": {
                "features": [0.106, 1001.0, 17.99, 122.8, 0.3001]
            }
        }

# ✅ Use a safe, non-conflicting name for the model
MODEL_PATH = "cancer_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file 'cancer_model.pkl' not found in project directory.")

with open(MODEL_PATH, "rb") as f:
    breast_cancer_model = pickle.load(f)

print("✅ Loaded model of type:", type(breast_cancer_model))

@app.get("/")
def root():
    return {"message": "✅ Breast Cancer Predictor is up!"}

@app.post("/predict")
def predict(data: InputData):
    try:
        input_array = np.array([data.features], dtype=float)
        print("📥 Features received:", input_array)
        print("🔎 Model type:", type(breast_cancer_model))

        prediction = int(breast_cancer_model.predict(input_array)[0])
        probability = breast_cancer_model.predict_proba(input_array)[0].tolist()

        print("✅ Prediction:", prediction)
        return {
            "prediction": prediction,
            "probability": probability
        }

    except Exception as e:
        print("❌ Error occurred:", e)
        return {"error": str(e)}
