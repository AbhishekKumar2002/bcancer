from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import os

# Load the pickle model
MODEL_PATH = "cancer_model1.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Please upload 'cancer_model.pkl' to the 'model' folder.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    features: list

@app.get("/")
def read_root():
    return {"message": "Breast Cancer Prediction API is running."}

@app.post("/predict")
def predict(data: InputData):
    try:
        input_array = np.array([data.features], dtype=float)
        prediction = int(model.predict(input_array)[0])
        probability = model.predict_proba(input_array)[0].tolist()
        return {"prediction": prediction, "probability": probability}
    except Exception as e:
        return {"error": str(e)}
