from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Define path to model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cancer_model1.joblib")

# Load the model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Create the FastAPI app
app = FastAPI()

# Define input schema
class CancerInput(BaseModel):
    features: list[float]  # Expecting a list of 30 feature values

# Define prediction route
@app.post("/predict")
def predict(input_data: CancerInput):
    features = np.array(input_data.features).reshape(1, -1)

    if features.shape[1] != 30:
        raise HTTPException(status_code=400, detail="Exactly 30 features are required.")

    try:
        prediction = model.predict(features)[0]
        result = "Malignant" if prediction == 0 else "Benign"
        return {"prediction": int(prediction), "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
