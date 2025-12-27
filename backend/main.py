from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

# ------------------ App ------------------
app = FastAPI(title="NIDS ML Backend")

# ------------------ Load model once ------------------
MODEL_PATH = os.path.join("models", "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

feature_names = model.feature_names_
feature_count = len(feature_names)

print("Model expects", feature_count, "features")
print("Feature names:", feature_names)

# ------------------ Label mapping ------------------
LABEL_MAP = {
    0: "Benign",
    1: "DDoS Attack"
}

# ------------------ Input schema ------------------
class PredictInput(BaseModel):
    flow_duration: float
    total_fwd_packets: float
    total_backward_packets: float

# ------------------ Routes ------------------
@app.get("/")
def health_check():
    return {"status": "Backend is running and model is loaded"}

@app.post("/predict")
def predict(input_data: PredictInput):
    # initialize all features with 0
    features = [0.0] * feature_count

    # map user inputs to correct feature positions
    features[feature_names.index("Flow Duration")] = input_data.flow_duration
    features[feature_names.index("Total Fwd Packets")] = input_data.total_fwd_packets
    features[feature_names.index("Total Backward Packets")] = input_data.total_backward_packets

    # model expects 2D array
    prediction = model.predict([features])
    label = int(prediction[0])

    return {
        "prediction": label,
        "label": LABEL_MAP.get(label, "Unknown"),
        "note": "Real prediction using partial feature mapping"
    }
