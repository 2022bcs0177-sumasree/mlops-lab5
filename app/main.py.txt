from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/predict")
def predict():
    dummy_input = np.zeros((1, 13))
    prediction = model.predict(dummy_input)[0]

    return {
        "name": "Allu Suma Sree",
        "roll_no": "2022BC",
        "wine_quality": int(prediction)
    }
