from fastapi import FastAPI 
from pydantic import BaseModel 
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

model_path = os.path.join(os.path.dirname(__file__), '../../models/classifier.pkl')
preprocessor_path = os.path.join(os.path.dirname(__file__), '../../models/preprocessor.pkl')

with open(preprocessor_path,"rb") as f:
    scaler=pickle.load(f)

with open(model_path,"rb") as m:
    model=pickle.load(m)

app=FastAPI()

origins=["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IrisFeatures(BaseModel):
    sepal_length:float 
    sepal_width:float 
    petal_length:float 
    petal_width:float 

@app.get("/")
async def index():
    return {"Msg":"Welcome to the iris-classifier. Move to the Predict Route"}

@app.post("/predict")
async def make_predictions(request:IrisFeatures):
    X=np.array([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]])

    X_scaled=scaler.transform(X)
    predictions=model.predict(X_scaled)

    return {"Predictions":int(predictions[0])}

