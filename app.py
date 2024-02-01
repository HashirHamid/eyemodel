from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from pydantic import BaseModel
from joblib import load 
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os
import tempfile

app = FastAPI()

# Function to load the model from a URL
def load_model_from_url(model_url):
    response = requests.get(model_url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        try:
            model = load_model(temp_file_path)  # Ensure you have the correct import
            print("Model loaded successfully.")
            return model
        finally:
            temp_file.close()
            os.remove(temp_file_path)
    else:
        raise HTTPException(status_code=500, detail=f"Failed to download the model. Status code: {response.status_code}")

# Load the model at application startup
model_url = 'https://epsoldevops.com/ML/model.h5'
model = load_model_from_url(model_url)

# Define the request model using Pydantic
class PredictionRequest(BaseModel):
    image_path: str

# Define the image preprocessing function
def preprocess_image(image_path):
    try:
        if image_path.startswith('http'):
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(image_path)

        img = img.resize((224, 224))  # Adjust the size as needed
        img = np.array(img) / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading image: {str(e)}")

# Prediction route
@app.post('/predict')
def predict(request: PredictionRequest):
    try:
        # Preprocess the image
        processed_image = preprocess_image(request.image_path)

        # Make a prediction
        prediction = model.predict(processed_image)

        # Assuming binary classification, convert the prediction to a human-readable label
        label = "Glaucoma" if prediction > 0.5 else "Not Glaucoma"

        response = {'prediction': label}
        return JSONResponse(content=jsonable_encoder(response), status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Index route
@app.get("/")
def index():
    return {"details": "Hello!"}
