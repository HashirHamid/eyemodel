from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import requests
import numpy as np
from io import BytesIO
import os
import uvicorn
app = Flask(__name__)

# Function to load the model from a URL
def load_model_from_url(model_url):
    response = requests.get(model_url)
    if response.status_code == 200:
        with open("model.h5", "wb") as model_file:
            model_file.write(response.content)

        try:
            model = load_model("model.h5")  # Ensure you have the correct import
            print("Model loaded successfully.")
            return model
        finally:
            os.remove("model.h5")
    else:
        return jsonify({"error": f"Failed to download the model. Status code: {response.status_code}"}), 500

# Load the model at application startup
model_url = 'https://epsoldevops.com/ML/model.h5'
model = load_model_from_url(model_url)

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
        return jsonify({"error": f"Error loading image: {str(e)}"}), 400

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image path from request
        image_path = request.json.get('image_path')

        # Preprocess the image
        processed_image = preprocess_image(image_path)

        # Make a prediction
        prediction = model.predict(processed_image)

        # Assuming binary classification, convert the prediction to a human-readable label
        label = "Glaucoma" if prediction > 0.5 else "Not Glaucoma"

        response = {'prediction': label}
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Index route
@app.route("/")
def index():
    return jsonify({"details": "Hello!"})



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4, loop="asyncio")
