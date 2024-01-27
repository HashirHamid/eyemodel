from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras.models import load_model
app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

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
        return str(e), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image_path from the POST request
        image_path = request.json['image_path']

        # Preprocess the image
        processed_image = preprocess_image(image_path)

        # Make a prediction
        prediction = model.predict(processed_image)

        # Assuming binary classification, convert the prediction to a human-readable label
        label = "Glaucoma" if prediction > 0.5 else "Not Glaucoma"

        response = {'prediction': label}
        return jsonify(response), 200

    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
