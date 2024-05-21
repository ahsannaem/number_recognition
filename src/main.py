import os
import pickle
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img = Image.open(file)
        img_array = preprocess_image(img)
        img_array = img_array.reshape(1, -1)
        model = load_model_from_pickle()
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction[0])
        return jsonify({'predicted_digit': int(predicted_digit), 'prediction': prediction.tolist()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def load_model_from_pickle():
    with open("./model.pkl", 'rb') as f:
        model = pickle.load(f)
    return model

def main():
    app.run(port=int(os.environ.get('PORT', 5000)), debug=True)

def preprocess_image(image):
    # Resize the image to 28x28
    image = image.resize((28, 28))
    # Convert image to grayscale
    image = image.convert('L')
    # Flatten the image array
    image_array = np.array(image).flatten()
    # Normalize pixel values to be between 0 and 1
    image_array = image_array / 255.0
    # Create DataFrame
    mnist_df = pd.DataFrame([image_array])
    # Create DataFrame with named features
    mnist_df = pd.DataFrame([image_array], columns=[f'pixel_{i}' for i in range(image_array.size)])
    # Convert DataFrame to NumPy array
    mnist_array_flattened = mnist_df.values.flatten()
    mnist_array_flattened[mnist_array_flattened == 1] = 0
    mnist_array_flattened = mnist_array_flattened * 1000
    return mnist_array_flattened


if __name__ == "__main__":
    main()



