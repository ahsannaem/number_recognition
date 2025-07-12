import os
import pickle
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing import Union

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index() -> str:
    """Renders the homepage."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file() -> Union[tuple[dict, int], tuple[dict, int]]:
    """Handles file upload and prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        filename = secure_filename(file.filename)
        image = Image.open(file)
        processed_array = preprocess_image(image)

        model = load_model_from_pickle("./model.pkl")
        processed_array = processed_array.reshape(1, -1)
        prediction = model.predict(processed_array)

        predicted_digit = int(np.argmax(prediction[0]))

        return jsonify({
            'predicted_digit': predicted_digit,
            'prediction': prediction.tolist()
        }), 200

    except (OSError, ValueError) as img_error:
        return jsonify({'error': f'Invalid image file: {str(img_error)}'}), 400
    except FileNotFoundError:
        return jsonify({'error': 'Model file not found'}), 500
    except pickle.PickleError:
        return jsonify({'error': 'Failed to load model'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesses the image:
    - Resize to 28x28
    - Convert to grayscale
    - Normalize and flatten
    """
    image = image.resize((28, 28)).convert('L')
    image_array = np.array(image, dtype=np.float32) / 255.0
    flat_array = image_array.flatten()

    # Create DataFrame with named columns
    df = pd.DataFrame([flat_array], columns=[f'pixel_{i}' for i in range(flat_array.size)])
    processed_array = df.values.flatten()

    # Post-processing logic (custom business logic)
    processed_array[processed_array == 1.0] = 0
    processed_array *= 1000

    return processed_array


def load_model_from_pickle(path: str) -> BaseEstimator:
    """Safely loads a scikit-learn model from a pickle file."""
    with open(path, 'rb') as f:
        model = pickle.load(f)

    if not hasattr(model, "predict"):
        raise ValueError("Loaded object is not a valid model with a predict method.")

    return model


def main() -> None:
    """Runs the Flask application."""
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


if __name__ == "__main__":
    main()
