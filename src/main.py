import os
import pickle
import sklearn

from flask import Flask, send_file

app = Flask(__name__)

@app.route("/")
def index():
    return send_file('index.html')

@app.route("/predict")
def predict():
    model = load_model_from_pickle()
    return f"{model}"

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    # Get the uploaded file
    file = request.files['file']
    
    # Check if the file is an image
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Preprocess the image
    img = Image.open(file)
    img_array = preprocess_image(img)
    model = load_model_from_pickle()
    # Predict the digit
    prediction = model.predict(img_array)
    
    # Get the predicted digit
    predicted_digit = np.argmax(prediction)
    
    return jsonify({'predicted_digit': int(predicted_digit)}), 200

def load_model_from_pickle():
    with open("/home/user/mlapp/src/model.pkl", 'rb') as f:
        model = pickle.load(f)
    return model


def main():
    app.run(port=int(os.environ.get('PORT', 80)))


if __name__ == "__main__":
    main()
