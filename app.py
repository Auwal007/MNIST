# app.py
import base64
import io
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained Scikit-learn model
try:
    model = joblib.load('mnist_model.joblib')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: mnist_model.joblib not found. Make sure the model file is in the correct directory.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(base64_image_data):
    """
    Preprocesses a base64 encoded image string to match MNIST format.
    Args:
        base64_image_data (str): Base64 encoded image string (with data URI header).
    Returns:
        np.array: Flattened 784-element NumPy array, or None if processing fails.
    """
    try:
        # Remove the header ("data:image/png;base64,")
        if ',' in base64_image_data:
            header, encoded = base64_image_data.split(',', 1)
        else:
            encoded = base64_image_data # Assume no header if comma not found

        # Decode base64 string
        image_data = base64.b64decode(encoded)

        # Open the image using Pillow
        image = Image.open(io.BytesIO(image_data))

        # 1. Convert to grayscale
        image = image.convert('L')

        # 2. Invert colors (MNIST has white digits on black background)
        #    Pillow's 'L' mode has 0 as black, 255 as white. We want the digit
        #    to be white (closer to 1 after normalization).
        #    If your drawing canvas has black ink on white background,
        #    inverting might be needed. Test this based on your canvas output.
        image = ImageOps.invert(image) # Uncomment if your input is black digit on white bg

        # 3. Resize to 28x28 pixels
        #    Use ANTIALIAS for better quality resizing
        image = image.resize((28, 28), Image.Resampling.LANCZOS) # Or Image.ANTIALIAS in older Pillow

        # 4. Convert image data to numpy array
        image_np = np.array(image)

        # 5. Normalize pixel values to be between 0 and 1
        # image_np = image_np / 255.0

        # 6. Flatten the 28x28 image into a 784-element vector
        image_flat = image_np.flatten()

        # Ensure the shape is correct (1, 784) for sklearn model prediction
        # Sklearn models often expect a 2D array where each row is a sample
        return image_flat.reshape(1, -1)

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receive image data, preprocess, predict, and return result."""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        if 'image_data' not in data:
             return jsonify({'error': 'No image_data found in request'}), 400

        image_b64 = data['image_data']

        # Preprocess the image
        processed_image = preprocess_image(image_b64)

        if processed_image is None:
            return jsonify({'error': 'Image preprocessing failed'}), 400

        # Make prediction
        prediction = model.predict(processed_image)

        # Return the prediction as JSON
        # prediction is likely a numpy array, get the first element
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    # Use port 5000 by default, suitable for many deployment platforms
    # Use 0.0.0.0 to make it accessible externally (needed for deployment)
    # Set debug=False for production/deployment
    app.run(host='0.0.0.0', port=5000, debug=True) # Set debug=False for production