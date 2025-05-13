# app.py
import base64
import io
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
import os
import datetime
import json

# Initialize Flask app
app = Flask(__name__)

# --- Directory for logging deployed model inputs and outputs ---
LOG_DIR = "deployed_model_logs_numerical"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    print(f"Created directory: {LOG_DIR}")

# Subdirectories for numerical data and metadata
NUMERICAL_DATA_LOG_DIR = os.path.join(LOG_DIR, "numerical_input_data")
METADATA_LOG_DIR = os.path.join(LOG_DIR, "metadata")
# Optionally, still save the viewable preprocessed image for quick visual checks
VIEWABLE_PREPROCESSED_IMG_LOG_DIR = os.path.join(LOG_DIR, "viewable_preprocessed_images")


for d in [NUMERICAL_DATA_LOG_DIR, METADATA_LOG_DIR, VIEWABLE_PREPROCESSED_IMG_LOG_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)
# --- End logging directory setup ---


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

# This will store the 28x28 normalized numpy array before flattening, for saving
last_numerical_array_for_saving = None
# This will store the 28x28 PIL image for viewable saving
last_viewable_preprocessed_pil_for_saving = None


def preprocess_image(base64_image_data):
    """
    Preprocesses a base64 encoded image string to match MNIST format.
    Args:
        base64_image_data (str): Base64 encoded image string.
    Returns:
        tuple: (flattened_input_for_model, original_pil_image_for_reference)
               The original_pil_image is just for potential reference.
               The numerical array for saving is handled via a global/accessible variable.
    """
    global last_numerical_array_for_saving
    global last_viewable_preprocessed_pil_for_saving

    try:
        if ',' in base64_image_data: # Check if the header is present
            header, encoded = base64_image_data.split(',', 1)
        else:
            encoded = base64_image_data # Assume no header

        image_data = base64.b64decode(encoded)
        original_image = Image.open(io.BytesIO(image_data))

        image_gray = original_image.convert('L')
        # Ensure you uncomment the next line if your input (e.g., black drawing on white canvas)
        # needs to be inverted to match MNIST (white digit on black background for the model)
        image_gray = ImageOps.invert(image_gray)
        image_resized = image_gray.resize((28, 28), Image.Resampling.LANCZOS)

        # Store this for viewable saving (this is the 28x28 PIL image *before* normalization)
        last_viewable_preprocessed_pil_for_saving = image_resized.copy()

        image_np_28x28 = np.array(image_resized) # This is the 28x28 array (0-255)
        
        # Normalize (important: this is what the model expects)
        #image_np_normalized_28x28 = image_np_28x28 / 255.0

        # Store the 28x28 *normalized* numpy array for saving
        last_numerical_array_for_saving = image_np_28x28.copy()

        # Flatten for the model input
        image_flat_for_model = image_np_28x28.flatten()

        return image_flat_for_model.reshape(1, -1), original_image

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        last_numerical_array_for_saving = None
        last_viewable_preprocessed_pil_for_saving = None
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global last_numerical_array_for_saving # Access the 28x28 numerical array
    global last_viewable_preprocessed_pil_for_saving

    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        if 'image_data' not in data:
             return jsonify({'error': 'No image_data found in request'}), 400

        image_b64_from_frontend = data['image_data']
        # The second returned value (original_image) from preprocess_image is not strictly needed here
        # unless you wanted to save it in its original form too.
        processed_input_for_model, _ = preprocess_image(image_b64_from_frontend)

        if processed_input_for_model is None or last_numerical_array_for_saving is None:
            return jsonify({'error': 'Image preprocessing failed'}), 400

        prediction_array = model.predict(processed_input_for_model)
        predicted_digit = int(prediction_array[0])

        # --- Logging the numerical data and prediction ---
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename_base = f"log_{timestamp}" # Consistent base for related log files

        # 1. Save the 28x28 numerical array data
        # This is the key data you want for re-feeding to your local model
        if last_numerical_array_for_saving is not None:
            numerical_data_path = os.path.join(NUMERICAL_DATA_LOG_DIR, f"{filename_base}_data.npy")
            try:
                np.save(numerical_data_path, last_numerical_array_for_saving)
            except Exception as e:
                print(f"Error saving numerical data: {e}")

        # 2. (Optional but recommended) Save the viewable 28x28 preprocessed image
        # This helps visually confirm what the numerical array represents.
        if last_viewable_preprocessed_pil_for_saving is not None:
            viewable_img_path = os.path.join(VIEWABLE_PREPROCESSED_IMG_LOG_DIR, f"{filename_base}_view.png")
            try:
                last_viewable_preprocessed_pil_for_saving.save(viewable_img_path)
            except Exception as e:
                print(f"Error saving viewable preprocessed image: {e}")

        # 3. Save metadata (prediction and filename reference)
        metadata = {
            'timestamp': timestamp,
            'predicted_digit': predicted_digit,
            'numerical_data_file': f"{filename_base}_data.npy", # Reference to the .npy file
            'viewable_image_file': f"{filename_base}_view.png" if last_viewable_preprocessed_pil_for_saving is not None else None,
        }
        metadata_path = os.path.join(METADATA_LOG_DIR, f"{filename_base}_meta.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            print(f"Error saving metadata: {e}")
        # --- End Logging ---

        # Reset global variables for the next request to avoid using stale data
        last_numerical_array_for_saving = None
        last_viewable_preprocessed_pil_for_saving = None

        return jsonify({'prediction': predicted_digit})

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Reset global variables in case of error during processing too
        last_numerical_array_for_saving = None
        last_viewable_preprocessed_pil_for_saving = None
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) # Set debug=False for actual deployment