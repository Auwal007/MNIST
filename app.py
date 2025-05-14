# app.py
import base64
import io
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps # Make sure ImageOps is imported
import os
import datetime
import json

# Google Drive API imports
from google.oauth2 import service_account # For service account auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# Initialize Flask app
app = Flask(__name__)

# --- Google Drive Configuration ---
# (Keep your Google Drive configuration as is)
try:
    SERVICE_ACCOUNT_INFO_JSON_STR = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if SERVICE_ACCOUNT_INFO_JSON_STR is None:
        print("CRITICAL: GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable not set.")
        # Fallback for local testing (ensure you have a credentials.json or similar)
        # try:
        #     with open('path_to_your_service_account.json', 'r') as f: # Replace with actual path
        #         SERVICE_ACCOUNT_INFO_JSON_STR = f.read()
        # except FileNotFoundError:
        #     print("Local service account JSON file not found.")
        #     SERVICE_ACCOUNT_INFO_JSON_STR = "{}"
        SERVICE_ACCOUNT_INFO_JSON_STR = "{}" # Default to empty if not found to avoid early crash

    SERVICE_ACCOUNT_INFO = json.loads(SERVICE_ACCOUNT_INFO_JSON_STR)
    DRIVE_FOLDER_ID = os.environ.get('GOOGLE_DRIVE_FOLDER_ID')
    if DRIVE_FOLDER_ID is None:
        print("CRITICAL: GOOGLE_DRIVE_FOLDER_ID environment variable not set.")

    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    creds = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO, scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=creds)
    print("Google Drive service initialized successfully.")
except json.JSONDecodeError as e:
    print(f"CRITICAL: Failed to parse GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")
    drive_service = None
except Exception as e:
    print(f"CRITICAL: Failed to initialize Google Drive service: {e}")
    drive_service = None
# --- End Google Drive Configuration ---


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

##############################
# Your defined apply_threshold function
def apply_threshold_func(image_gray_pil, threshold=128):
    """Applies a simple binary threshold to a grayscale PIL image."""
    # Create a new black and white image of the same mode as input for consistency
    image_binary = Image.new(image_gray_pil.mode, image_gray_pil.size)
    pixels_in = image_gray_pil.load()
    pixels_out = image_binary.load()

    for i in range(image_gray_pil.width):
        for j in range(image_gray_pil.height):
            if pixels_in[i, j] < threshold:
                pixels_out[i, j] = 0  # Black
            else:
                pixels_out[i, j] = 255 # White
    return image_binary
##############################

def preprocess_image(base64_image_data):
    """
    Preprocesses a base64 encoded image string to match MNIST format.
    Args:
        base64_image_data (str): Base64 encoded image string.
    Returns:
        tuple: (flattened_input_for_model, original_pil_image_for_reference)
    """
    global last_numerical_array_for_saving
    global last_viewable_preprocessed_pil_for_saving

    try:
        if ',' in base64_image_data:
            header, encoded = base64_image_data.split(',', 1)
        else:
            encoded = base64_image_data

        image_data = base64.b64decode(encoded)
        original_image = Image.open(io.BytesIO(image_data))

        image_gray = original_image.convert('L')

        # <<< --- ADDED THRESHOLDING STEP --- >>>
        # image_thresholded = apply_threshold_func(image_gray, threshold=128) # Using default threshold 128

        # MNIST typically has white digits on a black background.
        # If your canvas has black drawing on white, inversion is needed.
        # After thresholding, if 0 is black and 255 is white:
        #   - If drawn digit is black (0) on white (255), inverting makes it white (255) on black (0).
        image_inverted = ImageOps.invert(image_gray) # Perform inversion on the thresholded image
        
        image_resized = image_inverted.resize((28, 28), Image.Resampling.LANCZOS)

        # Store this for viewable saving (this is the 28x28 PIL image *before* normalization, but after thresholding and inversion)
        last_viewable_preprocessed_pil_for_saving = image_resized.copy()

        image_np_28x28 = np.array(image_resized) # This is the 28x28 array (0 or 255 after threshold and invert)
        
        # Normalize (important: this is what the model expects)
        # After thresholding and inversion, values will be 0 or 255. Normalizing makes them 0.0 or 1.0.
        #image_np_normalized_28x28 = image_np_28x28 / 255.0

        last_numerical_array_for_saving = image_np_28x28.copy()

        image_flat_for_model = image_np_28x28.flatten()

        return image_flat_for_model.reshape(1, -1), original_image

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        last_numerical_array_for_saving = None
        last_viewable_preprocessed_pil_for_saving = None
        return None, None

def upload_to_drive(filename, file_content_bytes, mimetype, parent_folder_id):
    """Uploads a file (given as bytes) to Google Drive."""
    if not drive_service:
        print("Google Drive service not available. Cannot upload.")
        return None
    try:
        file_metadata = {'name': filename, 'parents': [parent_folder_id]}
        media = MediaIoBaseUpload(io.BytesIO(file_content_bytes), mimetype=mimetype, resumable=True)
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f"Uploaded '{filename}' to Google Drive. File ID: {file.get('id')}")
        return file.get('id')
    except Exception as e:
        print(f"Error uploading '{filename}' to Google Drive: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global last_numerical_array_for_saving
    global last_viewable_preprocessed_pil_for_saving

    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        if 'image_data' not in data:
            return jsonify({'error': 'No image_data found in request'}), 400

        image_b64_from_frontend = data['image_data']
        processed_input_for_model, _ = preprocess_image(image_b64_from_frontend)

        if processed_input_for_model is None: # Removed check for last_numerical_array_for_saving here as it's set in preprocess_image
            return jsonify({'error': 'Image preprocessing failed'}), 400

        prediction_array = model.predict(processed_input_for_model)
        predicted_digit = int(prediction_array[0])

        # --- Logging to Google Drive ---
        if drive_service and DRIVE_FOLDER_ID:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename_base = f"log_{timestamp}"

            if last_numerical_array_for_saving is not None:
                npy_filename = f"{filename_base}_data.npy"
                npy_bytes_io = io.BytesIO()
                np.save(npy_bytes_io, last_numerical_array_for_saving)
                npy_bytes_io.seek(0)
                upload_to_drive(npy_filename, npy_bytes_io.read(), 'application/octet-stream', DRIVE_FOLDER_ID)

            if last_viewable_preprocessed_pil_for_saving is not None:
                png_filename = f"{filename_base}_view.png"
                png_bytes_io = io.BytesIO()
                last_viewable_preprocessed_pil_for_saving.save(png_bytes_io, format='PNG')
                png_bytes_io.seek(0)
                upload_to_drive(png_filename, png_bytes_io.read(), 'image/png', DRIVE_FOLDER_ID)

            metadata_dict = {
                'timestamp': timestamp,
                'predicted_digit': predicted_digit,
                'numerical_data_file': f"{filename_base}_data.npy" if last_numerical_array_for_saving is not None else None,
                'viewable_image_file': f"{filename_base}_view.png" if last_viewable_preprocessed_pil_for_saving is not None else None,
            }
            json_filename = f"{filename_base}_meta.json"
            json_bytes = json.dumps(metadata_dict, indent=4).encode('utf-8')
            upload_to_drive(json_filename, json_bytes, 'application/json', DRIVE_FOLDER_ID)
        else:
            print("Google Drive not configured or available. Skipping log saving.")
        # --- End Logging ---

        last_numerical_array_for_saving = None
        last_viewable_preprocessed_pil_for_saving = None

        return jsonify({'prediction': predicted_digit})

    except Exception as e:
        print(f"Error during prediction: {e}")
        last_numerical_array_for_saving = None
        last_viewable_preprocessed_pil_for_saving = None
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON') or not os.environ.get('GOOGLE_DRIVE_FOLDER_ID'):
        print("---------------------------------------------------------------------------")
        print("WARNING: Google Drive credentials or Folder ID not set as environment variables.")
        # ... (rest of your warning message)
        print("---------------------------------------------------------------------------")

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)