// static/script.js
document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('drawing-canvas');
    const ctx = canvas.getContext('2d');
    const clearButton = document.getElementById('clear-canvas');
    const predictDrawingButton = document.getElementById('predict-drawing');
    const predictionResult = document.getElementById('prediction-result');

    const imageUpload = document.getElementById('image-upload');
    const imagePreview = document.getElementById('image-preview');
    const imagePreviewArea = document.getElementById('image-preview-area');

    const startCameraButton = document.getElementById('start-camera');
    const cameraArea = document.getElementById('camera-area');
    const videoElement = document.getElementById('camera-stream');
    const captureCanvas = document.getElementById('capture-canvas'); // Hidden canvas for capture
    const captureButton = document.getElementById('capture-photo');
    const stopCameraButton = document.getElementById('stop-camera');

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let stream = null; // To hold the camera stream

    // --- Canvas Drawing Logic ---

    // Set initial canvas properties (white background, thick black line)
    function initializeCanvas() {
        ctx.fillStyle = 'white'; // Background color
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'black'; // Ink color
        ctx.lineWidth = 5; // Thickness of the line
        ctx.lineCap = 'round'; // Rounded ends for lines
        ctx.lineJoin = 'round'; // Rounded corners where lines meet
        ctx.globalAlpha = 1;
        ctx.globalCompositeOperation = 'source-over'
        predictionResult.textContent = '?'; // Reset prediction on clear
    }

    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = getEventCoordinates(e);
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault(); // Prevent scrolling while drawing

        const [currentX, currentY] = getEventCoordinates(e);

        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(currentX, currentY);
        ctx.stroke();

        [lastX, lastY] = [currentX, currentY];
    }

    function stopDrawing() {
        isDrawing = false;
        ctx.beginPath(); // Reset the path to prevent unwanted lines
    }

    function getEventCoordinates(e) {
        let x, y;
        const rect = canvas.getBoundingClientRect(); // Get canvas position
        if (e.touches) { // Touch event
            x = e.touches[0].clientX - rect.left;
            y = e.touches[0].clientY - rect.top;
        } else { // Mouse event
            x = e.clientX - rect.left;
            y = e.clientY - rect.top;
        }
        return [x, y];
    }

    // Add event listeners for drawing (mouse and touch)
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing); // Stop if mouse leaves canvas

    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('touchmove', draw);
    canvas.addEventListener('touchend', stopDrawing);

    clearButton.addEventListener('click', initializeCanvas);

    // --- Image Upload Logic ---

    imageUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                imagePreviewArea.style.display = 'block';
                 // Automatically predict when image is loaded
                sendImageForPrediction(e.target.result);
            }
            reader.readAsDataURL(file);
            stopCameraStream(); // Stop camera if running
            cameraArea.style.display = 'none';
        }
    });

    // --- Camera Logic ---

    startCameraButton.addEventListener('click', async () => {
        stopCameraStream(); // Ensure previous stream is stopped
        imagePreview.style.display = 'none'; // Hide uploaded image preview
        imagePreviewArea.style.display = 'none';
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoElement.srcObject = stream;
            cameraArea.style.display = 'flex'; // Show camera controls
        } catch (err) {
            console.error("Error accessing camera: ", err);
            alert("Could not access camera. Please ensure permission is granted and using HTTPS.");
            cameraArea.style.display = 'none';
        }
    });

    stopCameraButton.addEventListener('click', stopCameraStream);

    function stopCameraStream() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            videoElement.srcObject = null;
            stream = null;
            cameraArea.style.display = 'none';
        }
    }

    captureButton.addEventListener('click', () => {
        if (!stream) return;
        const captureCtx = captureCanvas.getContext('2d');
        // Set capture canvas dimensions based on video aspect ratio
        const aspectRatio = videoElement.videoWidth / videoElement.videoHeight;
        captureCanvas.width = videoElement.videoWidth;
        captureCanvas.height = videoElement.videoHeight;

        // Draw the current video frame onto the hidden canvas
        captureCtx.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);

        // Get the image data from the capture canvas
        const imageDataUrl = captureCanvas.toDataURL('image/png');

        // Display captured image in the preview area
        imagePreview.src = imageDataUrl;
        imagePreview.style.display = 'block';
        imagePreviewArea.style.display = 'block';

        // Send captured image for prediction
        sendImageForPrediction(imageDataUrl);

        // Optionally stop the camera after capture
        // stopCameraStream();
    });


    // --- Prediction Logic ---

    predictDrawingButton.addEventListener('click', () => {
        const imageDataUrl = canvas.toDataURL('image/png'); // Get image data from canvas
        sendImageForPrediction(imageDataUrl);
    });

    async function sendImageForPrediction(imageDataUrl) {
        predictionResult.textContent = 'Predicting 😊...'; // Indicate processing
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image_data: imageDataUrl }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const result = await response.json();
            if (result.prediction !== undefined) {
                predictionResult.textContent = result.prediction;
            } else if (result.error) {
                 predictionResult.textContent = `Error: ${result.error}`;
            } else {
                predictionResult.textContent = 'Error';
            }
        } catch (error) {
            console.error('Error during prediction fetch:', error);
            predictionResult.textContent = `Error: ${error.message}`;
        }
    }

    // --- Initialize ---
    initializeCanvas(); // Set up the canvas on page load
});