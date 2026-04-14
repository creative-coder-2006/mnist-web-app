from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from scipy.ndimage import center_of_mass

app = Flask(__name__)

# Load model
try:
    model = tf.keras.models.load_model('mnist_model.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(image):
    # Ensure image is in grayscale ('L')
    if image.mode != 'L':
        image = image.convert('L')
    
    img_array = np.array(image)
    non_zero_coords = np.argwhere(img_array > 0)
    if len(non_zero_coords) == 0:
        return np.zeros((1, 28, 28, 1))
        
    y_min, x_min = non_zero_coords.min(axis=0)
    y_max, x_max = non_zero_coords.max(axis=0)
    
    cropped = image.crop((x_min, y_min, x_max, y_max))
    
    width, height = cropped.size
    if width == 0 or height == 0:
         return np.zeros((1, 28, 28, 1))

    if width > height:
        new_w = 20
        new_h = int(20 * (height / width))
    else:
        new_h = 20
        new_w = int(20 * (width / height))
    
    new_w = max(1, new_w)
    new_h = max(1, new_h)
        
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    temp_obj = np.array(resized)
    cy, cx = center_of_mass(temp_obj)
    if np.isnan(cy) or np.isnan(cx):
        cy, cx = new_h / 2.0, new_w / 2.0
        
    start_x = int(round(14.0 - cx))
    start_y = int(round(14.0 - cy))
    
    start_x = max(0, min(28 - new_w, start_x))
    start_y = max(0, min(28 - new_h, start_y))
    
    final_img = Image.new("L", (28, 28), 0)
    final_img.paste(resized, (start_x, start_y))
    
    final_array = np.array(final_img).astype('float32') / 255.0
    return final_array.reshape(1, 28, 28, 1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data'}), 400
        
    img_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(img_data)
    
    # Open the image using Pillow. The canvas drawn image is black with white strokes
    image = Image.open(io.BytesIO(image_bytes))
    
    processed = preprocess_image(image)
    if np.max(processed) == 0:
        return jsonify({'error': 'Canvas is empty. Please draw a digit.'}), 400
        
    pred = model.predict(processed)
    digit = int(np.argmax(pred[0]))
    confidence = float(np.max(pred[0]))
    
    return jsonify({
        'digit': digit,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
