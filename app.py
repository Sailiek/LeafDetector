from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from leaf_processor2 import LeafProcessor
from plant_detector import PlantDetector

app = Flask(__name__)

# Initialize plant detector
plant_detector = PlantDetector(model_path='models/best.keras')

# Configurations
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image(image):
    """Resize image to 800x600 while maintaining aspect ratio with padding"""
    h, w = image.shape[:2]
    
    # Determine if image is portrait or landscape
    if h > w:  # Portrait
        # Rotate to landscape if needed
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        h, w = image.shape[:2]
    
    # Calculate scaling factor
    scale = min(800/w, 600/h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Calculate padding
    pad_w = (800 - new_w) // 2
    pad_h = (600 - new_h) // 2
    
    # Add padding
    padded = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w, 
                               cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return padded

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Main page
@app.route('/')
def index():
    return render_template('index.html')

# Process Image page
@app.route('/process-image', methods=['GET', 'POST'])
def process_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process image based on selected filter
            filter_type = request.form.get('filter')
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            # Ensure grayscale image
            image = LeafProcessor.to_grayscale(image)
            
            # Process image based on filter type
            if filter_type in ['gaussian', 'median', 'bilateral', 'nlm']:
                processed = LeafProcessor.reduce_noise(image, method=filter_type)
            elif filter_type == 'canny':
                processed = LeafProcessor.detect_edges_canny(image)
            elif filter_type == 'sobel':
                processed = LeafProcessor.detect_edges_sobel(image)
            elif filter_type == 'prewitt':
                processed = LeafProcessor.detect_edges_prewitt(image)
            elif filter_type == 'otsu':
                processed = LeafProcessor.segment_otsu(image)
            elif filter_type == 'adaptive':
                processed = LeafProcessor.segment_adaptive(image)
            else:
                return jsonify({'error': 'Invalid filter type'}), 400
                
            # Save processed image
            processed_filename = f'processed_{filename}'
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            cv2.imwrite(processed_path, processed)
            
            return jsonify({
                'original': filename,
                'processed': processed_filename
            })
            
    return render_template('process_image.html')

# Plant Detection page
@app.route('/plant-detection', methods=['GET', 'POST'])
def plant_detection_page():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and preprocess image
            image = cv2.imread(filepath)
            resized = resize_image(image)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Apply selected edge detection
            edge_method = request.form.get('edge_method', 'sobel')
            if edge_method == 'sobel':
                edges = LeafProcessor.detect_edges_sobel(gray)
            elif edge_method == 'canny':
                edges = LeafProcessor.detect_edges_canny(gray)
            elif edge_method == 'prewitt':
                edges = LeafProcessor.detect_edges_prewitt(gray)
            else:
                return jsonify({'error': 'Invalid edge detection method'}), 400
                
            # Perform plant detection
            detection_result = plant_detector.detect(edges)
            
            # Save processed image
            processed_filename = f'processed_{filename}'
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            cv2.imwrite(processed_path, edges)
            
            return jsonify({
                'original': filename,
                'processed': processed_filename,
                'detection': {
                    'plant_type': detection_result['class_name'],
                    'confidence': f"{detection_result['confidence']:.2%}"
                }
            })
            
    return render_template('plant_detection.html')

# Tea Leaf Age Detection page
@app.route('/tea-leaf-age')
def tea_leaf_age():
    return render_template('tea_leaf_age.html')

if __name__ == '__main__':
    app.run(debug=True)
