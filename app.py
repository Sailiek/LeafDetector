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


# Add these imports at the top of your Flask file
from flask import send_from_directory
import os

# Add this new route to handle downloads
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

# Process Image route
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
           
            # Load and preprocess image
            image = cv2.imread(filepath)
            # Standardize image size
            image = resize_image(image)
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Process image based on selected filter
            filter_type = request.form.get('filter')
            
            if filter_type == 'sobel':
                processed = LeafProcessor.detect_edges_sobel(gray)
            elif filter_type == 'canny':
                processed = LeafProcessor.detect_edges_canny(gray)
            elif filter_type == 'prewitt':
                processed = LeafProcessor.detect_edges_prewitt(gray)
            elif filter_type in ['gaussian', 'median', 'bilateral', 'nlm']:
                processed = LeafProcessor.reduce_noise(gray, method=filter_type)
            elif filter_type == 'otsu':
                processed = LeafProcessor.segment_otsu(gray)
            elif filter_type == 'adaptive':
                processed = LeafProcessor.segment_adaptive(gray)
            else:
                return jsonify({'error': 'Invalid filter type'}), 400
               
            # Save processed image
            processed_filename = f'processed_{filename}'
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            cv2.imwrite(processed_path, processed)
           
            return jsonify({
                'original': filename,
                'processed': processed_filename,
                'download_url': f'/download/{processed_filename}'
            })
           
    return render_template('process_image.html')

# Plant Detection route
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
           
            # Load and preprocess image - using same preprocessing as process_image
            image = cv2.imread(filepath)
            # Standardize image size
            image = resize_image(image)
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
           
            # Apply selected edge detection using same methods as process_image
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

# Add this helper function if you don't already have it
def resize_image(image, target_size=(800, 600)):
    """Resize image while maintaining aspect ratio"""
    height, width = image.shape[:2]
    aspect = width / height
    
    if aspect > target_size[0] / target_size[1]:
        # Width is the limiting factor
        new_width = target_size[0]
        new_height = int(new_width / aspect)
    else:
        # Height is the limiting factor
        new_height = target_size[1]
        new_width = int(new_height * aspect)
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
# Age Detection page
@app.route('/leaf-age', methods=['GET', 'POST'])
def leaf_age():
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
            
            # Load and process image
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'error': 'Could not load image'}), 400
                
            # Step 1: Preprocessing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Step 2: Otsu's Thresholding
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Step 3: Morphological Operations
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Close small holes
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Remove small noise
            
            # Step 4: Segmentation and Contour Extraction
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Step 5: Find and Keep Only the Largest Contour
            min_area = 1000  
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

            results = []

            # Step 6: Contour Refinement and Analysis
            for i, contour in enumerate(filtered_contours):
                # Approximate the contour
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

                # Calculate area and perimeter
                pixel_to_mm = 0.0265  # Conversion factor for pixels to cm
                area = cv2.contourArea(contour) * (pixel_to_mm**2)  # Convert to cm²
                perimeter = cv2.arcLength(contour, True) * pixel_to_mm  # Convert to cm

                # Approximation of growth rate for Acer rubrum
                growth_rate = 40  # Growth rate in mm²/day
                estimated_age_days = area / growth_rate
                results.append({
                    'contour_id': i + 1,
                    'area': f"{area:.2f}",
                    'perimeter': f"{perimeter:.2f}",
                    'estimated_age': f"{estimated_age_days:.2f}"
                })    
            
            # Save processed images
            binary_filename = f'binary_{filename}'
            result_filename = f'result_{filename}'
            
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], binary_filename), binary)
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], result_filename), image)
            
            return jsonify({
                'species': "Acer rubrum",
                'binary_image': binary_filename,
                'result_image': result_filename,
                'results': results
            })
            
    return render_template('leaf_age.html')
    
if __name__ == '__main__':
    app.run(debug=True)
