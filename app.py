from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from leaf_processor import ImagePreprocessor, LinearNoiseReduction, NonLinearNoiseReduction, EdgeDetector, Segmentation, ContourRefinement
from plant_detector import PlantDetector

app = Flask(__name__)

# Initialize plant detector
plant_detector = PlantDetector(model_path='models/best.keras')

# Configurations
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle image upload
        file = request.files['image']  # Ensure this matches the form input `name`
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('stages', filename=filename))
    return render_template('index.html')

# Dictionary to store processing state for each image
processing_states = {}

# Route for stages
@app.route('/stages/<filename>', methods=['GET', 'POST'])
def stages(filename):
    if request.method == 'POST':
        # Get the current stage being processed
        current_stage = request.form.get('stage')
        
        # Initialize state for this image if it doesn't exist
        if filename not in processing_states:
            processing_states[filename] = {
                'original_image': None,
                'current_image': None,
                'noise_cleaning': None,
                'edge_detection': None,
                'segmentation': None,
                'contour_refinement': None
            }
        
        # Load the image if not already in state
        if processing_states[filename]['original_image'] is None:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image = cv2.imread(image_path)
            gray = ImagePreprocessor.to_grayscale(image)
            processing_states[filename]['original_image'] = gray
            processing_states[filename]['current_image'] = gray

        # Get the current processing state
        processed_image = processing_states[filename]['current_image']

        # Process current stage
        if current_stage == 'noise_cleaning':
            filter_category = request.form.get('noise_cleaning')
            input_image = processing_states[filename]['original_image']
            
            if filter_category == 'linear':
                linear_filter = request.form.get('linear_filter_type')
                if linear_filter == 'gaussian':
                    processed_image = LinearNoiseReduction.gaussian_filter(input_image)
                elif linear_filter == 'mean':
                    processed_image = LinearNoiseReduction.mean_filter(input_image)
                elif linear_filter == 'laplacian':
                    processed_image = LinearNoiseReduction.laplacian_filter(input_image)
                elif linear_filter == 'butterworth':
                    processed_image = LinearNoiseReduction.butterworth_filter(input_image)
            
            elif filter_category == 'nonlinear':
                nonlinear_filter = request.form.get('nonlinear_filter_type')
                if nonlinear_filter == 'median':
                    processed_image = NonLinearNoiseReduction.median_filter(input_image)
                elif nonlinear_filter == 'bilateral':
                    processed_image = NonLinearNoiseReduction.bilateral_filter(input_image)
                elif nonlinear_filter == 'nlm':
                    processed_image = NonLinearNoiseReduction.nlm_filter(input_image)
                elif nonlinear_filter == 'adaptive_median':
                    processed_image = NonLinearNoiseReduction.adaptive_median_filter(input_image)
            
            processing_states[filename]['noise_cleaning'] = filter_category
            
        elif current_stage == 'edge_detection':
            # Use result from noise cleaning if it exists, otherwise use original
            input_image = processed_image
            if processing_states[filename]['noise_cleaning']:
                input_image = processing_states[filename]['current_image']
                
            method = request.form.get('edge_detection')
            if method == 'canny':
                processed_image = EdgeDetector.canny(input_image)
            elif method == 'sobel':
                processed_image = EdgeDetector.sobel(input_image)
            elif method == 'prewitt':
                processed_image = EdgeDetector.prewitt(input_image)
            elif method == 'log':
                processed_image = EdgeDetector.log(input_image)
            processing_states[filename]['edge_detection'] = method
            
        elif current_stage == 'segmentation':
            # Use the edge-detected image for segmentation if available, otherwise use noise-cleaned
            if processing_states[filename]['edge_detection']:
                input_image = processing_states[filename]['current_image']
            elif processing_states[filename]['noise_cleaning']:
                # Get the noise-cleaned image state
                noise_cleaned = processing_states[filename]['current_image']
                # Apply light Gaussian blur for better segmentation
                input_image = cv2.GaussianBlur(noise_cleaned, (3,3), 0)
            else:
                # If no previous processing, use original with preprocessing
                input_image = cv2.GaussianBlur(processing_states[filename]['original_image'], (3,3), 0)
            method = request.form.get('segmentation')
            if method == 'thresholding':
                # Get thresholding parameters
                block_size = int(request.form.get('block_size', 35))
                c_value = int(request.form.get('c_value', 5))
                processed_image = Segmentation.otsu(input_image, block_size, c_value)
            elif method == 'active-contours':
                # Get active contours parameters
                alpha = float(request.form.get('alpha', 10)) / 1000  # Convert from 0-100 to 0-0.1
                beta = float(request.form.get('beta', 100)) / 1000   # Convert from 0-500 to 0-0.5
                iterations = int(request.form.get('iterations', 200))
                
                height, width = input_image.shape
                # Create initial contour as 80% of image dimensions
                margin_x = int(width * 0.1)  # 10% margin from edges
                margin_y = int(height * 0.1)
                initial_contour = np.array([
                    [margin_x, margin_y],
                    [width - margin_x, margin_y],
                    [width - margin_x, height - margin_y],
                    [margin_x, height - margin_y]
                ])
                processed_image = Segmentation.active_contour(
                    input_image, 
                    initial_contour,
                    alpha=alpha,
                    beta=beta,
                    gamma=0.001,
                    iterations=iterations
                )
            processing_states[filename]['segmentation'] = method
            
        elif current_stage == 'contour_refinement':
            # Always use the segmented image for contour refinement
            if processing_states[filename]['segmentation']:
                input_image = processing_states[filename]['current_image']
            else:
                # If no segmentation, apply basic preprocessing
                input_image = processing_states[filename]['original_image']
                # Apply CLAHE for better contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                input_image = clahe.apply(input_image)
                # Apply bilateral filter for edge preservation
                input_image = cv2.bilateralFilter(input_image, 9, 75, 75)
            method = request.form.get('contour_refinement')
            # Get user-defined parameters with defaults
            threshold = int(request.form.get('threshold', 127))
            min_area = int(request.form.get('min_area', 100))
            max_complexity = int(request.form.get('max_complexity', 50))
            kernel_size = int(request.form.get('kernel_size', 5))
            iterations = int(request.form.get('iterations', 1))
            
            if method == 'morphological':
                processed_image = ContourRefinement.apply_morphology(
                    input_image, 
                    operation='close',
                    kernel_size=kernel_size,
                    iterations=iterations,
                    threshold=threshold
                )
            elif method == 'enhancement':
                processed_image = ContourRefinement.enhance_contours(
                    input_image,
                    threshold=threshold,
                    min_area=min_area,
                    max_complexity=max_complexity
                )
            processing_states[filename]['contour_refinement'] = method

        # Update current image state
        processing_states[filename]['current_image'] = processed_image

        # Save the processed image
        stage_filename = f'processed_{current_stage}_{filename}'
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], stage_filename), processed_image)
        
        # If this is the final stage (contour_refinement), perform plant detection
        if current_stage == 'contour_refinement' and method != 'none':
            # Load the processed image for detection
            detection_result = plant_detector.detect(processed_image)
            return jsonify({
                'processed_image': stage_filename,
                'detection': {
                    'plant_type': detection_result['class_name'],
                    'confidence': f"{detection_result['confidence']:.2%}"
                }
            })
        
        return jsonify({'processed_image': stage_filename})

    # Render stages page
    return render_template('stages.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
