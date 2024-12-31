# Leaf Detector - Plant Species Identification

![Leaf Detector Interface](screenshots/interface.png)

Leaf Detector is a Flask-based web application that allows users to upload leaf images, apply various image processing techniques, and identify plant species using a machine learning model.

## Features

- Image upload and processing pipeline
- Multiple image processing stages:
  - Noise reduction (linear and non-linear filters)
  - Edge detection (Canny, Sobel, Prewitt, LoG)
  - Segmentation (thresholding, active contours)
  - Contour refinement
- Plant species identification using a trained Keras model
- Modern and responsive web interface
- Real-time image processing previews

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Sailiek/LeafDetector
   cd LeafDetector
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Download the trained model and place it in the `model/` directory

6. Run the application:
   ```bash
   python app.py
   ```

7. Access the application at `http://localhost:5000`

## Model Training Details

### Dataset
- **Source**: Modified LeafSnap dataset (20 species) Link to the original Leaf Snap Dataset : [kaggle](https://www.kaggle.com/datasets/xhlulu/leafsnap-dataset)
- **Images**:
  - Training: 27,915
  - Validation: 5,991
  - Test: 5,982
- **Preprocessing**:
  - Images cropped and resized to 800x600
  - Converted to grayscale
  - Edge detection applied
  - Augmentation:
    - Horizontal and vertical flipping
    - Rotation
- **Dataset Link**: [Modified LeafSnap Dataset](https://drive.google.com/file/d/1qajwcZcF_A5V7VBxefO2o2j9J5SLEK8W/view?usp=sharing)

### Training Process
- **Epochs**: 13 (originally planned for 35, stopped due to resource constraints)
- **Model Architecture**: Custom CNN
- **Training Metrics**:
  - Final Training Accuracy: 94.03%
  - Final Training Loss: 0.1892
  - Final Top-3 Accuracy: 99.48%
- **Validation Metrics**:
  - Best Validation Accuracy: 90.19% (Epoch 11)
  - Best Validation Loss: 0.3016 (Epoch 11)
  - Best Top-3 Validation Accuracy: 98.66% (Epoch 11)

### Performance
- The model achieved good performance despite early stopping
- Top-3 accuracy consistently above 95% after epoch 5
- Model shows strong generalization capabilities

## Usage

1. Upload a leaf image using the upload interface
2. Process the image through various stages:
   - Noise cleaning
   - Edge detection
   - Segmentation
   - Contour refinement
3. View the final plant species identification results

## File Structure

```
LeafDetector/
├── app.py                # Main Flask application
├── leaf_processor.py     # Image processing functions
├── plant_detector.py     # Plant species detection model
├── requirements.txt      # Python dependencies
├── sc_names.txt          # Scientific names of plant species
├── static/               # Static files (CSS, JS, images)
│   └── styles.css        # Main stylesheet
├── templates/            # HTML templates
│   ├── index.html        # Main page
│   └── stages.html       # Processing stages page
├── uploads/              # Uploaded and processed images
└── model/                # Trained model files
    └── best.keras        # Keras model file
```

## Technical Details

- **Backend**: Flask (Python)
- **Image Processing**: OpenCV, scikit-image
- **Machine Learning**: TensorFlow/Keras
- **Frontend**: HTML5, CSS3, JavaScript
- **Model**: Custom CNN trained on 20 plant species
