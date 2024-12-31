import tensorflow as tf
import numpy as np
import cv2

class PlantDetector:
    def __init__(self, model_path='model/best.keras', class_names_path='sc_names.txt'):
        """Initialize the plant detector with model and class names."""
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = self._load_class_names(class_names_path)
        
    def _load_class_names(self, class_names_path):
        """Load class names from file, keeping only the scientific names."""
        class_names = []
        with open(class_names_path, 'r') as f:
            for line in f:
                # Extract scientific name (part before the dash)
                scientific_name = line.split('-')[0].strip()
                class_names.append(scientific_name)
        return class_names
    
    def preprocess_image(self, image):
        """Preprocess image for model input."""
        # Resize to match model's expected input size
        resized = cv2.resize(image, (800, 600))
        
        # Convert to grayscale if needed
        if len(resized.shape) == 3:  # If color image
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Normalize pixel values
        normalized = resized / 255.0
        
        # Add channel and batch dimensions
        expanded = np.expand_dims(normalized, axis=-1)  # Add channel dimension
        batched = np.expand_dims(expanded, axis=0)  # Add batch dimension
        return batched
    
    def detect(self, image):
        """Detect plant type from image."""
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Get model predictions
        predictions = self.model.predict(processed_image)
        
        # Get top prediction
        top_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][top_idx])
        
        return {
            'class_name': self.class_names[top_idx],
            'confidence': confidence
        }
