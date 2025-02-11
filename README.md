# Leaf Detector - Identification des Espèces de Plantes

[English Version Below](#leaf-detector---plant-species-identification)

## Liens Importants
- [GitHub Repository](https://github.com/Sailiek/LeafDetector)
- [Vidéo Explicative](https://drive.google.com/file/d/1WLMzm9IGtDohwYH7XLjYWhpBwAbR9GDD/view?usp=sharing)
- [Notebook Colab](https://colab.research.google.com/drive/1duMranwbSOXRW2UPsuXhXqcvCU6VZdEv)
- [Dataset LeafSnap Original](https://www.kaggle.com/datasets/xhlulu/leafsnap-dataset)
- [Dataset Modifié (Utilisé pour l'Entraînement)](https://drive.google.com/file/d/1qajwcZcF_A5V7VBxefO2o2j9J5SLEK8W/view?usp=sharing)
- [Model utilisé](https://drive.google.com/file/d/10cDsTFXXV6t43PU6HY0u9EISz5s18EhG/view?usp=drive_link)

## Description
Leaf Detector est une application web basée sur Flask qui permet aux utilisateurs d'appliquer diverses techniques de traitement d'image et d'identifier les espèces de plantes à l'aide d'un modèle d'apprentissage automatique.

## Fonctionnalités

### 1. Pipeline de Traitement d'Image
- Réduction du bruit :
  - Filtres linéaires (Gaussien)
  - Filtres non linéaires (Médian, Bilatéral)
- Détection des contours :
  - Canny
  - Sobel
  - Prewitt
  - Laplacien de Gaussienne (LoG)
- Segmentation :
  - Seuillage d'Otsu
  - Seuillage adaptatif
  - Contours actifs

### 2. Identification des Espèces
- Modèle CNN personnalisé
- Précision d'entraînement : 94.03%
- Top-3 précision : 99.48%
- 20 espèces de plantes supportées

### 3. Estimation de l'Âge des Feuilles
- Analyse morphologique
- Calcul de surface et périmètre
- Estimation basée sur les taux de croissance

## Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/Sailiek/LeafDetector
cd LeafDetector
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
```

3. Activer l'environnement :
- Windows :
```bash
venv\Scripts\activate
```
- macOS/Linux :
```bash
source venv/bin/activate
```

4. Installer les dépendances :
```bash
pip install -r requirements.txt
```

5. Télécharger le modèle entraîné et le placer dans le dossier `model/`

6. Lancer l'application :
```bash
python app.py
```

7. Accéder à l'application : `http://localhost:5000`

## Détails Techniques

### Architecture
- **Backend** : Flask (Python)
- **Traitement d'Image** : OpenCV, scikit-image
- **Machine Learning** : TensorFlow/Keras
- **Frontend** : HTML5, CSS3, JavaScript

### Détails du Modèle
- **Dataset** : LeafSnap modifié (20 espèces)
- **Images** :
  - Entraînement : 27,915
  - Validation : 5,991
  - Test : 5,982
- **Prétraitement** :
  - Redimensionnement : 800x600
  - Conversion en niveaux de gris
  - Détection des contours
  - Augmentation des données

### Performance du Modèle
- Précision finale d'entraînement : 94.03%
- Perte finale d'entraînement : 0.1892
- Meilleure précision de validation : 90.19%
- Meilleure perte de validation : 0.3016

---

# Leaf Detector - Plant Species Identification

## Important Links
- [GitHub Repository](https://github.com/Sailiek/LeafDetector)
- [Explanatory Video](https://drive.google.com/file/d/1WLMzm9IGtDohwYH7XLjYWhpBwAbR9GDD/view?usp=sharing)
- [Colab Notebook](https://colab.research.google.com/drive/1duMranwbSOXRW2UPsuXhXqcvCU6VZdEv)
- [Original LeafSnap Dataset](https://www.kaggle.com/datasets/xhlulu/leafsnap-dataset)
- [Modified Dataset (Used for Training)](https://drive.google.com/file/d/1qajwcZcF_A5V7VBxefO2o2j9J5SLEK8W/view?usp=sharing)

## Description
Leaf Detector is a Flask-based web application that allows users to upload leaf images, apply various image processing techniques, and identify plant species using a machine learning model.

## Features

### 1. Image Processing Pipeline
- Noise Reduction:
  - Linear filters (Gaussian)
  - Non-linear filters (Median, Bilateral)
- Edge Detection:
  - Canny
  - Sobel
  - Prewitt
  - Laplacian of Gaussian (LoG)
- Segmentation:
  - Otsu's thresholding
  - Adaptive thresholding
  - Active contours

### 2. Species Identification
- Custom CNN model
- Training accuracy: 94.03%
- Top-3 accuracy: 99.48%
- Support for 20 plant species

### 3. Leaf Age Estimation
- Morphological analysis
- Area and perimeter calculation
- Growth rate-based estimation

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

3. Activate the environment:
- Windows:
```bash
venv\Scripts\activate
```
- macOS/Linux:
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

7. Access the application: `http://localhost:5000`

## Technical Details

### Architecture
- **Backend**: Flask (Python)
- **Image Processing**: OpenCV, scikit-image
- **Machine Learning**: TensorFlow/Keras
- **Frontend**: HTML5, CSS3, JavaScript

### Model Details
- **Dataset**: Modified LeafSnap (20 species)
- **Images**:
  - Training: 27,915
  - Validation: 5,991
  - Test: 5,982
- **Preprocessing**:
  - Resizing: 800x600
  - Grayscale conversion
  - Edge detection
  - Data augmentation

### Model Performance
- Final Training Accuracy: 94.03%
- Final Training Loss: 0.1892
- Best Validation Accuracy: 90.19%
- Best Validation Loss: 0.3016
