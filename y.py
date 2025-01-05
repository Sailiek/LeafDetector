import cv2
import numpy as np
import argparse
from PIL import Image

PENNY_LOWER_BOUND = (0, 50, 50)       # Lower bound of penny color in HSV
PENNY_UPPER_BOUND = (30, 255, 255)    # Upper bound of penny color in HSV


PENNY_DIAMETER_MM = 19.05
def get_image_resolution(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded. Check the file path.")

    # Get dimensions
    height, width = img.shape[:2]

    # Get DPI if available in metadata
    dpi = None
    try:
        with Image.open(image_path) as img_pil:
            dpi = img_pil.info.get('dpi')
    except Exception as e:
        print(f"Warning: Unable to read DPI: {e}")

    return {
        'width_px': width,
        'height_px': height,
        'dpi': dpi
    }


def calculate_scale_factor(image_path):
    # For Leafsnap lab images, the white background provides consistent scale
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded. Check the file path.")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get image dimensions
    height, width = gray.shape

    # Calculate diagonal length of image in pixels
    diagonal_pixels = np.sqrt(height**2 + width**2)

    return diagonal_pixels  # Return diagonal length in pixels


def estimate_real_size(image_path, lower_bound, upper_bound):
    # If working with lab images that have penny/coin for scale
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded. Check the file path.")

    # Convert to HSV for better coin detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create a mask for penny detection
    penny_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours in the penny mask
    penny_contours, _ = cv2.findContours(penny_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if penny_contours:
        # Get the largest contour (assumed to be the penny)
        penny_contour = max(penny_contours, key=cv2.contourArea)

        # Get the bounding box of the penny
        x, y, w, h = cv2.boundingRect(penny_contour)

        # Assume the diameter is the width of the bounding box
        penny_diameter_pixels = max(w, h)

        # Calculate mm per pixel
        scale_factor = PENNY_DIAMETER_MM / penny_diameter_pixels

        return scale_factor

    print("Penny not detected in image.")
    return None


def main(image_path):
    try:
        # Get image resolution
        resolution_info = get_image_resolution(image_path)
        print("Image Resolution:")
        print(f"  Width (px): {resolution_info['width_px']}")
        print(f"  Height (px): {resolution_info['height_px']}")
        if resolution_info['dpi']:
            print(f"  DPI: {resolution_info['dpi']}")
        else:
            print("  DPI: Not available")

        # Calculate scale factor
        scale_in_pixels = calculate_scale_factor(image_path)
        print(f"\nImage Diagonal Length (px): {scale_in_pixels:.2f}")

        # Estimate real-world size using a penny as a reference
        scale_factor = estimate_real_size(image_path, PENNY_LOWER_BOUND, PENNY_UPPER_BOUND)
        if scale_factor:
            print(f"\nReal-World Scale Factor (mm/px): {scale_factor:.4f}")
        else:
            print("\nScale factor couldn't be determined. Penny not detected.")

    except Exception as e:
        print(f"Error: {e}")



if __name__ == "__main__":
    main("C:/Users/salah/Downloads/13001155906273.jpg")