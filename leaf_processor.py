import cv2
import numpy as np
from skimage import filters, segmentation, morphology, feature
from scipy import ndimage

class ImagePreprocessor:
    @staticmethod
    def to_grayscale(image):
        """Convert an image to grayscale if it's not already."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def binary_threshold(image, threshold=127, invert=False):
        """Apply binary thresholding to an image."""
        _, thresh = cv2.threshold(image, threshold, 255, 
                                cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY)
        return thresh

    @staticmethod
    def adaptive_threshold(image, block_size=11, C=2):
        """Apply adaptive thresholding to an image."""
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size, C)

    @staticmethod
    def otsu_threshold(image):
        """Apply Otsu's thresholding to an image."""
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

class LinearNoiseReduction:
    """Class for linear noise reduction methods that use convolution operations."""
    
    @staticmethod
    def gaussian_filter(image, kernel_size=(5, 5), sigma=0):
        """Apply Gaussian filter for noise reduction."""
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    @staticmethod
    def mean_filter(image, kernel_size=5):
        """Apply mean (averaging) filter."""
        return cv2.blur(image, (kernel_size, kernel_size))
    
    @staticmethod
    def laplacian_filter(image, kernel_size=3):
        """Apply Laplacian filter for detail enhancement."""
        # First apply Gaussian to reduce noise sensitivity
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)
        # Convert back to uint8 and enhance details
        abs_laplacian = np.absolute(laplacian)
        enhanced = cv2.normalize(abs_laplacian, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.addWeighted(image, 1.5, enhanced, -0.5, 0)
    
    @staticmethod
    def butterworth_filter(image, cutoff_frequency=30, order=2):
        """Apply Butterworth low-pass filter in frequency domain."""
        # Convert to frequency domain
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        
        # Create Butterworth filter
        rows, cols = image.shape
        crow, ccol = rows//2, cols//2
        y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
        d = np.sqrt(x*x + y*y)
        butterworth = 1 / (1 + (d/cutoff_frequency)**(2*order))
        
        # Apply filter and convert back to spatial domain
        fshift_filtered = fshift * butterworth
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_filtered = np.abs(img_back)
        
        return cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

class NonLinearNoiseReduction:
    """Class for non-linear noise reduction methods."""
    
    @staticmethod
    def median_filter(image, kernel_size=5):
        """Apply median filter, effective for salt-and-pepper noise."""
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
        """Apply bilateral filter for edge-preserving smoothing."""
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    @staticmethod
    def nlm_filter(image, h=10, template_window=7, search_window=21):
        """Apply Non-local Means denoising."""
        return cv2.fastNlMeansDenoising(image, None, h, template_window, search_window)
    
    @staticmethod
    def adaptive_median_filter(image, max_size=7):
        """Apply adaptive median filter with variable window size."""
        filtered = np.copy(image)
        padding = max_size // 2
        padded = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)
        
        for i in range(padding, padded.shape[0] - padding):
            for j in range(padding, padded.shape[1] - padding):
                window_size = 3
                while window_size <= max_size:
                    half = window_size // 2
                    window = padded[i-half:i+half+1, j-half:j+half+1]
                    median = np.median(window)
                    min_val = np.min(window)
                    max_val = np.max(window)
                    
                    if min_val < median < max_val:
                        if min_val < padded[i,j] < max_val:
                            filtered[i-padding,j-padding] = padded[i,j]
                        else:
                            filtered[i-padding,j-padding] = median
                        break
                    else:
                        window_size += 2
                        if window_size > max_size:
                            filtered[i-padding,j-padding] = median
                            
        return filtered

class EdgeDetector:
    @staticmethod
    def canny(image, low_threshold=50, high_threshold=150, aperture_size=3):
        """Apply Canny edge detection."""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        return cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=aperture_size)

    @staticmethod
    def sobel(image, ksize=3, scale=1, delta=0):
        """Apply Sobel edge detection."""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        # Apply Sobel in x and y directions
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale, delta=delta)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale, delta=delta)
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        # Normalize and threshold
        normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, thresholded = cv2.threshold(normalized, 50, 255, cv2.THRESH_BINARY)
        return thresholded

    @staticmethod
    def prewitt(image):
        """Apply Prewitt edge detection."""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        # Define Prewitt kernels
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        
        # Apply kernels
        grad_x = cv2.filter2D(blurred, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(blurred, cv2.CV_64F, kernel_y)
        
        # Calculate magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        # Normalize and threshold
        normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, thresholded = cv2.threshold(normalized, 50, 255, cv2.THRESH_BINARY)
        return thresholded

    @staticmethod
    def log(image, sigma=1.0):
        """Apply Laplacian of Gaussian (LoG) edge detection."""
        # Apply Gaussian blur with specified sigma
        blurred = ndimage.gaussian_filter(image, sigma)
        # Apply Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
        
        zero_crossings = np.zeros_like(laplacian, dtype=np.uint8)
        rows, cols = laplacian.shape
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if (laplacian[i,j] * laplacian[i+1,j] < 0) or \
                   (laplacian[i,j] * laplacian[i-1,j] < 0) or \
                   (laplacian[i,j] * laplacian[i,j+1] < 0) or \
                   (laplacian[i,j] * laplacian[i,j-1] < 0):
                    zero_crossings[i,j] = 255
                    
        return zero_crossings

class Segmentation:
    @staticmethod
    def otsu(image, block_size=35, c=5):
        """Perform enhanced Otsu's thresholding with preprocessing."""
        # Ensure 8-bit image
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # Apply bilateral filter to preserve edges while reducing noise
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Apply adaptive thresholding instead of global Otsu
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            c
        )
        
        # Clean up using morphological operations
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return cleaned

    @staticmethod
    def adaptive(image, block_size=11, C=2):
        """Perform adaptive thresholding."""
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size, C)

    @staticmethod
    def active_contour(image, initial_contour, alpha=0.01, beta=0.1, gamma=0.001, iterations=200):
        """Apply improved active contour (snake) segmentation."""
        # Enhanced preprocessing
        # 1. CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # 2. Denoise while preserving edges
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # 3. Apply bilateral filter to preserve edges
        bilateral = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        # 4. Create strong edge map
        edges = cv2.Canny(bilateral, 30, 150)
        gradient = cv2.Sobel(bilateral, cv2.CV_64F, 1, 1)
        gradient = cv2.normalize(gradient, None, 0, 1, cv2.NORM_MINMAX)
        
        # Combine edge information
        edge_force = (edges.astype(float) / 255.0) + gradient
        edge_force = cv2.normalize(edge_force, None, 0, 1, cv2.NORM_MINMAX)
        
        # Normalize image for snake algorithm
        img_normalized = bilateral.astype(float) / 255.0
        
        # Create external force field
        external_force = cv2.GaussianBlur(edge_force, (5,5), 0)
        
        # Evolution
        snake = segmentation.active_contour(
            external_force,
            initial_contour,
            alpha=alpha,    # Controls elasticity
            beta=beta,      # Controls rigidity
            gamma=gamma,    # External force weight
            max_num_iter=iterations,
            convergence=0.1
        )
        
        # Create refined mask with inverted colors (white background, black object)
        mask = np.ones_like(image, dtype=np.uint8) * 255
        snake_points = np.round(snake).astype(np.int32)
        cv2.fillPoly(mask, [snake_points], 0)
        
        # Post-process the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask

class ContourRefinement:
    @staticmethod
    def apply_morphology(image, operation='close', kernel_size=7, iterations=2, threshold=90):
        """Apply enhanced morphological operations for contour refinement."""
        # Ensure proper input format
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # Apply thresholding with user-defined threshold (inverted to match enhance_contours)
        _, binary = cv2.threshold(enhanced, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Create kernels of different sizes for multi-scale processing
        kernel_small = np.ones((3,3), np.uint8)
        kernel_main = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Initial noise removal with bilateral filter
        denoised = cv2.bilateralFilter(binary, 9, 75, 75)
        
        # Apply main morphological operation
        if operation == 'dilate':
            result = cv2.dilate(denoised, kernel_main, iterations=iterations)
        elif operation == 'erode':
            result = cv2.erode(denoised, kernel_main, iterations=iterations)
        elif operation == 'open':
            result = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel_main, iterations=iterations)
        elif operation == 'close':
            result = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel_main, iterations=iterations)
        else:
            raise ValueError(f"Unknown morphological operation: {operation}")
        
        # Advanced post-processing
        # 1. Remove small objects
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
        min_size = 100  # Adjusted minimum size threshold
        
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] < min_size:
                result[labels == label] = 0
        
        # 2. Fill holes using floodfill (adjusted for white background)
        filled = result.copy()
        h, w = filled.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(filled, mask, (0,0), 0)  # Fill with black
        filled_inv = cv2.bitwise_not(filled)
        result = cv2.bitwise_and(result, filled_inv)
        
        # 3. Smooth boundaries while preserving edges
        result = cv2.bilateralFilter(result, 9, 75, 75)
        
        return result

    @staticmethod
    def enhance_contours(image, threshold=127, min_area=100, max_complexity=50):
        """Enhanced contour detection and refinement."""
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # 2. Denoise while preserving edges
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # 3. Apply bilateral filter for edge preservation
        bilateral = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        # 4. Create binary image using user-defined threshold
        _, binary = cv2.threshold(bilateral, threshold, 255, cv2.THRESH_BINARY)
        
        # 5. Find and filter contours
        contours, hierarchy = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_TC89_KCOS
        )
        
        # Filter contours by area and complexity
        # Use provided min_area parameter
        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                # Calculate contour complexity
                perimeter = cv2.arcLength(cnt, True)
                complexity = perimeter * perimeter / (4 * np.pi * area)
                if complexity < max_complexity:  # Use provided max_complexity parameter
                    filtered_contours.append(cnt)
        
        # 6. Create result image (white background)
        result = np.ones_like(image) * 255
        
        # 7. Draw contours with adaptive thickness
        for contour in filtered_contours:
            area = cv2.contourArea(contour)
            # Smooth contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Calculate adaptive thickness based on contour size
            thickness = max(1, min(3, int(np.sqrt(area) / 100)))
            
            # Draw filled contour with small border (black on white)
            cv2.drawContours(result, [approx], -1, 0, -1)  # Fill with black
            cv2.drawContours(result, [approx], -1, 0, thickness)  # Border in black
        
        # 8. Post-processing
        # Apply morphological operations to smooth boundaries
        kernel = np.ones((3,3), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        
        # Final smoothing with edge preservation
        result = cv2.bilateralFilter(result, 9, 75, 75)
        
        return result

    @staticmethod
    def skeletonize(binary_image):
        """Create a skeleton representation of the binary image."""
        # Ensure binary image
        if len(binary_image.shape) > 2:
            binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
        
        # Apply skeletonization
        skeleton = morphology.skeletonize(binary > 0).astype(np.uint8) * 255
        return skeleton
