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
    def otsu(image):
        """Perform Otsu's thresholding."""
        # Convert to 8-bit unsigned integer if needed
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def adaptive(image, block_size=11, C=2):
        """Perform adaptive thresholding."""
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size, C)

    @staticmethod
    def active_contour(image, initial_contour, alpha=0.01, beta=0.1, gamma=0.001, iterations=100):
        """Apply active contour (snake) segmentation."""
        return segmentation.active_contour(
            image,
            initial_contour,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            max_num_iter=iterations
        )

class ContourRefinement:
    @staticmethod
    def apply_morphology(image, operation='dilate', kernel_size=3, iterations=1):
        """Apply morphological operations for contour refinement."""
        # First ensure we have a binary image
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # First apply closing to connect broken contours
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Apply requested operation
        if operation == 'dilate':
            result = cv2.dilate(closed, kernel, iterations=iterations)
        elif operation == 'erode':
            result = cv2.erode(closed, kernel, iterations=iterations)
        elif operation == 'open':
            result = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == 'close':
            result = closed
        else:
            raise ValueError(f"Unknown morphological operation: {operation}")
            
        # Remove small objects and fill holes
        # Find all connected components
        num_labels, labels = cv2.connectedComponents(result)
        min_size = 100  # Minimum size threshold
        
        # Remove small objects
        for label in range(1, num_labels):
            mask = labels == label
            if np.sum(mask) < min_size:
                result[mask] = 0
        
        # Fill holes using floodfill
        filled = result.copy()
        mask = np.zeros((result.shape[0] + 2, result.shape[1] + 2), np.uint8)
        cv2.floodFill(filled, mask, (0,0), 255)
        filled_inv = cv2.bitwise_not(filled)
        result = result | filled_inv
                
        return result

    @staticmethod
    def enhance_contours(image, method='normalize'):
        """Enhance contours using advanced techniques."""
        # First ensure we have a grayscale image
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find edges using Canny
        edges = cv2.Canny(image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a blank image to draw contours
        contour_img = np.zeros_like(image)
        cv2.drawContours(contour_img, contours, -1, (255,255,255), 2)
        
        # Apply enhancement method
        if method == 'normalize':
            enhanced = cv2.normalize(contour_img, None, 0, 255, cv2.NORM_MINMAX)
        elif method == 'equalize':
            enhanced = cv2.equalizeHist(contour_img)
        elif method == 'adaptive':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(contour_img)
        else:
            raise ValueError(f"Unknown enhancement method: {method}")
        
        return enhanced

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
