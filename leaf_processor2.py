import cv2
import numpy as np
from skimage import filters, segmentation, morphology, feature
from scipy import ndimage



class LeafProcessor:
    """
    A comprehensive image processing library for leaf analysis, providing methods for
    noise reduction, edge detection, segmentation, and morphological operations.
    """
    
    @staticmethod
    def reduce_noise(image, method='gaussian', params=None):
        """
        Reduce noise in the image using various filtering methods.
        
        Args:
            image: Input image (numpy array)
            method: String specifying the noise reduction method
                   ('gaussian', 'median', 'bilateral', 'nlm')
            params: Dictionary of parameters for the chosen method
        
        Returns:
            Denoised image
        """
        if params is None:
            params = {}
            
        if method == 'gaussian':
            kernel_size = params.get('kernel_size', (5, 5))
            sigma = params.get('sigma', 0)
            return cv2.GaussianBlur(image, kernel_size, sigma)
        
        elif method == 'median':
            kernel_size = params.get('kernel_size', 5)
            return cv2.medianBlur(image, kernel_size)
        
        elif method == 'bilateral':
            d = params.get('d', 9)
            sigma_color = params.get('sigma_color', 75)
            sigma_space = params.get('sigma_space', 75)
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        elif method == 'nlm':  # Non-local means denoising
            h = params.get('h', 10)  # Filter strength
            template_window = params.get('template_window', 7)
            search_window = params.get('search_window', 21)
            return cv2.fastNlMeansDenoising(image, None, h, template_window, search_window)
        
        else:
            raise ValueError(f"Unknown noise reduction method: {method}")

    @staticmethod
    def to_grayscale(image):
        """
        Convert an image to grayscale if it's not already.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def detect_edges_canny(image, low_threshold=50, high_threshold=150, aperture_size=3):
        """
        Apply Canny edge detection.
        
        Args:
            image: Grayscale input image
            low_threshold: Lower threshold for edge detection
            high_threshold: Higher threshold for edge detection
            aperture_size: Aperture size for the Sobel operator
            
        Returns:
            Binary edge image
        """
        return cv2.Canny(image, low_threshold, high_threshold, apertureSize=aperture_size)

    @staticmethod
    def detect_edges_sobel(image, ksize=3, scale=1, delta=0):
        """
        Apply Sobel edge detection.
        
        Args:
            image: Grayscale input image
            ksize: Kernel size
            scale: Scale factor for computed derivative values
            delta: Value added to results
            
        Returns:
            Edge magnitude image
        """
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale, delta=delta)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale, delta=delta)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    @staticmethod
    def detect_edges_prewitt(image):
        """
        Apply Prewitt edge detection.
        
        Args:
            image: Grayscale input image
            
        Returns:
            Edge magnitude image
        """
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        
        grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    @staticmethod
    def detect_edges_log(image, sigma=1.0):
        """
        Apply Laplacian of Gaussian (LoG) edge detection.
        
        Args:
            image: Grayscale input image
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            Edge image
        """
        # Apply Gaussian blur
        blurred = ndimage.gaussian_filter(image, sigma)
        
        # Apply Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        
        # Find zero crossings
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

    @staticmethod
    def segment_otsu(image):
        """
        Perform image segmentation using Otsu's thresholding.
        
        Args:
            image: Grayscale input image
            
        Returns:
            Binary segmentation mask
        """
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def segment_adaptive(image, block_size=11, C=2):
        """
        Perform adaptive thresholding for segmentation.
        
        Args:
            image: Grayscale input image
            block_size: Size of pixel neighborhood for threshold calculation
            C: Constant subtracted from mean
            
        Returns:
            Binary segmentation mask
        """
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size, C)

    @staticmethod
    def active_contour(image, initial_contour, alpha=0.01, beta=0.1, gamma=0.001, iterations=100):
        """
        Apply active contour (snake) segmentation.
        
        Args:
            image: Grayscale input image
            initial_contour: Initial contour points
            alpha: Snake length shape parameter
            beta: Snake smoothness shape parameter
            gamma: Step size parameter
            iterations: Number of iterations
            
        Returns:
            Final contour points
        """
        return segmentation.active_contour(
            image,
            initial_contour,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            max_iterations=iterations
        )

    @staticmethod
    def apply_morphology(image, operation='dilate', kernel_size=3, iterations=1):
        """
        Apply morphological operations.
        
        Args:
            image: Binary input image
            operation: Type of morphological operation
                      ('dilate', 'erode', 'open', 'close')
            kernel_size: Size of the structuring element
            iterations: Number of times to apply the operation
            
        Returns:
            Processed binary image
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operation == 'dilate':
            return cv2.dilate(image, kernel, iterations=iterations)
        elif operation == 'erode':
            return cv2.erode(image, kernel, iterations=iterations)
        elif operation == 'open':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == 'close':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        else:
            raise ValueError(f"Unknown morphological operation: {operation}")

    @staticmethod
    def enhance_contours(image, method='normalize'):
        """
        Enhance contours in the image.
        
        Args:
            image: Input edge or contour image
            method: Enhancement method ('normalize', 'equalize', 'adaptive')
            
        Returns:
            Enhanced image
        """
        if method == 'normalize':
            return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        elif method == 'equalize':
            return cv2.equalizeHist(image)
        elif method == 'adaptive':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(image)
        else:
            raise ValueError(f"Unknown enhancement method: {method}")

    @staticmethod
    def skeletonize(binary_image):
        """
        Create a skeleton representation of the binary image.
        
        Args:
            binary_image: Binary input image
            
        Returns:
            Skeletonized image
        """
        return morphology.skeletonize(binary_image > 0).astype(np.uint8) * 255
