"""
preprocessor.py
---------------
AI/ML Module — Data Pre-processing
Cleans and enhances bill images before OCR scanning
for better text recognition accuracy.

Author: AI/ML Intern
Project: AI-Powered Billing Management System
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Handles image enhancement and cleanup pipeline.
    Applies a series of transformations to improve OCR accuracy
    on real-world bill photos (blurry, skewed, low contrast, etc.)
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

    def preprocess(self, image_input) -> np.ndarray:
        """
        Full preprocessing pipeline.
        Accepts: file path (str), bytes, PIL Image, or numpy array.
        Returns: cleaned numpy array ready for OCR.
        """
        img = self._load_image(image_input)

        logger.info("Starting image preprocessing pipeline...")

        # Step 1: Convert to grayscale
        img = self._to_grayscale(img)

        # Step 2: Resize for better resolution
        img = self._resize_for_ocr(img)

        # Step 3: Remove noise
        img = self._denoise(img)

        # Step 4: Fix brightness and contrast
        img = self._fix_contrast(img)

        # Step 5: Deskew (straighten tilted images)
        img = self._deskew(img)

        # Step 6: Apply adaptive thresholding (makes text sharp black/white)
        img = self._threshold(img)

        # Step 7: Remove borders/shadows
        img = self._remove_borders(img)

        logger.info("Preprocessing complete.")
        return img

    # ------------------------------------------------------------------ #
    #  Private Helpers
    # ------------------------------------------------------------------ #

    def _load_image(self, image_input) -> np.ndarray:
        """Load image from path, bytes, PIL Image, or numpy array."""
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise FileNotFoundError(f"Image not found: {image_input}")
            return img

        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        elif isinstance(image_input, Image.Image):
            return cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)

        elif isinstance(image_input, np.ndarray):
            return image_input

        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert BGR image to grayscale."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _resize_for_ocr(self, img: np.ndarray, target_height: int = 1200) -> np.ndarray:
        """
        Scale image so height is at least target_height pixels.
        Tesseract performs better on larger images.
        """
        h, w = img.shape[:2]
        if h < target_height:
            scale = target_height / h
            new_w = int(w * scale)
            img = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
            logger.debug(f"Resized from {h}x{w} to {target_height}x{new_w}")
        return img

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Remove noise while preserving text edges."""
        # Gaussian blur for general noise
        img = cv2.GaussianBlur(img, (3, 3), 0)
        # Bilateral filter to preserve edges (text borders)
        img = cv2.bilateralFilter(img, 9, 75, 75)
        return img

    def _fix_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        to fix uneven lighting across the bill image.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Detect and correct skew angle in the image.
        Handles bills photographed at a slight angle.
        """
        try:
            coords = np.column_stack(np.where(img > 0))
            angle = cv2.minAreaRect(coords)[-1]

            # Normalize angle to range [-45, 45]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            # Only correct if skew is significant (> 0.5 degrees)
            if abs(angle) > 0.5:
                h, w = img.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(
                    img, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                logger.debug(f"Deskewed by {angle:.2f} degrees")
        except Exception as e:
            logger.warning(f"Deskew failed (non-critical): {e}")

        return img

    def _threshold(self, img: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding to create a clean black-and-white image.
        Works better than simple threshold on real-world bill photos
        with uneven lighting.
        """
        return cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,   # block size
            2     # constant subtracted from mean
        )

    def _remove_borders(self, img: np.ndarray) -> np.ndarray:
        """
        Crop out dark borders or shadows around the bill.
        Finds the largest white content area and crops to it.
        """
        try:
            # Invert to find content region
            inverted = cv2.bitwise_not(img)
            contours, _ = cv2.findContours(
                inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                # Add small padding
                pad = 10
                x, y = max(0, x - pad), max(0, y - pad)
                img = img[y:y + h + pad, x:x + w + pad]
        except Exception as e:
            logger.warning(f"Border removal failed (non-critical): {e}")

        return img

    def to_pil(self, img: np.ndarray) -> Image.Image:
        """Convert processed numpy array back to PIL Image."""
        return Image.fromarray(img)

    def to_bytes(self, img: np.ndarray, format: str = "PNG") -> bytes:
        """Convert processed image to bytes."""
        pil_img = self.to_pil(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format=format)
        return buffer.getvalue()
