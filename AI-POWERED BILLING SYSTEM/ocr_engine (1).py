"""
ocr_engine.py
-------------
AI/ML Module — Automated OCR Scanning
Wrapper around Tesseract OCR to extract raw text
from preprocessed bill images.

Author: AI/ML Intern
Project: AI-Powered Billing Management System
"""

import pytesseract
from PIL import Image
import numpy as np
import logging
import re

logger = logging.getLogger(__name__)


class OCREngine:
    """
    Tesseract OCR wrapper optimized for bill/receipt scanning.
    Handles configuration, text extraction, and basic cleanup.
    """

    # Tesseract page segmentation modes
    # PSM 6 = Assume a single uniform block of text (best for receipts)
    # PSM 4 = Assume a single column of text of variable sizes
    DEFAULT_CONFIG = r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/:- "

    def __init__(self, tesseract_path: str = None, lang: str = "eng"):
        """
        Args:
            tesseract_path: Path to tesseract executable (optional).
                           Usually auto-detected on Linux/Mac.
                           On Windows: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            lang: OCR language. Default "eng" (English).
                  Use "eng+hin" for Hindi+English bills.
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        self.lang = lang
        self._verify_tesseract()

    def _verify_tesseract(self):
        """Check if Tesseract is installed and accessible."""
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception:
            logger.error(
                "Tesseract not found! Install it:\n"
                "  Ubuntu: sudo apt-get install tesseract-ocr\n"
                "  macOS:  brew install tesseract\n"
                "  Windows: https://github.com/tesseract-ocr/tesseract"
            )
            raise RuntimeError("Tesseract OCR engine not installed.")

    def extract_text(self, image) -> str:
        """
        Extract raw text from a preprocessed image.
        
        Args:
            image: numpy array or PIL Image
        
        Returns:
            Raw extracted text string
        """
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        try:
            raw_text = pytesseract.image_to_string(
                pil_img,
                lang=self.lang,
                config=self.DEFAULT_CONFIG
            )
            cleaned = self._basic_cleanup(raw_text)
            logger.debug(f"Extracted {len(cleaned.splitlines())} lines of text")
            return cleaned

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise

    def extract_with_confidence(self, image) -> list[dict]:
        """
        Extract text with per-word confidence scores.
        Useful for flagging uncertain extractions.
        
        Returns:
            List of dicts: [{"text": "Rice", "conf": 95, "line": 3}, ...]
        """
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        data = pytesseract.image_to_data(
            pil_img,
            lang=self.lang,
            config=self.DEFAULT_CONFIG,
            output_type=pytesseract.Output.DICT
        )

        results = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            conf = int(data["conf"][i])
            if text and conf > 30:  # Filter out very low-confidence noise
                results.append({
                    "text": text,
                    "confidence": conf,
                    "line_num": data["line_num"][i],
                    "block_num": data["block_num"][i],
                    "left": data["left"][i],
                    "top": data["top"][i],
                })

        return results

    def _basic_cleanup(self, text: str) -> str:
        """
        Basic cleanup of raw OCR output:
        - Remove excessive whitespace
        - Fix common OCR misreads (0 vs O, 1 vs l, etc.)
        - Normalize line endings
        """
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove lines that are just symbols/noise
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            # Keep line only if it has at least one alphanumeric character
            if stripped and re.search(r"[a-zA-Z0-9]", stripped):
                lines.append(stripped)

        return "\n".join(lines)
