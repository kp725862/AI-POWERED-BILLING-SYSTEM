"""
bill_scanner.py
---------------
AI/ML Module — Main Entry Point for OCR Pipeline
Combines ImagePreprocessor + OCREngine into one simple interface.

Author: AI/ML Intern
Project: AI-Powered Billing Management System
"""

import logging
from ocr_module.preprocessor import ImagePreprocessor
from ocr_module.ocr_engine import OCREngine
from ml_module.extractor import BillExtractor

logger = logging.getLogger(__name__)


class BillScanner:
    """
    Main scanner class.
    
    Usage:
        scanner = BillScanner()
        result = scanner.scan("bill.jpg")
        print(result)
    """

    def __init__(self, tesseract_path: str = None, lang: str = "eng", debug: bool = False):
        self.preprocessor = ImagePreprocessor(debug=debug)
        self.ocr = OCREngine(tesseract_path=tesseract_path, lang=lang)
        self.extractor = BillExtractor()
        self.debug = debug

    def scan(self, image_input) -> dict:
        """
        Full pipeline: image → preprocess → OCR → extract → structured JSON.
        
        Args:
            image_input: file path (str), bytes, PIL Image, or numpy array
        
        Returns:
            dict with keys: shop_name, date, items, subtotal, tax, total, raw_text
        """
        logger.info("=== Starting Bill Scan Pipeline ===")

        # Step 1: Pre-process the image
        logger.info("[1/3] Pre-processing image...")
        processed_img = self.preprocessor.preprocess(image_input)

        # Step 2: OCR — extract raw text
        logger.info("[2/3] Running OCR...")
        raw_text = self.ocr.extract_text(processed_img)

        if self.debug:
            print("--- RAW OCR OUTPUT ---")
            print(raw_text)
            print("----------------------")

        # Step 3: Intelligence Extraction — parse the raw text
        logger.info("[3/3] Extracting bill information...")
        result = self.extractor.extract(raw_text)
        result["raw_text"] = raw_text

        logger.info("=== Scan Complete ===")
        return result

    def scan_with_confidence(self, image_input) -> dict:
        """
        Extended scan that also returns per-word confidence scores.
        Useful for highlighting uncertain fields in the UI.
        """
        processed_img = self.preprocessor.preprocess(image_input)
        raw_text = self.ocr.extract_text(processed_img)
        word_data = self.ocr.extract_with_confidence(processed_img)

        result = self.extractor.extract(raw_text)
        result["raw_text"] = raw_text
        result["word_confidence"] = word_data
        result["avg_confidence"] = (
            sum(w["confidence"] for w in word_data) / len(word_data)
            if word_data else 0
        )

        return result
