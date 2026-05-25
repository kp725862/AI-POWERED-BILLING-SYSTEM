"""
extractor.py
------------
AI/ML Module — Intelligence Extraction
The "brain" of the system: parses raw OCR text and extracts
structured data (shop name, date, items, prices, quantities, total).

Author: AI/ML Intern
Project: AI-Powered Billing Management System
"""

import re
import logging
from dateutil import parser as date_parser
from ml_module.classifier import LineClassifier
from ml_module.postprocessor import DataPostprocessor

logger = logging.getLogger(__name__)


class BillExtractor:
    """
    Extracts structured bill data from raw OCR text using:
    - Regex patterns for prices, dates, quantities
    - LineClassifier to tell headers from line items
    - DataPostprocessor to clean and validate extracted data
    """

    # ------------------------------------------------------------------ #
    #  Regex Patterns
    # ------------------------------------------------------------------ #

    # Price patterns: 45.00 / Rs.45 / ₹45 / 45/-
    PRICE_PATTERN = re.compile(
        r"(?:Rs\.?|INR|₹)?\s*(\d{1,6}(?:[.,]\d{1,2})?)\s*(?:/-|Rs|INR)?",
        re.IGNORECASE
    )

    # Quantity patterns: 2 kg / 3 pcs / Qty: 4 / x2
    QUANTITY_PATTERN = re.compile(
        r"(?:qty[:\s]*|x\s*|×\s*)?(\d{1,4})\s*(?:kg|gm|g|pcs?|nos?|units?|ltrs?|l\b)?",
        re.IGNORECASE
    )

    # Date patterns: many formats
    DATE_PATTERN = re.compile(
        r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b"
    )

    # Total line patterns
    TOTAL_PATTERN = re.compile(
        r"\b(total|grand\s*total|net\s*total|amount\s*due|balance)\b.*?(\d{1,6}(?:[.,]\d{1,2})?)",
        re.IGNORECASE
    )

    # Tax patterns
    TAX_PATTERN = re.compile(
        r"\b(gst|tax|vat|cgst|sgst|igst)\b.*?(\d{1,4}(?:[.,]\d{1,2})?)",
        re.IGNORECASE
    )

    def __init__(self):
        self.classifier = LineClassifier()
        self.postprocessor = DataPostprocessor()

    def extract(self, raw_text: str) -> dict:
        """
        Main extraction method.
        
        Args:
            raw_text: cleaned OCR output string
        
        Returns:
            dict: {shop_name, date, items, subtotal, tax, total}
        """
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

        # Classify each line as header, item, total, date, or noise
        classified = self.classifier.classify_lines(lines)

        result = {
            "shop_name": self._extract_shop_name(classified),
            "date": self._extract_date(classified, raw_text),
            "items": self._extract_items(classified),
            "subtotal": None,
            "tax": self._extract_tax(raw_text),
            "total": self._extract_total(raw_text),
        }

        # Calculate subtotal from items if not found directly
        if result["items"] and result["total"] is None:
            result["total"] = sum(
                (item.get("price", 0) or 0) * (item.get("quantity", 1) or 1)
                for item in result["items"]
            )

        # Post-process: validate, clean, fix types
        result = self.postprocessor.clean(result)

        return result

    # ------------------------------------------------------------------ #
    #  Extraction Helpers
    # ------------------------------------------------------------------ #

    def _extract_shop_name(self, classified: list[dict]) -> str:
        """
        Shop name is typically the FIRST prominent header line
        (large text, all caps, at the top of the bill).
        """
        for item in classified:
            if item["label"] == "header" and item["rank"] == 0:
                return item["text"]

        # Fallback: first line that isn't a date or price
        for item in classified:
            if item["label"] == "header":
                return item["text"]

        return "Unknown Shop"

    def _extract_date(self, classified: list[dict], raw_text: str) -> str | None:
        """Extract bill date from classified lines or full text."""
        # Try classified date lines first
        for item in classified:
            if item["label"] == "date":
                return item["text"]

        # Fallback: regex on full text
        match = self.DATE_PATTERN.search(raw_text)
        if match:
            try:
                parsed = date_parser.parse(match.group(1), dayfirst=True)
                return parsed.strftime("%Y-%m-%d")
            except Exception:
                return match.group(1)

        return None

    def _extract_items(self, classified: list[dict]) -> list[dict]:
        """
        Extract line items (product rows).
        Each item dict: {name, quantity, unit, price}
        """
        items = []
        for line_data in classified:
            if line_data["label"] != "item":
                continue

            text = line_data["text"]
            item = self._parse_item_line(text)
            if item:
                items.append(item)

        return items

    def _parse_item_line(self, line: str) -> dict | None:
        """
        Parse a single item line like:
          "Rice         2 kg      90.00"
          "Sugar 1kg Rs.42"
          "Bread x2 @ 25 = 50"
        """
        # Find all numbers in the line
        numbers = re.findall(r"\d+(?:[.,]\d{1,2})?", line)
        if not numbers:
            return None

        # Last number is usually the price
        price = self._to_float(numbers[-1]) if numbers else None

        # Second-to-last number might be quantity (if there are 2+ numbers)
        quantity = None
        unit = None
        if len(numbers) >= 2:
            qty_match = self.QUANTITY_PATTERN.search(line)
            if qty_match:
                quantity = self._to_float(qty_match.group(1))
                # Extract unit if present
                unit_match = re.search(
                    r"\b(\d+)\s*(kg|gm|g|pcs?|nos?|units?|ltrs?|l)\b",
                    line, re.IGNORECASE
                )
                if unit_match:
                    unit = unit_match.group(2).lower()

        # Product name: text before the numbers start
        name = re.split(r"\d", line)[0].strip()
        name = re.sub(r"[^a-zA-Z\s\-\&]", "", name).strip()

        if not name or len(name) < 2:
            return None

        return {
            "name": name,
            "quantity": quantity or 1,
            "unit": unit,
            "price": price,
        }

    def _extract_total(self, raw_text: str) -> float | None:
        """Find the grand total amount."""
        match = self.TOTAL_PATTERN.search(raw_text)
        if match:
            return self._to_float(match.group(2))
        return None

    def _extract_tax(self, raw_text: str) -> float | None:
        """Find tax/GST amount."""
        match = self.TAX_PATTERN.search(raw_text)
        if match:
            return self._to_float(match.group(2))
        return None

    def _to_float(self, value: str) -> float | None:
        """Safely convert string number to float."""
        try:
            return float(str(value).replace(",", ""))
        except (ValueError, TypeError):
            return None
