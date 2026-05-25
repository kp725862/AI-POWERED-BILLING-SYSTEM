"""
classifier.py
-------------
AI/ML Module — Intelligence Extraction: Line Classifier
Distinguishes between different types of lines in a bill:
  - header    (shop name, address, phone)
  - date      (bill date)
  - item      (product name + price row)
  - total     (total/subtotal/tax row)
  - noise     (irrelevant lines)

Uses rule-based logic with scoring — no ML model needed for this,
which keeps it fast and transparent.

Author: AI/ML Intern
Project: AI-Powered Billing Management System
"""

import re
import logging

logger = logging.getLogger(__name__)


class LineClassifier:
    """
    Classifies each line of OCR text into a semantic category.
    Rule-based classifier using keyword + pattern scoring.
    """

    # Keywords that suggest a TOTAL row
    TOTAL_KEYWORDS = re.compile(
        r"\b(total|grand\s*total|net\s*total|amount|subtotal|balance|payable|due)\b",
        re.IGNORECASE
    )

    # Keywords that suggest a TAX row
    TAX_KEYWORDS = re.compile(
        r"\b(gst|tax|vat|cgst|sgst|igst|cess)\b",
        re.IGNORECASE
    )

    # Date patterns
    DATE_PATTERN = re.compile(
        r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b|\b\d{4}[\/\-]\d{2}[\/\-]\d{2}\b"
    )

    # Keywords for header lines (shop/business info)
    HEADER_KEYWORDS = re.compile(
        r"\b(store|shop|mart|traders|enterprises|pvt|ltd|medical|pharmacy|"
        r"general|super|market|bill|receipt|invoice|thank\s*you|welcome|"
        r"address|phone|mob|tel|email|fax|gstin|reg)\b",
        re.IGNORECASE
    )

    # Item row: typically has a product name + one or more numbers
    ITEM_PATTERN = re.compile(
        r"[a-zA-Z]{2,}.*\d+(?:[.,]\d{1,2})?"
    )

    def classify_lines(self, lines: list[str]) -> list[dict]:
        """
        Classify each line and return enriched list.
        
        Returns:
            List of dicts: [{text, label, score, rank}, ...]
        """
        classified = []
        header_count = 0

        for i, line in enumerate(lines):
            label, score = self._classify_single(line, position=i, total_lines=len(lines))

            if label == "header":
                rank = header_count
                header_count += 1
            else:
                rank = 0

            classified.append({
                "text": line,
                "label": label,
                "score": score,
                "rank": rank,
                "position": i,
            })

        return classified

    def _classify_single(self, line: str, position: int, total_lines: int) -> tuple[str, float]:
        """
        Classify a single line.
        Returns (label, confidence_score).
        """
        line = line.strip()
        if not line:
            return "noise", 0.0

        # --- Check for DATE ---
        if self.DATE_PATTERN.search(line):
            # Lines that are MOSTLY a date
            date_fraction = len(self.DATE_PATTERN.findall(line)) / max(len(line.split()), 1)
            if date_fraction > 0.3:
                return "date", 0.9

        # --- Check for TOTAL ---
        if self.TOTAL_KEYWORDS.search(line):
            return "total", 0.95

        # --- Check for TAX ---
        if self.TAX_KEYWORDS.search(line):
            return "tax", 0.90

        # --- Scoring for HEADER vs ITEM ---
        header_score = self._header_score(line, position, total_lines)
        item_score = self._item_score(line, position, total_lines)

        if header_score > item_score and header_score > 0.4:
            return "header", header_score
        elif item_score > 0.4:
            return "item", item_score
        elif position < 3:
            # First few lines are usually headers even if ambiguous
            return "header", 0.5
        else:
            return "noise", 0.1

    def _header_score(self, line: str, position: int, total_lines: int) -> float:
        """Score how likely this line is a header."""
        score = 0.0

        # Headers are usually near the top
        if position < 5:
            score += 0.3
        elif position < total_lines * 0.2:
            score += 0.1

        # Long lines without prices are often headers
        has_price = bool(re.search(r"\d{2,}", line))
        if not has_price:
            score += 0.2

        # Header keywords
        if self.HEADER_KEYWORDS.search(line):
            score += 0.4

        # ALL CAPS lines are often shop names
        alpha = re.sub(r"[^a-zA-Z]", "", line)
        if alpha and alpha.isupper() and len(alpha) > 3:
            score += 0.2

        return min(score, 1.0)

    def _item_score(self, line: str, position: int, total_lines: int) -> float:
        """Score how likely this line is a product item."""
        score = 0.0

        # Items are in the middle section of the bill
        relative_pos = position / max(total_lines, 1)
        if 0.2 < relative_pos < 0.8:
            score += 0.2

        # Items have both text and numbers
        has_alpha = bool(re.search(r"[a-zA-Z]{2,}", line))
        has_number = bool(re.search(r"\d+", line))
        if has_alpha and has_number:
            score += 0.4

        # Items match the item pattern
        if self.ITEM_PATTERN.search(line):
            score += 0.2

        # Items usually have a price-like number at the end
        if re.search(r"\d+(?:[.,]\d{2})?\s*$", line):
            score += 0.2

        return min(score, 1.0)
