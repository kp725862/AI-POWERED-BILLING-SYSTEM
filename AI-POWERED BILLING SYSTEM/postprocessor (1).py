"""
postprocessor.py
----------------
AI/ML Module — Data Post-processing
Cleans, validates, and structures the extracted bill data
before sending it to the frontend.

Author: AI/ML Intern
Project: AI-Powered Billing Management System
"""

import re
import logging

logger = logging.getLogger(__name__)


class DataPostprocessor:
    """
    Validates and cleans the extracted bill dictionary.
    Ensures correct data types, removes junk, fixes common errors.
    """

    # Common OCR misread corrections
    OCR_FIXES = {
        "0": "O",  # Zero vs letter O in names (handled per context)
        "1": "l",  # One vs lowercase L (handled per context)
        "5": "S",  # Five vs S
        "rn": "m",  # 'rn' often misread as 'm'
    }

    # Common product name noise words to remove
    NOISE_WORDS = re.compile(
        r"\b(pls|please|ref|no|number|sl|sr|s\.no|item|product|desc|description)\b",
        re.IGNORECASE
    )

    def clean(self, data: dict) -> dict:
        """
        Run all cleaning steps on extracted bill data.
        
        Args:
            data: raw extracted dict with possible None/bad values
        
        Returns:
            Cleaned and validated dict
        """
        data["shop_name"] = self._clean_shop_name(data.get("shop_name"))
        data["date"] = self._clean_date(data.get("date"))
        data["items"] = self._clean_items(data.get("items", []))
        data["total"] = self._clean_amount(data.get("total"))
        data["tax"] = self._clean_amount(data.get("tax"))
        data["subtotal"] = self._clean_amount(data.get("subtotal"))

        # Derive subtotal if missing
        if data["subtotal"] is None and data["items"]:
            data["subtotal"] = round(
                sum(
                    (item.get("price") or 0) * (item.get("quantity") or 1)
                    for item in data["items"]
                ), 2
            )

        # Derive total from subtotal + tax if missing
        if data["total"] is None and data["subtotal"] is not None:
            tax = data["tax"] or 0
            data["total"] = round(data["subtotal"] + tax, 2)

        logger.debug(f"Post-processed: {len(data['items'])} items, total={data['total']}")
        return data

    # ------------------------------------------------------------------ #
    #  Cleaners
    # ------------------------------------------------------------------ #

    def _clean_shop_name(self, name: str | None) -> str:
        """Clean and title-case shop name."""
        if not name:
            return "Unknown Shop"

        # Remove non-printable chars
        name = re.sub(r"[^\x20-\x7E]", "", name)
        # Remove noise words
        name = self.NOISE_WORDS.sub("", name)
        # Remove leading/trailing punctuation
        name = name.strip(".,:-|/\\")
        # Collapse multiple spaces
        name = re.sub(r"\s+", " ", name).strip()
        # Title case
        name = name.title()

        return name if name else "Unknown Shop"

    def _clean_date(self, date_str: str | None) -> str | None:
        """Standardize date to YYYY-MM-DD format."""
        if not date_str:
            return None

        # Already in standard format
        if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
            return date_str

        # Try common Indian formats: DD/MM/YYYY, DD-MM-YYYY
        match = re.match(r"(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})", date_str)
        if match:
            day, month, year = match.group(1), match.group(2), match.group(3)
            if len(year) == 2:
                year = "20" + year
            try:
                return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
            except ValueError:
                pass

        return date_str

    def _clean_items(self, items: list) -> list:
        """Clean each item in the list."""
        cleaned = []
        seen_names = set()

        for item in items:
            if not isinstance(item, dict):
                continue

            name = self._clean_item_name(item.get("name", ""))
            if not name or len(name) < 2:
                continue

            # Deduplicate
            if name.lower() in seen_names:
                continue
            seen_names.add(name.lower())

            price = self._clean_amount(item.get("price"))
            quantity = self._clean_quantity(item.get("quantity"))

            cleaned.append({
                "name": name,
                "quantity": quantity,
                "unit": item.get("unit"),
                "price": price,
                "line_total": round(price * quantity, 2) if price and quantity else None,
            })

        return cleaned

    def _clean_item_name(self, name: str) -> str:
        """Clean a product name."""
        if not name:
            return ""

        # Remove noise words
        name = self.NOISE_WORDS.sub("", name)
        # Remove special characters but keep hyphens and &
        name = re.sub(r"[^a-zA-Z0-9\s\-\&]", "", name)
        # Collapse spaces
        name = re.sub(r"\s+", " ", name).strip()
        # Title case
        name = name.title()

        return name

    def _clean_amount(self, value) -> float | None:
        """Convert to float and round to 2 decimal places."""
        if value is None:
            return None
        try:
            return round(float(str(value).replace(",", "")), 2)
        except (ValueError, TypeError):
            return None

    def _clean_quantity(self, value) -> float:
        """Clean quantity — default to 1 if missing/invalid."""
        if value is None:
            return 1
        try:
            qty = float(str(value).replace(",", ""))
            return max(qty, 1)
        except (ValueError, TypeError):
            return 1
