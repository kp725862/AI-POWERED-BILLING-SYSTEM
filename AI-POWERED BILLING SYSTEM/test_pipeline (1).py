"""
test_pipeline.py
----------------
Unit tests for the AI/ML pipeline.
Tests each module independently and the full pipeline.

Run with:
    pytest tests/test_pipeline.py -v

Author: AI/ML Intern
Project: AI-Powered Billing Management System
"""

import pytest
from ml_module.extractor import BillExtractor
from ml_module.classifier import LineClassifier
from ml_module.postprocessor import DataPostprocessor


# ------------------------------------------------------------------ #
#  Sample test data (simulates OCR output from a real bill)
# ------------------------------------------------------------------ #

SAMPLE_BILL_TEXT = """
KRISHNA GENERAL STORE
Near Bus Stand, Pune - 411001
Mobile: 9876543210

Date: 20/05/2024
Bill No: 142

Rice           2 kg     90.00
Sugar          1 kg     42.00
Bread          2 pcs    50.00
Milk           1 ltr    28.00
Biscuits       3 pcs    30.00

Subtotal              240.00
GST (5%)               12.00
Total                 252.00

Thank You! Visit Again
"""

MINIMAL_BILL_TEXT = """
Shop ABC
05/01/2024
Apple 3 pcs 60
Banana 12 36
Total 96
"""


# ------------------------------------------------------------------ #
#  Classifier Tests
# ------------------------------------------------------------------ #

class TestLineClassifier:

    def setup_method(self):
        self.clf = LineClassifier()

    def test_classifies_header_lines(self):
        lines = ["KRISHNA GENERAL STORE", "Near Bus Stand, Pune"]
        result = self.clf.classify_lines(lines)
        assert result[0]["label"] == "header"

    def test_classifies_item_lines(self):
        lines = ["Rice 2 kg 90.00"]
        result = self.clf.classify_lines(lines)
        assert result[0]["label"] == "item"

    def test_classifies_total_line(self):
        lines = ["Total 252.00"]
        result = self.clf.classify_lines(lines)
        assert result[0]["label"] == "total"

    def test_classifies_date_line(self):
        lines = ["Date: 20/05/2024"]
        result = self.clf.classify_lines(lines)
        assert result[0]["label"] == "date"

    def test_first_header_has_rank_zero(self):
        lines = ["SHOP NAME", "Address line"]
        result = self.clf.classify_lines(lines)
        headers = [r for r in result if r["label"] == "header"]
        assert headers[0]["rank"] == 0


# ------------------------------------------------------------------ #
#  Extractor Tests
# ------------------------------------------------------------------ #

class TestBillExtractor:

    def setup_method(self):
        self.extractor = BillExtractor()

    def test_extracts_shop_name(self):
        result = self.extractor.extract(SAMPLE_BILL_TEXT)
        assert "krishna" in result["shop_name"].lower() or result["shop_name"] != "Unknown Shop"

    def test_extracts_date(self):
        result = self.extractor.extract(SAMPLE_BILL_TEXT)
        assert result["date"] is not None
        assert "2024" in str(result["date"])

    def test_extracts_items(self):
        result = self.extractor.extract(SAMPLE_BILL_TEXT)
        assert isinstance(result["items"], list)
        assert len(result["items"]) >= 3

    def test_item_has_required_fields(self):
        result = self.extractor.extract(SAMPLE_BILL_TEXT)
        for item in result["items"]:
            assert "name" in item
            assert "quantity" in item
            assert "price" in item

    def test_extracts_total(self):
        result = self.extractor.extract(SAMPLE_BILL_TEXT)
        assert result["total"] is not None
        assert result["total"] > 0

    def test_extracts_tax(self):
        result = self.extractor.extract(SAMPLE_BILL_TEXT)
        assert result["tax"] is not None

    def test_handles_minimal_bill(self):
        result = self.extractor.extract(MINIMAL_BILL_TEXT)
        assert isinstance(result["items"], list)

    def test_handles_empty_text(self):
        result = self.extractor.extract("")
        assert result["shop_name"] == "Unknown Shop"
        assert result["items"] == []


# ------------------------------------------------------------------ #
#  Postprocessor Tests
# ------------------------------------------------------------------ #

class TestDataPostprocessor:

    def setup_method(self):
        self.pp = DataPostprocessor()

    def test_cleans_shop_name(self):
        data = {"shop_name": "  KRISHNA STORE!!  ", "date": None,
                "items": [], "total": None, "tax": None, "subtotal": None}
        result = self.pp.clean(data)
        assert result["shop_name"] == "Krishna Store"

    def test_standardizes_date_format(self):
        data = {"shop_name": "Shop", "date": "20/05/2024",
                "items": [], "total": None, "tax": None, "subtotal": None}
        result = self.pp.clean(data)
        assert result["date"] == "2024-05-20"

    def test_cleans_item_prices(self):
        data = {
            "shop_name": "Shop", "date": None,
            "items": [{"name": "Rice", "quantity": 2, "price": "90.00", "unit": "kg"}],
            "total": None, "tax": None, "subtotal": None
        }
        result = self.pp.clean(data)
        assert isinstance(result["items"][0]["price"], float)
        assert result["items"][0]["price"] == 90.0

    def test_calculates_subtotal_from_items(self):
        data = {
            "shop_name": "Shop", "date": None,
            "items": [
                {"name": "Rice", "quantity": 2, "price": 45.0, "unit": None},
                {"name": "Sugar", "quantity": 1, "price": 42.0, "unit": None},
            ],
            "total": None, "tax": None, "subtotal": None
        }
        result = self.pp.clean(data)
        assert result["subtotal"] == 132.0

    def test_default_quantity_is_one(self):
        data = {
            "shop_name": "Shop", "date": None,
            "items": [{"name": "Bread", "quantity": None, "price": 25.0, "unit": None}],
            "total": None, "tax": None, "subtotal": None
        }
        result = self.pp.clean(data)
        assert result["items"][0]["quantity"] == 1

    def test_handles_none_shop_name(self):
        data = {"shop_name": None, "date": None,
                "items": [], "total": None, "tax": None, "subtotal": None}
        result = self.pp.clean(data)
        assert result["shop_name"] == "Unknown Shop"


# ------------------------------------------------------------------ #
#  Full Pipeline Integration Test (no image needed)
# ------------------------------------------------------------------ #

class TestFullPipeline:

    def test_end_to_end_text_extraction(self):
        """Test the full extraction pipeline on sample bill text."""
        extractor = BillExtractor()
        result = extractor.extract(SAMPLE_BILL_TEXT)

        # Core fields exist
        assert "shop_name" in result
        assert "date" in result
        assert "items" in result
        assert "total" in result

        # Types are correct
        assert isinstance(result["items"], list)
        assert result["total"] is None or isinstance(result["total"], float)

    def test_multiple_bills(self):
        """Test pipeline handles different bill formats."""
        extractor = BillExtractor()
        for bill_text in [SAMPLE_BILL_TEXT, MINIMAL_BILL_TEXT]:
            result = extractor.extract(bill_text)
            assert "items" in result
            assert isinstance(result["items"], list)
