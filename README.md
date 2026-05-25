# 🤖 AI/ML Module — AI-Powered Billing Management System

**Intern:** AI/ML Domain  
**Project:** Project 1 — AI-Powered Billing Management  
**Role Contribution:** OCR Scanning, Intelligence Extraction & Data Pre-processing

---

## 📌 What This Module Does

This module covers the **complete AI/ML backend** for the billing system:

| Feature | Description |
|---|---|
| 🔍 OCR Scanning | Reads physical paper bills using Tesseract/OpenCV |
| 🧠 Intelligence Extraction | Extracts Shop Name, Date, Products, Price, Quantity |
| 🖼️ Data Pre-processing | Enhances low-quality images before scanning |
| 📊 Structured Output | Returns clean JSON for the frontend to consume |

---

## 📁 Project Structure

```
ai_billing_project/
│
├── ocr_module/
│   ├── preprocessor.py        # Image enhancement & cleanup
│   ├── ocr_engine.py          # Tesseract OCR wrapper
│   └── bill_scanner.py        # Main scanner (combines both)
│
├── ml_module/
│   ├── extractor.py           # Intelligence extraction logic
│   ├── classifier.py          # Header vs line-item classifier
│   └── postprocessor.py       # Clean & structure extracted data
│
├── api/
│   └── app.py                 # FastAPI server (connects to frontend)
│
├── tests/
│   └── test_pipeline.py       # Unit tests for all modules
│
├── notebooks/
│   └── demo.ipynb             # Jupyter demo notebook
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-billing-ml.git
cd ai-billing-ml

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Tesseract OCR engine
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows: Download from https://github.com/tesseract-ocr/tesseract

# 5. Run the API server
uvicorn api.app:app --reload
```

---

## 🚀 Quick Usage

```python
from ocr_module.bill_scanner import BillScanner

scanner = BillScanner()
result = scanner.scan("path/to/bill_image.jpg")
print(result)
# Output:
# {
#   "shop_name": "Krishna General Store",
#   "date": "2024-05-20",
#   "items": [
#     {"name": "Rice", "quantity": 2, "price": 45.00},
#     {"name": "Sugar", "quantity": 1, "price": 42.00}
#   ],
#   "total": 132.00
# }
```

---

## 🔌 API Endpoint

```
POST /api/scan-bill
Content-Type: multipart/form-data
Body: image file

Returns: JSON with extracted bill data
```

---

## 👨‍💻 Technologies Used

- **Python 3.10+**
- **Tesseract OCR** — text recognition engine
- **OpenCV** — image processing
- **Pillow** — image handling
- **FastAPI** — REST API
- **Regex + NLP** — data extraction logic
- **pytest** — testing
