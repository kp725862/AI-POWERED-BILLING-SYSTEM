"""
app.py
------
FastAPI REST API — connects the AI/ML pipeline to the frontend.

Endpoints:
  POST /api/scan-bill     → Upload image, get structured bill data
  POST /api/scan-text     → Send raw text, get structured extraction
  GET  /api/health        → Health check

Author: AI/ML Intern
Project: AI-Powered Billing Management System
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Billing Management — ML API",
    description="OCR + Intelligence Extraction API for bill scanning",
    version="1.0.0",
)

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load scanner to avoid startup delay if Tesseract isn't installed
_scanner = None

def get_scanner():
    global _scanner
    if _scanner is None:
        from ocr_module.bill_scanner import BillScanner
        _scanner = BillScanner()
    return _scanner


# ------------------------------------------------------------------ #
#  Response Models
# ------------------------------------------------------------------ #

class BillItem(BaseModel):
    name: str
    quantity: float
    unit: str | None
    price: float | None
    line_total: float | None

class BillResponse(BaseModel):
    shop_name: str
    date: str | None
    items: list[BillItem]
    subtotal: float | None
    tax: float | None
    total: float | None
    raw_text: str
    processing_time_ms: int

class TextExtractionRequest(BaseModel):
    text: str


# ------------------------------------------------------------------ #
#  Endpoints
# ------------------------------------------------------------------ #

@app.get("/api/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "AI Billing ML API", "version": "1.0.0"}


@app.post("/api/scan-bill", response_model=BillResponse)
async def scan_bill(image: UploadFile = File(...)):
    """
    Upload a bill image and get structured JSON back.
    
    - Accepts: JPG, PNG, WEBP images
    - Returns: shop name, date, items list, totals
    """
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/tiff"}
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {image.content_type}. Use JPG, PNG, or WEBP."
        )

    # Read image bytes
    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file.")

    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="Image too large. Max 10MB.")

    # Run the AI/ML pipeline
    start = time.time()
    try:
        scanner = get_scanner()
        result = scanner.scan(image_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Scan failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Bill scanning failed. Please try again.")

    elapsed_ms = int((time.time() - start) * 1000)
    logger.info(f"Scanned bill in {elapsed_ms}ms — found {len(result.get('items', []))} items")

    return BillResponse(
        shop_name=result.get("shop_name", "Unknown"),
        date=result.get("date"),
        items=result.get("items", []),
        subtotal=result.get("subtotal"),
        tax=result.get("tax"),
        total=result.get("total"),
        raw_text=result.get("raw_text", ""),
        processing_time_ms=elapsed_ms,
    )


@app.post("/api/scan-text")
def scan_text(request: TextExtractionRequest):
    """
    Extract bill data from raw text (already OCR'd).
    Useful for testing extraction without an image.
    """
    from ml_module.extractor import BillExtractor
    extractor = BillExtractor()

    try:
        result = extractor.extract(request.text)
        result["raw_text"] = request.text
        return result
    except Exception as e:
        logger.error(f"Text extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Extraction failed.")


# ------------------------------------------------------------------ #
#  Run with: uvicorn api.app:app --reload
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
