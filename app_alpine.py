#!/usr/bin/env python3
"""
Alpine Linux Compatible Hand Gesture Recognition Service
A simplified version that works with Alpine Linux Docker
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import json
import logging
from datetime import datetime
import time
from typing import Optional
import io
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Alpine Hand Gesture Recognition Service",
    description="A lightweight hand gesture recognition service compatible with Alpine Linux",
    version="1.0.0-alpine"
)

class ImageRequest(BaseModel):
    image_base64: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    alpine_compatible: bool

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint"""
    return {
        "message": "Alpine Hand Gesture Recognition Service",
        "status": "running",
        "version": "1.0.0-alpine",
        "note": "This is a lightweight version for Alpine Linux compatibility"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0-alpine",
        alpine_compatible=True
    )

@app.get("/docs")
async def get_docs():
    """Get API documentation"""
    return {
        "message": "API Documentation",
        "endpoints": {
            "/": "Root endpoint",
            "/health": "Health check",
            "/detect-fingers": "Detect fingers in base64 image",
            "/detect-fingers-upload": "Detect fingers in uploaded image"
        },
        "note": "This service is optimized for Alpine Linux Docker containers"
    }

def process_image_alpine(image_data: bytes) -> dict:
    """
    Process image using Alpine-compatible libraries only
    This is a simplified version that demonstrates the service structure
    """
    try:
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Get basic image info
        width, height = image.size
        mode = image.mode
        
        # Simple image analysis (placeholder for actual ML processing)
        # In a real implementation, you'd use Alpine-compatible ML libraries
        
        # For demonstration, return basic image analysis
        return {
            "success": True,
            "image_info": {
                "width": width,
                "height": height,
                "mode": mode,
                "size_bytes": len(image_data)
            },
            "gesture_prediction": "demo_mode",
            "confidence": 0.95,
            "processing_method": "alpine_compatible"
        }
        
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/detect-fingers")
async def detect_fingers(request: ImageRequest):
    """Detect fingers in base64 encoded image"""
    start_time = time.time()
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        
        # Process image
        result = process_image_alpine(image_data)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "success": True,
            "detected_gesture": result.get("gesture_prediction", "unknown"),
            "confidence": result.get("confidence", 0.0),
            "processing_time_ms": processing_time,
            "image_info": result.get("image_info", {}),
            "note": "Alpine Linux compatible version - simplified processing"
        }
        
    except Exception as e:
        error_msg = str(e) if e else "Unknown error occurred"
        logger.error(f"Request processing failed: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_msg}")

@app.post("/detect-fingers-upload")
async def detect_fingers_upload(file: UploadFile = File(...)):
    """Detect fingers in uploaded image file"""
    start_time = time.time()
    
    try:
        # Read uploaded file
        image_data = await file.read()
        
        # Process image
        result = process_image_alpine(image_data)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "success": True,
            "detected_gesture": result.get("gesture_prediction", "unknown"),
            "confidence": result.get("confidence", 0.0),
            "processing_time_ms": processing_time,
            "image_info": result.get("image_info", {}),
            "note": "Alpine Linux compatible version - simplified processing"
        }
        
    except Exception as e:
        error_msg = str(e) if e else "Unknown error occurred"
        logger.error(f"Request processing failed: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_msg}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
