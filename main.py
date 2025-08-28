#!/usr/bin/env python3
"""
Enhanced Production-Ready Hand Gesture Recognition Service
"""

import os
import time
import uuid
import base64
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Depends
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# Import our enhanced modules
from config import settings
from logger import setup_logging, get_logger, RequestLogger
from monitoring import metrics, MetricsMiddleware, get_health_metrics
from rate_limiter import RateLimitMiddleware, check_rate_limit_dependency
from batch_processor import BatchProcessor, BatchRequest, validate_batch_request
from ml_processor import ml_processor

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Global startup time for uptime calculation
startup_time = time.time()


def get_health_metrics() -> Dict[str, Any]:
    """Get current health metrics"""
    return {
        "requests_total": metrics.get_request_count(),
        "requests_per_second": metrics.get_requests_per_second(),
        "average_response_time": metrics.get_average_response_time(),
        "error_rate": metrics.get_error_rate(),
        "ml_inference_count": metrics.get_ml_inference_count(),
        "cache_hit_rate": metrics.get_cache_hit_rate()
    }


# Request/Response Models
class FingerDetectionRequest(BaseModel):
    """Request model for single finger detection (base64 or direct image)"""
    image_base64: str = Field(None, description="Base64 encoded image (optional)")
    
    @validator('image_base64')
    def validate_image_base64(cls, v):
        if v is not None and (not v or len(v) == 0):
            raise ValueError('image_base64 cannot be empty if provided')
        return v


class FingerDetectionResponse(BaseModel):
    """Response model for finger detection"""
    finger_count: int = Field(..., description="Number of fingers detected")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    detected_gesture: str = Field(..., description="Detected hand gesture")
    success: bool = Field(..., description="Whether detection was successful")
    request_id: str = Field(..., description="Unique request identifier")


class BatchDetectionRequest(BaseModel):
    """Request model for batch finger detection"""
    images: List[BatchRequest] = Field(..., description="List of images to process")
    
    @validator('images')
    def validate_images(cls, v):
        if not v:
            raise ValueError('images list cannot be empty')
        if len(v) > settings.batch_size:
            raise ValueError(f'batch size {len(v)} exceeds maximum {settings.batch_size}')
        return v


class BatchDetectionResponse(BaseModel):
    """Response model for batch detection"""
    batch_id: str = Field(..., description="Unique batch identifier")
    total_images: int = Field(..., description="Total number of images processed")
    successful_images: int = Field(..., description="Number of successfully processed images")
    failed_images: int = Field(..., description="Number of failed images")
    total_processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    average_processing_time_ms: float = Field(..., description="Average processing time per image")
    results: List[Dict[str, Any]] = Field(..., description="Individual image results")
    summary: Dict[str, Any] = Field(..., description="Batch processing summary")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    models_loaded: bool = Field(..., description="Whether ML models are loaded")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    metrics: Dict[str, Any] = Field(..., description="Current metrics")


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_loaded: bool = Field(..., description="Whether model is loaded")
    mediapipe_initialized: bool = Field(..., description="MediaPipe initialization status")
    classifier_loaded: bool = Field(..., description="Classifier loading status")
    available_labels: List[str] = Field(..., description="Available gesture labels")
    redis_cache_enabled: bool = Field(..., description="Redis cache status")
    detection_confidence: float = Field(..., description="Detection confidence threshold")
    tracking_confidence: float = Field(..., description="Tracking confidence threshold")
    max_hands: int = Field(..., description="Maximum number of hands to detect")


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Hand Gesture Recognition Service...")
    
    # Start metrics server
    metrics.start_metrics_server()
    
    # Initialize ML processor
    try:
        if ml_processor.model_loaded:
            logger.info("Service initialized successfully!")
        else:
            logger.error("Failed to initialize ML models. Service may not work properly.")
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Hand Gesture Recognition Service...")
    
    # Cleanup resources
    try:
        await ml_processor.cleanup()
        batch_processor.cleanup()
        await rate_limiter.cleanup()
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Production-ready hand gesture recognition service with monitoring, caching, and batch processing",
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(MetricsMiddleware)
app.add_middleware(RateLimitMiddleware)

# Global variables
startup_time = time.time()


# Utility functions
def preprocess_image(image_bytes: bytes):
    """Preprocess image for ML inference"""
    try:
        import cv2
        import numpy as np
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Convert BGR to RGB (MediaPipe expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image_rgb
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise


# API Endpoints
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint"""
    return {
        "message": "Hand Gesture Recognition Service",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "metrics": f"/metrics"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint"""
    uptime = time.time() - startup_time
    
    health_data = {
        "status": "healthy" if ml_processor.model_loaded else "degraded",
        "service": "hand-gesture-recognition-service",
        "version": settings.app_version,
        "models_loaded": ml_processor.model_loaded,
        "uptime_seconds": uptime,
        "metrics": get_health_metrics()
    }
    
    return health_data


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return Response(
        content=metrics.get_metrics(),
        media_type="text/plain"
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get detailed model information"""
    return ml_processor.get_model_info()


@app.post("/detect-fingers", response_model=FingerDetectionResponse)
async def detect_fingers(
    file: UploadFile = File(None),
    request: FingerDetectionRequest = None
):
    """Detect finger count in image file or base64 encoded image"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Create request logger
    req_logger = RequestLogger(request_id)
    req_logger.log_request("POST", "/detect-fingers")
    
    try:
        image_bytes = None
        
        # Handle file upload (priority)
        if file:
            # Validate file type
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            # Read uploaded file
            image_bytes = await file.read()
            
            # Validate file size
            if len(image_bytes) > settings.max_image_size:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File size {len(image_bytes)} exceeds maximum {settings.max_image_size}"
                )
        
        # Handle base64 if no file provided
        elif request and request.image_base64:
            # Validate image size
            if len(request.image_base64) > settings.max_image_size:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Image size {len(request.image_base64)} exceeds maximum {settings.max_image_size}"
                )
            
            # Decode base64 image
            image_bytes = base64.b64decode(request.image_base64)
        
        else:
            raise HTTPException(
                status_code=400, 
                detail="Either image file or base64 image must be provided"
            )
        
        # Preprocess image
        image = preprocess_image(image_bytes)
        
        # Detect fingers
        result = ml_processor.detect_fingers_in_image(image)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        response = FingerDetectionResponse(
            finger_count=result["finger_count"],
            confidence=result["confidence"],
            processing_time_ms=processing_time_ms,
            detected_gesture=result["detected_gesture"],
            success=True,
            request_id=request_id
        )
        
        # Log response
        req_logger.log_response(200, processing_time_ms)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e) if e else "Unknown error occurred"
        logger.error(f"Request processing failed: {error_msg}")
        
        # Log error
        req_logger.log_error(e)
        
        # Record error metrics
        metrics.record_error(type(e).__name__, "detect_fingers")
        
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_msg}")


@app.post("/detect-fingers-upload")
async def detect_fingers_upload(
    file: UploadFile = File(...)
):
    """Detect finger count in uploaded image file"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Create request logger
    req_logger = RequestLogger(request_id)
    req_logger.log_request("POST", "/detect-fingers-upload")
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read uploaded file
        image_bytes = await file.read()
        
        # Validate file size
        if len(image_bytes) > settings.max_image_size:
            raise HTTPException(
                status_code=400, 
                detail=f"File size {len(image_bytes)} exceeds maximum {settings.max_image_size}"
            )
        
        # Preprocess image
        image = preprocess_image(image_bytes)
        
        # Detect fingers
        result = ml_processor.detect_fingers_in_image(image)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        response = {
            "finger_count": result["finger_count"],
            "confidence": result["confidence"],
            "processing_time_ms": processing_time_ms,
            "detected_gesture": result["detected_gesture"],
            "success": True,
            "request_id": request_id
        }
        
        # Log response
        req_logger.log_response(200, processing_time_ms)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e) if e else "Unknown error occurred"
        logger.error(f"Request processing failed: {error_msg}")
        
        # Log error
        req_logger.log_error(e)
        
        # Record error metrics
        metrics.record_error(type(e).__name__, "detect_fingers_upload")
        
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_msg}")


@app.post("/detect-fingers-batch", response_model=BatchDetectionResponse)
async def detect_fingers_batch(
    request: BatchDetectionRequest
):
    """Detect finger counts in multiple base64 encoded images (batch processing)"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Create request logger
    req_logger = RequestLogger(request_id)
    req_logger.log_request("POST", "/detect-fingers-batch", batch_size=len(request.images))
    
    try:
        # Validate batch request
        validate_batch_request(request.images)
        
        # Process batch
        batch_result = await batch_processor.process_batch(request.images, ml_processor)
        
        # Log response
        req_logger.log_response(200, batch_result.total_processing_time_ms)
        
        return BatchDetectionResponse(
            batch_id=batch_result.batch_id,
            total_images=batch_result.total_images,
            successful_images=batch_result.successful_images,
            failed_images=batch_result.failed_images,
            total_processing_time_ms=batch_result.total_processing_time_ms,
            average_processing_time_ms=batch_result.average_processing_time_ms,
            results=[vars(result) for result in batch_result.results],
            summary=batch_result.summary
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        error_msg = str(e) if e else "Unknown error occurred"
        logger.error(f"Batch processing failed: {error_msg}")
        
        # Log error
        req_logger.log_error(e)
        
        # Record error metrics
        metrics.record_error(type(e).__name__, "detect_fingers_batch")
        
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_msg}")


@app.post("/detect-fingers-batch-files")
async def detect_fingers_batch_files(
    files: List[UploadFile] = File(...)
):
    """Detect finger counts in multiple uploaded image files (batch processing)"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Create request logger
    req_logger = RequestLogger(request_id)
    req_logger.log_request("POST", "/detect-fingers-batch-files", batch_size=len(files))
    
    try:
        # Validate batch size
        if len(files) > settings.batch_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Batch size {len(files)} exceeds maximum {settings.batch_size}"
            )
        
        # Validate files
        for i, file in enumerate(files):
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {i+1} must be an image, got {file.content_type}"
                )
        
        # Process files
        results = []
        total_processing_time = 0
        
        for i, file in enumerate(files):
            try:
                # Read file
                image_bytes = await file.read()
                
                # Validate file size
                if len(image_bytes) > settings.max_image_size:
                    results.append({
                        "file_index": i,
                        "filename": file.filename,
                        "success": False,
                        "error": f"File size {len(image_bytes)} exceeds maximum {settings.max_image_size}"
                    })
                    continue
                
                # Preprocess image
                image = preprocess_image(image_bytes)
                
                # Detect fingers
                result = ml_processor.detect_fingers_in_image(image)
                
                if result["success"]:
                    results.append({
                        "file_index": i,
                        "filename": file.filename,
                        "success": True,
                        "finger_count": result["finger_count"],
                        "confidence": result["confidence"],
                        "detected_gesture": result["detected_gesture"]
                    })
                else:
                    results.append({
                        "file_index": i,
                        "filename": file.filename,
                        "success": False,
                        "error": result.get("error", "Unknown error")
                    })
                    
            except Exception as e:
                results.append({
                    "file_index": i,
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
        
        # Calculate processing time
        total_processing_time_ms = int((time.time() - start_time) * 1000)
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        # Create summary
        summary = {
            "gesture_distribution": {},
            "error_distribution": {},
            "success_rate": len(successful) / len(results) if results else 0.0,
            "average_confidence": 0.0,
            "min_processing_time_ms": total_processing_time_ms,
            "max_processing_time_ms": total_processing_time_ms,
            "median_processing_time_ms": total_processing_time_ms
        }
        
        # Calculate gesture distribution and confidence
        if successful:
            confidence_scores = []
            for result in successful:
                gesture = result["detected_gesture"]
                summary["gesture_distribution"][gesture] = summary["gesture_distribution"].get(gesture, 0) + 1
                confidence_scores.append(result["confidence"])
            
            if confidence_scores:
                summary["average_confidence"] = sum(confidence_scores) / len(confidence_scores)
        
        # Calculate error distribution
        if failed:
            for result in failed:
                error_type = result.get("error", "Unknown")
                summary["error_distribution"][error_type] = summary["error_distribution"].get(error_type, 0) + 1
        
        # Log response
        req_logger.log_response(200, total_processing_time_ms)
        
        return {
            "batch_id": f"file_batch_{request_id[:8]}",
            "total_files": len(files),
            "successful_files": len(successful),
            "failed_files": len(failed),
            "total_processing_time_ms": total_processing_time_ms,
            "average_processing_time_ms": total_processing_time_ms / len(files) if files else 0.0,
            "results": results,
            "summary": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e) if e else "Unknown error occurred"
        logger.error(f"Batch file processing failed: {error_msg}")
        
        # Log error
        req_logger.log_error(e)
        
        # Record error metrics
        metrics.record_error(type(e).__name__, "detect_fingers_batch_files")
        
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_msg}")


@app.get("/status")
async def get_status():
    """Get comprehensive service status"""
    uptime = time.time() - startup_time
    
    return {
        "service": "hand-gesture-recognition-service",
        "version": settings.app_version,
        "status": "running",
        "uptime_seconds": uptime,
        "models": ml_processor.get_model_info(),
        "configuration": {
            "rate_limit_enabled": settings.rate_limit_enabled,
            "rate_limit_requests": settings.rate_limit_requests,
            "rate_limit_window": settings.rate_limit_window,
            "batch_size": settings.batch_size,
            "max_image_size": settings.max_image_size,
            "metrics_enabled": settings.metrics_enabled,
            "redis_enabled": settings.redis_enabled
        },
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Service information"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/metrics", "method": "GET", "description": "Prometheus metrics"},
            {"path": "/model/info", "method": "GET", "description": "Model information"},
            {"path": "/detect-fingers", "method": "POST", "description": "Single image detection (file or base64)"},
            {"path": "/detect-fingers-upload", "method": "POST", "description": "File upload detection (legacy)"},
            {"path": "/detect-fingers-batch", "method": "POST", "description": "Batch base64 image detection"},
            {"path": "/detect-fingers-batch-files", "method": "POST", "description": "Batch file upload detection"},
            {"path": "/status", "method": "GET", "description": "Service status"}
        ]
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
        access_log=True
    )
