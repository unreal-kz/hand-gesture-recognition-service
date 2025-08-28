#!/usr/bin/env python3
"""
Batch processing for multiple hand gesture images
"""

import asyncio
import time
import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import base64
import cv2
import numpy as np

from config import settings
from logger import get_logger, log_function_call, log_performance_metric
from monitoring import metrics

logger = get_logger(__name__)


@dataclass
class BatchRequest:
    """Single image request within a batch"""
    image_id: str
    image_base64: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchResponse:
    """Response for a single image in batch"""
    image_id: str
    success: bool
    finger_count: Optional[int] = None
    confidence: Optional[float] = None
    detected_gesture: Optional[str] = None
    processing_time_ms: Optional[int] = None
    error: Optional[str] = None


@dataclass
class BatchResult:
    """Complete batch processing result"""
    batch_id: str
    total_images: int
    successful_images: int
    failed_images: int
    total_processing_time_ms: int
    average_processing_time_ms: float
    results: List[BatchResponse]
    summary: Dict[str, Any]


class BatchProcessor:
    """Batch processor for multiple hand gesture images"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.batch_counter = 0
        
        logger.info(f"Batch processor initialized with {self.max_workers} workers")
    
    async def process_batch(
        self, 
        requests: List[BatchRequest],
        ml_processor
    ) -> BatchResult:
        """Process a batch of images concurrently"""
        start_time = time.time()
        batch_id = f"batch_{self.batch_counter:06d}"
        self.batch_counter += 1
        
        log_function_call("process_batch", batch_id=batch_id, batch_size=len(requests))
        
        if not requests:
            return BatchResult(
                batch_id=batch_id,
                total_images=0,
                successful_images=0,
                failed_images=0,
                total_processing_time_ms=0,
                average_processing_time_ms=0.0,
                results=[],
                summary={}
            )
        
        # Validate batch size
        if len(requests) > settings.batch_size:
            raise ValueError(f"Batch size {len(requests)} exceeds maximum {settings.batch_size}")
        
        # Process images concurrently
        loop = asyncio.get_event_loop()
        tasks = [
            self._process_single_image(loop, request, ml_processor)
            for request in requests
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate metrics
        total_time_ms = int((time.time() - start_time) * 1000)
        successful = [r for r in results if isinstance(r, BatchResponse) and r.success]
        failed = [r for r in results if isinstance(r, BatchResponse) and not r.success]
        
        # Record metrics
        metrics.record_batch_processing(len(requests), total_time_ms / 1000.0)
        
        # Create summary
        summary = self._create_batch_summary(results, successful, failed)
        
        batch_result = BatchResult(
            batch_id=batch_id,
            total_images=len(requests),
            successful_images=len(successful),
            failed_images=len(failed),
            total_processing_time_ms=total_time_ms,
            average_processing_time_ms=total_time_ms / len(requests) if requests else 0.0,
            results=results,
            summary=summary
        )
        
        logger.info(
            "Batch processing completed",
            batch_id=batch_id,
            total_images=len(requests),
            successful=len(successful),
            failed=len(failed),
            total_time_ms=total_time_ms,
            avg_time_ms=batch_result.average_processing_time_ms
        )
        
        return batch_result
    
    async def _process_single_image(
        self, 
        loop: asyncio.AbstractEventLoop,
        request: BatchRequest,
        ml_processor
    ) -> BatchResponse:
        """Process a single image within the batch"""
        start_time = time.time()
        
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(request.image_base64)
            
            # Validate image size
            if len(image_bytes) > settings.max_image_size:
                return BatchResponse(
                    image_id=request.image_id,
                    success=False,
                    error=f"Image size {len(image_bytes)} exceeds maximum {settings.max_image_size}"
                )
            
            # Preprocess image
            image = await loop.run_in_executor(
                self.executor, 
                self._preprocess_image, 
                image_bytes
            )
            
            if image is None:
                return BatchResponse(
                    image_id=request.image_id,
                    success=False,
                    error="Failed to decode or preprocess image"
                )
            
            # Process with ML model
            result = await loop.run_in_executor(
                self.executor,
                ml_processor.detect_fingers_in_image,
                image
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            if result["success"]:
                return BatchResponse(
                    image_id=request.image_id,
                    success=True,
                    finger_count=result["finger_count"],
                    confidence=result["confidence"],
                    detected_gesture=result["detected_gesture"],
                    processing_time_ms=processing_time_ms
                )
            else:
                return BatchResponse(
                    image_id=request.image_id,
                    success=False,
                    error=result.get("error", "Unknown error"),
                    processing_time_ms=processing_time_ms
                )
                
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "Batch image processing failed",
                image_id=request.image_id,
                error=str(e)
            )
            
            return BatchResponse(
                image_id=request.image_id,
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )
    
    def _preprocess_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Preprocess image for ML inference"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return None
            
            # Convert BGR to RGB (MediaPipe expects RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image_rgb
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
    
    def _create_batch_summary(
        self, 
        results: List[BatchResponse], 
        successful: List[BatchResponse], 
        failed: List[BatchResponse]
    ) -> Dict[str, Any]:
        """Create summary statistics for the batch"""
        
        # Gesture distribution
        gesture_counts = {}
        confidence_scores = []
        processing_times = []
        
        for result in successful:
            if result.detected_gesture:
                gesture_counts[result.detected_gesture] = gesture_counts.get(result.detected_gesture, 0) + 1
            
            if result.confidence is not None:
                confidence_scores.append(result.confidence)
            
            if result.processing_time_ms is not None:
                processing_times.append(result.processing_time_ms)
        
        # Error distribution
        error_counts = {}
        for result in failed:
            error_type = result.error or "Unknown"
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        summary = {
            "gesture_distribution": gesture_counts,
            "error_distribution": error_counts,
            "success_rate": len(successful) / len(results) if results else 0.0,
            "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
            "min_processing_time_ms": min(processing_times) if processing_times else 0,
            "max_processing_time_ms": max(processing_times) if processing_times else 0,
            "median_processing_time_ms": sorted(processing_times)[len(processing_times)//2] if processing_times else 0
        }
        
        return summary
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        logger.info("Batch processor cleaned up")


# Global batch processor instance
batch_processor = BatchProcessor()


def validate_batch_request(requests: List[BatchRequest]) -> None:
    """Validate batch request parameters"""
    if not requests:
        raise ValueError("Batch cannot be empty")
    
    if len(requests) > settings.batch_size:
        raise ValueError(f"Batch size {len(requests)} exceeds maximum {settings.batch_size}")
    
    # Validate individual requests
    for i, request in enumerate(requests):
        if not request.image_id:
            raise ValueError(f"Request {i}: image_id is required")
        
        if not request.image_base64:
            raise ValueError(f"Request {i}: image_base64 is required")
        
        # Validate base64 format
        try:
            base64.b64decode(request.image_base64)
        except Exception:
            raise ValueError(f"Request {i}: Invalid base64 image data")
