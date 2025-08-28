#!/usr/bin/env python3
"""
Monitoring and metrics for the Hand Gesture Recognition Service
"""

import time
from typing import Dict, Any
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, multiprocess
)
from prometheus_client.exposition import start_http_server
import threading

from config import settings
from logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Prometheus metrics collector for the service"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Request metrics
        self.request_counter = Counter(
            'hand_gesture_requests_total',
            'Total number of hand gesture detection requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'hand_gesture_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # ML inference metrics
        self.inference_counter = Counter(
            'ml_inference_total',
            'Total number of ML inference operations',
            ['gesture', 'confidence_level'],
            registry=self.registry
        )
        
        self.inference_duration = Histogram(
            'ml_inference_duration_seconds',
            'ML inference duration in seconds',
            ['gesture'],
            registry=self.registry
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            'model_accuracy',
            'Current model accuracy percentage',
            ['model_name'],
            registry=self.registry
        )
        
        self.model_confidence = Summary(
            'model_confidence',
            'Model confidence distribution',
            ['gesture'],
            registry=self.registry
        )
        
        # System metrics
        self.active_requests = Gauge(
            'active_requests',
            'Number of currently active requests',
            registry=self.registry
        )
        
        self.model_loaded = Gauge(
            'model_loaded',
            'Whether the ML model is loaded (1) or not (0)',
            registry=self.registry
        )
        
        # Error metrics
        self.error_counter = Counter(
            'hand_gesture_errors_total',
            'Total number of errors',
            ['error_type', 'endpoint'],
            registry=self.registry
        )
        
        # Batch processing metrics
        self.batch_size = Histogram(
            'batch_processing_size',
            'Batch size distribution',
            registry=self.registry
        )
        
        self.batch_duration = Histogram(
            'batch_processing_duration_seconds',
            'Batch processing duration in seconds',
            registry=self.registry
        )
    
    def record_request(self, method: str, endpoint: str, status: str, duration: float) -> None:
        """Record request metrics"""
        self.request_counter.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_inference(self, gesture: str, confidence: float, duration: float) -> None:
        """Record ML inference metrics"""
        confidence_level = self._get_confidence_level(confidence)
        self.inference_counter.labels(gesture=gesture, confidence_level=confidence_level).inc()
        self.inference_duration.labels(gesture=gesture).observe(duration)
        self.model_confidence.labels(gesture=gesture).observe(confidence)
    
    def record_error(self, error_type: str, endpoint: str) -> None:
        """Record error metrics"""
        self.error_counter.labels(error_type=error_type, endpoint=endpoint).inc()
    
    def set_model_loaded(self, loaded: bool) -> None:
        """Set model loaded status"""
        self.model_loaded.set(1 if loaded else 0)
    
    def set_model_accuracy(self, accuracy: float) -> None:
        """Set model accuracy"""
        self.model_accuracy.labels(model_name="keypoint_classifier").set(accuracy)
    
    def record_batch_processing(self, batch_size: int, duration: float) -> None:
        """Record batch processing metrics"""
        self.batch_size.observe(batch_size)
        self.batch_duration.observe(duration)
    
    def increment_active_requests(self) -> None:
        """Increment active requests counter"""
        self.active_requests.inc()
    
    def decrement_active_requests(self) -> None:
        """Decrement active requests counter"""
        self.active_requests.dec()
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence to categorical level"""
        if confidence >= 0.9:
            return "high"
        elif confidence >= 0.7:
            return "medium"
        elif confidence >= 0.5:
            return "low"
        else:
            return "very_low"
    
    def get_metrics(self) -> str:
        """Get current metrics in Prometheus format"""
        return generate_latest(self.registry)
    
    def start_metrics_server(self) -> None:
        """Start the metrics HTTP server"""
        if settings.metrics_enabled:
            try:
                start_http_server(settings.metrics_port, registry=self.registry)
                logger.info(
                    "Metrics server started",
                    port=settings.metrics_port,
                    endpoint=f"http://localhost:{settings.metrics_port}/metrics"
                )
            except Exception as e:
                logger.error("Failed to start metrics server", error=str(e))


# Global metrics instance
metrics = MetricsCollector()


class MetricsMiddleware:
    """FastAPI middleware for collecting request metrics"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            method = scope["method"]
            path = scope["path"]
            
            # Increment active requests
            metrics.increment_active_requests()
            
            # Create a custom send function to capture response status
            async def send_with_metrics(message):
                if message["type"] == "http.response.start":
                    status = message["status"]
                    duration = time.time() - start_time
                    
                    # Record metrics
                    metrics.record_request(
                        method=method,
                        endpoint=path,
                        status=str(status),
                        duration=duration
                    )
                    
                    # Decrement active requests
                    metrics.decrement_active_requests()
                
                await send(message)
            
            await self.app(scope, receive, send_with_metrics)
        else:
            await self.app(scope, receive, send)


def get_health_metrics() -> Dict[str, Any]:
    """Get health check metrics"""
    return {
        "model_loaded": metrics.model_loaded._value.get(),
        "active_requests": metrics.active_requests._value.get(),
        "total_requests": metrics.request_counter._value.sum(),
        "total_errors": metrics.error_counter._value.sum()
    }
