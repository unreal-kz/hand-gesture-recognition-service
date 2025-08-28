#!/usr/bin/env python3
"""
Structured logging configuration for the Hand Gesture Recognition Service
"""

import sys
import logging
from typing import Any, Dict
import structlog
from structlog.stdlib import LoggerFactory
from structlog.processors import JSONRenderer, TimeStamper, add_log_level
from structlog.typing import FilteringBoundLogger

from config import settings


def setup_logging() -> None:
    """Setup structured logging configuration"""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            JSONRenderer() if settings.log_format == "json" else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)


def get_logger(name: str) -> FilteringBoundLogger:
    """Get a structured logger instance"""
    return structlog.get_logger(name)


# Create default logger
logger = get_logger(__name__)


class RequestLogger:
    """Request-specific logging context"""
    
    def __init__(self, request_id: str = None):
        self.request_id = request_id or "unknown"
        self.logger = get_logger("request")
    
    def log_request(self, method: str, path: str, **kwargs) -> None:
        """Log incoming request"""
        self.logger.info(
            "Incoming request",
            request_id=self.request_id,
            method=method,
            path=path,
            **kwargs
        )
    
    def log_response(self, status_code: int, processing_time_ms: int, **kwargs) -> None:
        """Log response details"""
        self.logger.info(
            "Request completed",
            request_id=self.request_id,
            status_code=status_code,
            processing_time_ms=processing_time_ms,
            **kwargs
        )
    
    def log_error(self, error: Exception, **kwargs) -> None:
        """Log error details"""
        self.logger.error(
            "Request failed",
            request_id=self.request_id,
            error_type=type(error).__name__,
            error_message=str(error),
            **kwargs
        )


def log_function_call(func_name: str, **kwargs) -> None:
    """Log function call with parameters"""
    logger.info(
        "Function called",
        function=func_name,
        parameters=kwargs
    )


def log_ml_inference(gesture: str, confidence: float, processing_time_ms: int, **kwargs) -> None:
    """Log ML inference results"""
    logger.info(
        "ML inference completed",
        detected_gesture=gesture,
        confidence=confidence,
        processing_time_ms=processing_time_ms,
        **kwargs
    )


def log_performance_metric(metric_name: str, value: float, unit: str = None, **kwargs) -> None:
    """Log performance metrics"""
    logger.info(
        "Performance metric",
        metric=metric_name,
        value=value,
        unit=unit,
        **kwargs
    )
