#!/usr/bin/env python3
"""
Configuration management for the Hand Gesture Recognition Service
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Service Configuration
    app_name: str = Field(default="Hand Gesture Recognition Service", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")
    
    # ML Model Configuration
    model_path: str = Field(default="model/keypoint_classifier/keypoint_classifier.tflite", env="MODEL_PATH")
    min_detection_confidence: float = Field(default=0.5, env="MIN_DETECTION_CONFIDENCE")
    min_tracking_confidence: float = Field(default=0.5, env="MIN_TRACKING_CONFIDENCE")
    max_num_hands: int = Field(default=1, env="MAX_NUM_HANDS")
    
    # Performance Configuration
    batch_size: int = Field(default=10, env="BATCH_SIZE")
    max_image_size: int = Field(default=10 * 1024 * 1024, env="MAX_IMAGE_SIZE")  # 10MB
    processing_timeout: int = Field(default=30, env="PROCESSING_TIMEOUT")
    
    # Monitoring Configuration
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    metrics_port: int = Field(default=8001, env="METRICS_PORT")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Redis Configuration (for caching and rate limiting)
    redis_enabled: bool = Field(default=False, env="REDIS_ENABLED")
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Celery Configuration (for async processing)
    celery_enabled: bool = Field(default=False, env="CELERY_ENABLED")
    celery_broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
