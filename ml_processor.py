#!/usr/bin/env python3
"""
Enhanced ML processor for hand gesture recognition with monitoring and caching
"""

import os
import time
import hashlib
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
import mediapipe as mp
import redis.asyncio as redis

from config import settings
from logger import get_logger, log_function_call, log_ml_inference
from monitoring import metrics
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

logger = get_logger(__name__)


class MLProcessor:
    """Enhanced ML processor with monitoring, caching, and error handling"""
    
    def __init__(self):
        self.mp_hands = None
        self.keypoint_classifier = None
        self.keypoint_labels = []
        self.redis_client: Optional[redis.Redis] = None
        self.model_loaded = False
        
        # Initialize components
        self._init_mediapipe()
        self._init_classifier()
        self._init_redis()
        
        # Update metrics
        metrics.set_model_loaded(self.model_loaded)
    
    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe hands detection"""
        try:
            self.mp_hands = mp.solutions.hands
            logger.info("MediaPipe hands initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            raise
    
    def _init_classifier(self) -> None:
        """Initialize the keypoint classifier"""
        try:
            # Load keypoint classifier
            self.keypoint_classifier = KeyPointClassifier()
            
            # Load labels
            label_path = os.path.join(
                os.path.dirname(__file__), 
                'model/keypoint_classifier/keypoint_classifier_label.csv'
            )
            
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found: {label_path}")
            
            import csv
            with open(label_path, encoding='utf-8-sig') as f:
                self.keypoint_labels = [row[0] for row in csv.reader(f)]
            
            self.model_loaded = True
            logger.info(f"Keypoint classifier loaded successfully. Labels: {self.keypoint_labels}")
            
        except Exception as e:
            logger.error(f"Failed to load keypoint classifier: {e}")
            self.model_loaded = False
            raise
    
    def _init_redis(self) -> None:
        """Initialize Redis connection for caching"""
        if not settings.redis_enabled:
            return
        
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}, continuing without caching")
            self.redis_client = None
    
    def _get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for image"""
        return f"gesture_cache:{image_hash}"
    
    def _hash_image(self, image: np.ndarray) -> str:
        """Generate hash for image for caching"""
        # Resize to small size for faster hashing
        small_image = cv2.resize(image, (64, 64))
        # Convert to grayscale and flatten
        gray = cv2.cvtColor(small_image, cv2.COLOR_RGB2GRAY)
        flat = gray.flatten()
        # Generate hash
        return hashlib.md5(flat.tobytes()).hexdigest()
    
    async def _get_cached_result(self, image_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._get_cache_key(image_hash)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                import json
                result = json.loads(cached_data)
                logger.debug("Cache hit", image_hash=image_hash[:8])
                return result
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def _set_cached_result(self, image_hash: str, result: Dict[str, Any], ttl: int = 3600) -> None:
        """Cache result for future use"""
        if not self.redis_client:
            return
        
        try:
            cache_key = self._get_cache_key(image_hash)
            import json
            cached_data = json.dumps(result)
            await self.redis_client.setex(cache_key, ttl, cached_data)
            logger.debug("Result cached", image_hash=image_hash[:8], ttl=ttl)
            
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def detect_fingers_in_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect fingers in the given image with enhanced error handling"""
        start_time = time.time()
        
        try:
            log_function_call("detect_fingers_in_image", image_shape=image.shape)
            
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid image input")
            
            # Process image with MediaPipe
            hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=settings.max_num_hands,
                min_detection_confidence=settings.min_detection_confidence,
                min_tracking_confidence=settings.min_tracking_confidence,
            )
            
            results = hands.process(image)
            
            if not results.multi_hand_landmarks:
                return {
                    "success": False,
                    "error": "No hands detected in image"
                }
            
            # Get the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Calculate landmark list
            image_height, image_width = image.shape[:2]
            landmark_list = []
            
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                landmark_list.append([x, y])
            
            # Preprocess landmarks for classification
            preprocessed_landmarks = self._preprocess_landmarks(landmark_list)
            
            # Classify hand gesture
            gesture_id, confidence = self.keypoint_classifier(preprocessed_landmarks)
            detected_gesture = self.keypoint_labels[gesture_id] if gesture_id < len(self.keypoint_labels) else "Unknown"
            
            # Map gestures to finger counts
            finger_count = self._map_gesture_to_finger_count(detected_gesture)
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            result = {
                "success": True,
                "finger_count": finger_count,
                "confidence": float(confidence),
                "detected_gesture": detected_gesture,
                "gesture_id": gesture_id,
                "processing_time_ms": processing_time_ms
            }
            
            # Log ML inference
            log_ml_inference(detected_gesture, confidence, processing_time_ms)
            
            # Record metrics
            metrics.record_inference(detected_gesture, confidence, processing_time_ms / 1000.0)
            
            return result
            
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e) if e else "Unknown error occurred"
            
            logger.error(
                "Finger detection failed",
                error=error_msg,
                error_type=type(e).__name__,
                processing_time_ms=processing_time_ms
            )
            
            # Record error metrics
            metrics.record_error(type(e).__name__, "detect_fingers")
            
            return {
                "success": False,
                "error": error_msg,
                "processing_time_ms": processing_time_ms
            }
    
    def _preprocess_landmarks(self, landmark_list: list) -> list:
        """Preprocess landmarks for classification"""
        import copy
        import itertools
        
        temp_landmark_list = copy.deepcopy(landmark_list)
        
        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        
        # Convert to one-dimensional list
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        
        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))
        if max_value > 0:
            temp_landmark_list = [n / max_value for n in temp_landmark_list]
        
        return temp_landmark_list
    
    def _map_gesture_to_finger_count(self, gesture: str) -> int:
        """Map detected gesture to finger count"""
        gesture_mapping = {
            "Open": 5,
            "Close": 0,
            "One": 1,
            "Two": 2,
            "Three": 3,
            "Four": 4
        }
        
        return gesture_mapping.get(gesture, 0)
    
    async def process_image_with_caching(self, image: np.ndarray) -> Dict[str, Any]:
        """Process image with caching support"""
        # Generate image hash
        image_hash = self._hash_image(image)
        
        # Try to get cached result
        cached_result = await self._get_cached_result(image_hash)
        if cached_result:
            return cached_result
        
        # Process image
        result = self.detect_fingers_in_image(image)
        
        # Cache successful results
        if result["success"]:
            await self._set_cached_result(image_hash, result)
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models"""
        return {
            "model_loaded": self.model_loaded,
            "mediapipe_initialized": self.mp_hands is not None,
            "classifier_loaded": self.keypoint_classifier is not None,
            "available_labels": self.keypoint_labels,
            "redis_cache_enabled": self.redis_client is not None,
            "detection_confidence": settings.min_detection_confidence,
            "tracking_confidence": settings.min_tracking_confidence,
            "max_hands": settings.max_num_hands
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("ML processor Redis connection closed")


# Global ML processor instance
ml_processor = MLProcessor()
