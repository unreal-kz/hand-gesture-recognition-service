#!/usr/bin/env python3
"""
Finger Detection ML Service
Lightweight service for detecting finger counts in images
"""

import os
import time
import logging
import base64
from typing import Dict, Any
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Finger Detection Service",
    description="ML service for detecting finger counts in images",
    version="1.0.0"
)

# Global variables for ML models
mp_hands = None
keypoint_classifier = None
keypoint_labels = []

class FingerDetectionRequest(BaseModel):
    """Request model for finger detection"""
    image_base64: str

class FingerDetectionResponse(BaseModel):
    """Response model for finger detection"""
    finger_count: int
    confidence: float
    processing_time_ms: int
    detected_gesture: str
    success: bool

def load_ml_models():
    """Load ML models on startup"""
    global mp_hands, keypoint_classifier, keypoint_labels
    
    try:
        # Initialize MediaPipe hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # Load keypoint classifier
        from model.keypoint_classifier import KeyPointClassifier
        keypoint_classifier = KeyPointClassifier()
        
        # Load labels
        import csv
        label_path = os.path.join(os.path.dirname(__file__), 'model/keypoint_classifier/keypoint_classifier_label.csv')
        with open(label_path, encoding='utf-8-sig') as f:
            keypoint_labels = [row[0] for row in csv.reader(f)]
        
        logger.info(f"ML models loaded successfully. Labels: {keypoint_labels}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load ML models: {e}")
        return False

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image for ML inference"""
    try:
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

def detect_fingers_in_image(image: np.ndarray) -> Dict[str, Any]:
    """Detect fingers in the given image"""
    try:
        # Process image with MediaPipe
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
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
        preprocessed_landmarks = preprocess_landmarks(landmark_list)
        
        # Classify hand gesture
        gesture_id, confidence = keypoint_classifier(preprocessed_landmarks)
        detected_gesture = keypoint_labels[gesture_id] if gesture_id < len(keypoint_labels) else "Unknown"
        
        # Map gestures to finger counts
        finger_count = map_gesture_to_finger_count(detected_gesture)
        
        return {
            "success": True,
            "finger_count": finger_count,
            "confidence": float(confidence),
            "detected_gesture": detected_gesture,
            "gesture_id": gesture_id
        }
        
    except Exception as e:
        error_msg = str(e) if e else "Unknown error occurred"
        logger.error(f"Finger detection failed: {error_msg}")
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Exception args: {e.args if hasattr(e, 'args') else 'No args'}")
        return {
            "success": False,
            "error": error_msg
        }

def preprocess_landmarks(landmark_list: list) -> list:
    """Preprocess landmarks for classification (same as your existing code)"""
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

def map_gesture_to_finger_count(gesture: str) -> int:
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

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("Starting Finger Detection Service...")
    
    if not load_ml_models():
        logger.error("Failed to initialize ML models. Service may not work properly.")
    else:
        logger.info("Service initialized successfully!")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "finger-detection-service",
        "version": "1.0.0",
        "models_loaded": keypoint_classifier is not None
    }

@app.post("/detect-fingers", response_model=FingerDetectionResponse)
async def detect_fingers(request: FingerDetectionRequest):
    """Detect finger count in base64 encoded image"""
    start_time = time.time()
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_base64)
        
        # Preprocess image
        image = preprocess_image(image_bytes)
        
        # Detect fingers
        result = detect_fingers_in_image(image)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return FingerDetectionResponse(
            finger_count=result["finger_count"],
            confidence=result["confidence"],
            processing_time_ms=processing_time_ms,
            detected_gesture=result["detected_gesture"],
            success=True
        )
        
    except Exception as e:
        error_msg = str(e) if e else "Unknown error occurred"
        logger.error(f"Request processing failed: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_msg}")

@app.post("/detect-fingers-upload")
async def detect_fingers_upload(file: UploadFile = File(...)):
    """Detect finger count in uploaded image file"""
    start_time = time.time()
    
    try:
        # Read uploaded file
        image_bytes = await file.read()
        
        # Preprocess image
        image = preprocess_image(image_bytes)
        
        # Detect fingers
        result = detect_fingers_in_image(image)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            "finger_count": result["finger_count"],
            "confidence": result["confidence"],
            "processing_time_ms": processing_time_ms,
            "detected_gesture": result["detected_gesture"],
            "success": True
        }
        
    except Exception as e:
        error_msg = str(e) if e else "Unknown error occurred"
        logger.error(f"Request processing failed: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_msg}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
