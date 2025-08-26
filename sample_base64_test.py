#!/usr/bin/env python3
"""
Sample Test Script for Base64 Dataset
Uses the converted base64 images to test your service
"""

import json
import requests
import random

def test_with_base64_dataset():
    """Test service with base64 encoded images"""
    # Load the base64 dataset
    with open('dataset_base64_20250826_182633.json', 'r') as f:
        dataset = json.load(f)
    
    print("🧪 Testing with Base64 Dataset")
    print(f"📊 Total images: {dataset['metadata']['total_images']}")
    
    # Test a few random images
    test_images = random.sample(dataset['images'], min(5, len(dataset['images'])))
    
    for i, image_data in enumerate(test_images, 1):
        print(f"\n{i}. Testing {image_data['gesture_label']} - {image_data['filename']}")
        
        # Test with your service (update URL as needed)
        try:
            response = requests.post(
                "http://localhost:8001/detect-fingers",
                json={"image_base64": image_data['base64']},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success: {result['detected_gesture']} ({result['confidence']:.3f})")
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    test_with_base64_dataset()
