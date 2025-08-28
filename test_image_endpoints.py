#!/usr/bin/env python3
"""
Test script for the new image file endpoints
Demonstrates how to use the updated /detect-fingers endpoint with image files
"""

import requests
import base64
from pathlib import Path
import json

# Configuration
BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "test_image.jpg"  # Replace with your test image path

def test_single_image_file():
    """Test single image file upload to /detect-fingers"""
    print("🧪 Testing single image file upload to /detect-fingers")
    
    if not Path(TEST_IMAGE_PATH).exists():
        print(f"❌ Test image not found: {TEST_IMAGE_PATH}")
        print("   Please place a test image in the current directory")
        return
    
    try:
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': (TEST_IMAGE_PATH, f, 'image/jpeg')}
            
            response = requests.post(f"{BASE_URL}/detect-fingers", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Success!")
                print(f"   🎯 Detected Gesture: {result['detected_gesture']}")
                print(f"   👆 Finger Count: {result['finger_count']}")
                print(f"   📊 Confidence: {result['confidence']:.3f}")
                print(f"   ⏱️  Processing Time: {result['processing_time_ms']}ms")
                print(f"   🆔 Request ID: {result['request_id']}")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_single_image_base64():
    """Test single base64 image to /detect-fingers"""
    print("\n🧪 Testing base64 image to /detect-fingers")
    
    if not Path(TEST_IMAGE_PATH).exists():
        print(f"❌ Test image not found: {TEST_IMAGE_PATH}")
        return
    
    try:
        with open(TEST_IMAGE_PATH, 'rb') as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            payload = {"image_base64": image_base64}
            
            response = requests.post(f"{BASE_URL}/detect-fingers", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Success!")
                print(f"   🎯 Detected Gesture: {result['detected_gesture']}")
                print(f"   👆 Finger Count: {result['finger_count']}")
                print(f"   📊 Confidence: {result['confidence']:.3f}")
                print(f"   ⏱️  Processing Time: {result['processing_time_ms']}ms")
                print(f"   🆔 Request ID: {result['request_id']}")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_batch_image_files():
    """Test batch image file upload to /detect-fingers-batch-files"""
    print("\n🧪 Testing batch image file upload to /detect-fingers-batch-files")
    
    if not Path(TEST_IMAGE_PATH).exists():
        print(f"❌ Test image not found: {TEST_IMAGE_PATH}")
        return
    
    try:
        # Create multiple files for batch testing
        files = []
        for i in range(3):  # Test with 3 images
            with open(TEST_IMAGE_PATH, 'rb') as f:
                files.append(('files', (f'test_image_{i}.jpg', f, 'image/jpeg')))
        
        response = requests.post(f"{BASE_URL}/detect-fingers-batch-files", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"   📦 Batch ID: {result['batch_id']}")
            print(f"   📁 Total Files: {result['total_files']}")
            print(f"   ✅ Successful: {result['successful_files']}")
            print(f"   ❌ Failed: {result['failed_files']}")
            print(f"   ⏱️  Total Time: {result['total_processing_time_ms']}ms")
            print(f"   📊 Success Rate: {result['summary']['success_rate']:.1%}")
            
            if result['summary']['gesture_distribution']:
                print("   🎯 Gesture Distribution:")
                for gesture, count in result['summary']['gesture_distribution'].items():
                    print(f"      {gesture}: {count}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_health_endpoint():
    """Test the health endpoint"""
    print("\n🏥 Testing health endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Service is healthy!")
            print(f"   📊 Status: {result['status']}")
            print(f"   🔧 Models Loaded: {result['models_loaded']}")
            print(f"   ⏱️  Uptime: {result['uptime_seconds']:.1f}s")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")

def main():
    """Run all tests"""
    print("🚀 Hand Gesture Recognition Service - Image Endpoint Tests")
    print("=" * 60)
    
    # Test health first
    test_health_endpoint()
    
    # Test image endpoints
    test_single_image_file()
    test_single_image_base64()
    test_batch_image_files()
    
    print("\n" + "=" * 60)
    print("✨ All tests completed!")
    print("\n💡 Usage Examples:")
    print("   Single image file: POST /detect-fingers with 'file' parameter")
    print("   Single base64: POST /detect-fingers with JSON body")
    print("   Batch files: POST /detect-fingers-batch-files with multiple files")
    print("   Batch base64: POST /detect-fingers-batch with JSON body")

if __name__ == "__main__":
    main()
