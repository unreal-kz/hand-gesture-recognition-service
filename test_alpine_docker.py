#!/usr/bin/env python3
"""
Test Script for Alpine Docker Container
Tests the running Alpine-compatible hand gesture recognition service
"""

import requests
import base64
from PIL import Image, ImageDraw
import io

def create_test_image():
    """Create a simple test image"""
    # Create a simple 200x200 test image
    img = Image.new('RGB', (200, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple shape
    draw.rectangle([50, 50, 150, 150], fill='lightblue', outline='black')
    draw.text((80, 100), "TEST", fill='black')
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()

def test_alpine_container():
    """Test the Alpine Docker container"""
    base_url = "http://localhost:8001"
    
    print("🐳 Testing Alpine Docker Container")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1️⃣ Testing Health Endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health Check: {health_data['status']}")
            print(f"   Version: {health_data['version']}")
            print(f"   Alpine Compatible: {health_data['alpine_compatible']}")
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
    
    # Test 2: Root Endpoint
    print("\n2️⃣ Testing Root Endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            root_data = response.json()
            print(f"✅ Root Endpoint: {root_data['message']}")
            print(f"   Status: {root_data['status']}")
        else:
            print(f"❌ Root Endpoint Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root Endpoint Error: {e}")
    
    # Test 3: API Documentation
    print("\n3️⃣ Testing API Documentation...")
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            docs_data = response.json()
            print(f"✅ API Documentation: {docs_data['message']}")
            print(f"   Available Endpoints: {len(docs_data['endpoints'])}")
        else:
            print(f"❌ API Documentation Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API Documentation Error: {e}")
    
    # Test 4: Image Processing (Base64)
    print("\n4️⃣ Testing Image Processing (Base64)...")
    try:
        # Create test image
        test_image = create_test_image()
        image_base64 = base64.b64encode(test_image).decode()
        
        # Send request
        response = requests.post(
            f"{base_url}/detect-fingers",
            json={"image_base64": image_base64},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Image Processing Success!")
            print(f"   Detected Gesture: {result['detected_gesture']}")
            print(f"   Confidence: {result['confidence']}")
            print(f"   Processing Time: {result['processing_time_ms']}ms")
            print(f"   Note: {result['note']}")
        else:
            print(f"❌ Image Processing Failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Image Processing Error: {e}")
    
    # Test 5: Container Info
    print("\n5️⃣ Container Information...")
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=finger-detection-service"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("✅ Container Status:")
            print(result.stdout)
        else:
            print("❌ Could not get container status")
    except Exception as e:
        print(f"❌ Container Info Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Alpine Docker Container Testing Complete!")

if __name__ == "__main__":
    test_alpine_container()
