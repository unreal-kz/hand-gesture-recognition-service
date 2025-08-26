#!/usr/bin/env python3
"""
Test Working Hand Gesture Service with Real Dataset Images
Uses images from hand_gesture_dataset folder to test the WORKING service
"""

import requests
import base64
import os
import json
from datetime import datetime

def test_working_service_with_dataset():
    """Test the WORKING hand gesture service using real dataset images"""
    # Use your LOCAL working service, not the broken Alpine container
    base_url = "http://localhost:8000"  # Your working local service
    dataset_dir = "hand_gesture_dataset"
    
    print("🎯 Testing WORKING Hand Gesture Service with Real Dataset")
    print("=" * 70)
    print("📍 Using LOCAL service on port 8000 (not Alpine container)")
    print("=" * 70)
    
    # Test 1: Health Check
    print("\n1️⃣ Testing Health Endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health Check: {health_data['status']}")
            if 'version' in health_data:
                print(f"   Version: {health_data['version']}")
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            print("   Make sure your local service is running with: python app.py")
            return
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        print("   Make sure your local service is running with: python app.py")
        return
    
    # Test 2: Find Dataset Images
    print("\n2️⃣ Scanning Dataset Directory...")
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    images_to_test = []
    gesture_mapping = {
        'close': 'Close',
        'one': 'One', 
        'two': 'Two',
        'three': 'Three',
        'four': 'Four',
        'open': 'Open'
    }
    
    for gesture_dir in os.listdir(dataset_dir):
        gesture_path = os.path.join(dataset_dir, gesture_dir)
        if os.path.isdir(gesture_path) and gesture_dir in gesture_mapping:
            for filename in os.listdir(gesture_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images_to_test.append({
                        'path': os.path.join(gesture_path, filename),
                        'true_gesture': gesture_dir,
                        'filename': filename
                    })
    
    if not images_to_test:
        print("❌ No images found in dataset")
        return
    
    print(f"✅ Found {len(images_to_test)} images to test")
    
    # Test 3: Test Each Image with REAL ML processing
    print(f"\n3️⃣ Testing {len(images_to_test)} Images with REAL ML...")
    successful_tests = 0
    results = []
    
    for i, image_info in enumerate(images_to_test, 1):
        print(f"\n[{i}/{len(images_to_test)}] Testing: {image_info['true_gesture']}/{image_info['filename']}")
        
        try:
            # Read and encode image
            with open(image_info['path'], 'rb') as image_file:
                image_data = image_file.read()
                image_base64 = base64.b64encode(image_data).decode()
            
            # Send to WORKING service
            response = requests.post(
                f"{base_url}/detect-fingers",
                json={"image_base64": image_base64},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ SUCCESS with REAL ML!")
                print(f"   🎯 True Gesture: {image_info['true_gesture'].title()}")
                print(f"   🔮 Predicted: {result['detected_gesture']}")
                print(f"   📊 Confidence: {result['confidence']}")
                print(f"   ⏱️  Time: {result['processing_time_ms']}ms")
                
                # Check if prediction is correct
                prediction_correct = result['detected_gesture'].lower() == image_info['true_gesture'].lower()
                
                if prediction_correct:
                    print(f"   🎉 CORRECT PREDICTION!")
                else:
                    print(f"   ❌ WRONG PREDICTION")
                
                # Store result
                test_result = {
                    'image': image_info['filename'],
                    'true_gesture': image_info['true_gesture'].title(),
                    'predicted_gesture': result['detected_gesture'],
                    'confidence': result['confidence'],
                    'processing_time_ms': result['processing_time_ms'],
                    'correct': prediction_correct,
                    'success': True,
                    'error': None
                }
                results.append(test_result)
                successful_tests += 1
                
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                results.append({
                    'image': image_info['filename'],
                    'true_gesture': image_info['true_gesture'].title(),
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}"
                })
                
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                'image': image_info['filename'],
                'true_gesture': image_info['true_gesture'].title(),
                'success': False,
                'error': str(e)
            })
    
    # Test 4: Performance Analysis
    print(f"\n4️⃣ Performance Analysis")
    print("=" * 70)
    print(f"Total Images: {len(images_to_test)}")
    print(f"Successful Tests: {successful_tests}")
    print(f"Failed Tests: {len(images_to_test) - successful_tests}")
    print(f"Success Rate: {(successful_tests/len(images_to_test)*100):.1f}%")
    
    if successful_tests > 0:
        # Calculate accuracy
        correct_predictions = sum(1 for r in results if r.get('correct', False))
        accuracy = correct_predictions / successful_tests if successful_tests > 0 else 0
        print(f"Prediction Accuracy: {(accuracy*100):.1f}%")
        
        # Average processing time
        avg_time = sum(r.get('processing_time_ms', 0) for r in results if r.get('success')) / successful_tests
        print(f"Average Processing Time: {avg_time:.1f}ms")
    
    # Save results
    results_file = f"real_service_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_data = {
        'test_date': datetime.now().isoformat(),
        'service_url': base_url,
        'service_type': 'local_working_service',
        'total_images': len(images_to_test),
        'successful_tests': successful_tests,
        'success_rate': successful_tests/len(images_to_test),
        'results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n💾 Test results saved to: {results_file}")
    
    print("\n" + "=" * 70)
    print("🎯 REAL ML Service Testing Complete!")
    print(f"📊 Tested {len(images_to_test)} real images with ACTUAL ML processing")
    print("💡 This is your WORKING service, not a mock!")

if __name__ == "__main__":
    test_working_service_with_dataset()
