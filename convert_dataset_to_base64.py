#!/usr/bin/env python3
"""
Convert Dataset Images to Base64
Converts all images from hand_gesture_dataset folder to base64 encoding
"""

import os
import base64
import json
from datetime import datetime
from PIL import Image
import io

def convert_image_to_base64(image_path):
    """Convert a single image to base64 encoding"""
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            base64_string = base64.b64encode(image_data).decode('utf-8')
            return base64_string
    except Exception as e:
        print(f"❌ Error converting {image_path}: {e}")
        return None

def get_image_info(image_path):
    """Get basic information about an image"""
    try:
        with Image.open(image_path) as img:
            return {
                'width': img.width,
                'height': img.height,
                'mode': img.mode,
                'format': img.format,
                'size_bytes': os.path.getsize(image_path)
            }
    except Exception as e:
        print(f"❌ Error getting info for {image_path}: {e}")
        return None

def convert_dataset_to_base64():
    """Convert entire dataset to base64 encoding"""
    dataset_dir = "hand_gesture_dataset"
    output_file = f"dataset_base64_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    print("🔄 Converting Dataset Images to Base64")
    print("=" * 60)
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    # Define gesture mapping
    gesture_mapping = {
        'close': 'Close',
        'one': 'One', 
        'two': 'Two',
        'three': 'Three',
        'four': 'Four',
        'open': 'Open'
    }
    
    # Collect all images
    all_images = []
    total_size = 0
    
    print("📁 Scanning dataset directory...")
    
    for gesture_dir in os.listdir(dataset_dir):
        gesture_path = os.path.join(dataset_dir, gesture_dir)
        if os.path.isdir(gesture_path) and gesture_dir in gesture_mapping:
            print(f"  📂 Processing {gesture_dir}...")
            
            for filename in os.listdir(gesture_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(gesture_path, filename)
                    
                    # Get image info
                    image_info = get_image_info(image_path)
                    if image_info:
                        # Convert to base64
                        base64_string = convert_image_to_base64(image_path)
                        if base64_string:
                            image_data = {
                                'gesture': gesture_dir,
                                'gesture_label': gesture_mapping[gesture_dir],
                                'filename': filename,
                                'filepath': image_path,
                                'base64': base64_string,
                                'image_info': image_info,
                                'conversion_timestamp': datetime.now().isoformat()
                            }
                            all_images.append(image_data)
                            total_size += image_info['size_bytes']
                            
                            print(f"    ✅ {filename} ({image_info['width']}x{image_info['height']})")
    
    if not all_images:
        print("❌ No images found to convert")
        return
    
    print(f"\n📊 Conversion Summary:")
    print(f"  Total Images: {len(all_images)}")
    print(f"  Total Size: {total_size / (1024*1024):.2f} MB")
    print(f"  Base64 Size: {len(json.dumps(all_images)) / (1024*1024):.2f} MB")
    
    # Group by gesture
    gesture_counts = {}
    for image_data in all_images:
        gesture = image_data['gesture']
        gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
    
    print(f"\n📋 Images per Gesture:")
    for gesture, count in gesture_counts.items():
        print(f"  {gesture.title()}: {count} images")
    
    # Save to JSON file
    print(f"\n💾 Saving to {output_file}...")
    
    output_data = {
        'metadata': {
            'conversion_date': datetime.now().isoformat(),
            'total_images': len(all_images),
            'total_size_bytes': total_size,
            'gesture_mapping': gesture_mapping,
            'gesture_counts': gesture_counts
        },
        'images': all_images
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"✅ Successfully saved {len(all_images)} images to {output_file}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return
    
    # Create a sample test file
    sample_file = "sample_base64_test.py"
    print(f"\n🔧 Creating sample test file: {sample_file}")
    
    sample_code = f'''#!/usr/bin/env python3
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
    with open('{output_file}', 'r') as f:
        dataset = json.load(f)
    
    print("🧪 Testing with Base64 Dataset")
    print(f"📊 Total images: {{dataset['metadata']['total_images']}}")
    
    # Test a few random images
    test_images = random.sample(dataset['images'], min(5, len(dataset['images'])))
    
    for i, image_data in enumerate(test_images, 1):
        print(f"\\n{{i}}. Testing {{image_data['gesture_label']}} - {{image_data['filename']}}")
        
        # Test with your service (update URL as needed)
        try:
            response = requests.post(
                "http://localhost:8001/detect-fingers",
                json={{"image_base64": image_data['base64']}},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success: {{result['detected_gesture']}} ({{result['confidence']:.3f}})")
            else:
                print(f"   ❌ Error: {{response.status_code}}")
                
        except Exception as e:
            print(f"   ❌ Failed: {{e}}")

if __name__ == "__main__":
    test_with_base64_dataset()
'''
    
    try:
        with open(sample_file, 'w') as f:
            f.write(sample_code)
        print(f"✅ Sample test file created: {sample_file}")
    except Exception as e:
        print(f"❌ Error creating sample file: {e}")
    
    print(f"\n🎯 Conversion Complete!")
    print(f"📁 Output file: {output_file}")
    print(f"🔧 Sample test: {sample_file}")
    print(f"💡 You can now use these base64 images to test your service!")

def create_compact_version():
    """Create a compact version with just essential data"""
    dataset_dir = "hand_gesture_dataset"
    compact_file = f"dataset_compact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    print(f"\n📦 Creating compact version...")
    
    compact_images = []
    
    for gesture_dir in os.listdir(dataset_dir):
        gesture_path = os.path.join(dataset_dir, gesture_dir)
        if os.path.isdir(gesture_path):
            for filename in os.listdir(gesture_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(gesture_path, filename)
                    
                    base64_string = convert_image_to_base64(image_path)
                    if base64_string:
                        compact_images.append({
                            'g': gesture_dir,  # gesture
                            'f': filename,     # filename
                            'b': base64_string # base64
                        })
    
    compact_data = {
        'm': {  # metadata
            'd': datetime.now().isoformat(),  # date
            'c': len(compact_images)         # count
        },
        'i': compact_images  # images
    }
    
    try:
        with open(compact_file, 'w') as f:
            json.dump(compact_data, f, separators=(',', ':'))
        print(f"✅ Compact version saved: {compact_file}")
        print(f"   Size: {os.path.getsize(compact_file) / 1024:.1f} KB")
    except Exception as e:
        print(f"❌ Error saving compact file: {e}")

if __name__ == "__main__":
    convert_dataset_to_base64()
    create_compact_version()
