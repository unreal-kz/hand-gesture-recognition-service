#!/usr/bin/env python3
"""
Hand Gesture Dataset Capture Script
Uses webcam to capture hand gesture images for testing the Finger Detection Service
"""

import cv2
import os
import time
import json
from datetime import datetime

class HandGestureCapture:
    def __init__(self):
        self.gestures = {
            '0': 'Close',      # Closed fist
            '1': 'One',        # One finger
            '2': 'Two',        # Two fingers
            '3': 'Three',      # Three fingers
            '4': 'Four',       # Four fingers
            '5': 'Open'        # Open hand
        }
        
        self.dataset_dir = "hand_gesture_dataset"
        self.current_gesture = None
        self.capture_count = 0
        self.cap = None
        
        # Create dataset directory structure
        self.setup_dataset_structure()
        
    def setup_dataset_structure(self):
        """Create the dataset directory structure"""
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Create subdirectories for each gesture
        for gesture in self.gestures.values():
            gesture_dir = os.path.join(self.dataset_dir, gesture.lower())
            os.makedirs(gesture_dir, exist_ok=True)
            
        # Create metadata file
        self.metadata_file = os.path.join(self.dataset_dir, "dataset_info.json")
        self.load_metadata()
        
    def load_metadata(self):
        """Load or create dataset metadata"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "created_date": datetime.now().isoformat(),
                "total_images": 0,
                "gestures": {gesture: 0 for gesture in self.gestures.values()},
                "capture_settings": {
                    "image_format": "jpg",
                    "image_quality": 95,
                    "resolution": "640x480"
                }
            }
            self.save_metadata()
            
    def save_metadata(self):
        """Save dataset metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def start_camera(self):
        """Initialize webcam"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Error: Could not open webcam")
            return False
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Webcam initialized successfully")
        return True
        
    def show_gesture_menu(self):
        """Display gesture selection menu"""
        print("\n" + "="*60)
        print("🤚 HAND GESTURE CAPTURE MENU")
        print("="*60)
        print("Select gesture to capture:")
        for key, gesture in self.gestures.items():
            count = self.metadata["gestures"][gesture]
            print(f"  {key}: {gesture} ({count} images)")
        print("  q: Quit")
        print("  h: Help")
        print("="*60)
        
    def capture_image(self, gesture):
        """Capture and save an image for the specified gesture"""
        if not self.cap or not self.cap.isOpened():
            print("❌ Camera not available")
            return False
            
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{gesture.lower()}_{timestamp}.jpg"
        filepath = os.path.join(self.dataset_dir, gesture.lower(), filename)
        
        # Capture frame
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            return False
            
        # Save image
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Update metadata
        self.metadata["gestures"][gesture] += 1
        self.metadata["total_images"] += 1
        self.save_metadata()
        
        print(f"✅ Captured: {filename}")
        return True
        
    def show_live_preview(self):
        """Show live camera preview with gesture info"""
        if not self.cap or not self.cap.isOpened():
            return
            
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Add text overlay
            if self.current_gesture:
                cv2.putText(frame, f"Gesture: {self.current_gesture}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Press SPACE to capture", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Select gesture from menu", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
            cv2.putText(frame, "Press ESC to return to menu", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Hand Gesture Capture', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 32 and self.current_gesture:  # SPACE
                if self.capture_image(self.current_gesture):
                    self.capture_count += 1
                    
        cv2.destroyAllWindows()
        
    def run(self):
        """Main capture loop"""
        if not self.start_camera():
            return
            
        print("🚀 Hand Gesture Dataset Capture Tool")
        print("📸 Use this tool to build your test dataset")
        
        while True:
            self.show_gesture_menu()
            
            choice = input("\n🎯 Enter your choice: ").strip().lower()
            
            if choice == 'q':
                print("👋 Goodbye!")
                break
            elif choice == 'h':
                self.show_help()
            elif choice in self.gestures:
                gesture = self.gestures[choice]
                self.current_gesture = gesture
                print(f"\n📸 Capturing {gesture} gesture...")
                print("💡 Position your hand and press SPACE to capture")
                print("   Press ESC to return to menu")
                
                self.show_live_preview()
                self.current_gesture = None
            else:
                print("❌ Invalid choice. Please try again.")
                
        # Cleanup
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Final summary
        self.show_summary()
        
    def show_help(self):
        """Display help information"""
        print("\n" + "="*60)
        print("📖 HELP - Hand Gesture Capture Tool")
        print("="*60)
        print("This tool helps you build a dataset of hand gesture images.")
        print("\n📋 How to use:")
        print("1. Select a gesture type from the menu")
        print("2. Position your hand in front of the camera")
        print("3. Press SPACE to capture the image")
        print("4. Press ESC to return to the menu")
        print("5. Repeat for different gestures and angles")
        print("\n💡 Tips for better results:")
        print("- Ensure good lighting")
        print("- Keep your hand clearly visible")
        print("- Try different angles and positions")
        print("- Capture multiple images per gesture")
        print("- Use consistent background if possible")
        print("\n📁 Images are saved in organized folders:")
        print(f"   {self.dataset_dir}/")
        print("   ├── close/")
        print("   ├── one/")
        print("   ├── two/")
        print("   ├── three/")
        print("   ├── four/")
        print("   └── open/")
        print("\n" + "="*60)
        
    def show_summary(self):
        """Display dataset summary"""
        print("\n" + "="*60)
        print("📊 DATASET SUMMARY")
        print("="*60)
        print(f"📁 Dataset location: {os.path.abspath(self.dataset_dir)}")
        print(f"📸 Total images: {self.metadata['total_images']}")
        print("\n📋 Images per gesture:")
        for gesture, count in self.metadata["gestures"].items():
            print(f"   {gesture}: {count}")
        print("\n🎯 Next steps:")
        print("1. Test your service with the captured images")
        print("2. Use the test script: python test_real_image.py <image_path>")
        print("3. Or use the Swagger UI: http://localhost:8000/docs")
        print("\n" + "="*60)

def main():
    """Main function"""
    try:
        capture_tool = HandGestureCapture()
        capture_tool.run()
    except KeyboardInterrupt:
        print("\n\n👋 Capture interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure your webcam is available and not in use by another application")

if __name__ == "__main__":
    main()
