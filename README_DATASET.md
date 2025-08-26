# 🤚 Hand Gesture Dataset Tools

This directory contains tools to build and test your own hand gesture dataset for the Finger Detection Service.

## 📁 Files Overview

- **`capture_dataset.py`** - Webcam tool to capture hand gesture images
- **`test_dataset.py`** - Test all captured images against the service
- **`test_real_image.py`** - Test individual images
- **`test_with_images.py`** - Test with generated images (for debugging)

## 🚀 Quick Start

### 1. Capture Your Dataset

```bash
# Start the webcam capture tool
python capture_dataset.py
```

**How to use:**
1. Select gesture type (0-5)
2. Position your hand in front of the camera
3. Press **SPACE** to capture
4. Press **ESC** to return to menu
5. Repeat for different gestures and angles

**Supported Gestures:**
- `0` - Close (Closed fist)
- `1` - One (One finger)
- `2` - Two (Two fingers)
- `3` - Three (Three fingers)
- `4` - Four (Four fingers)
- `5` - Open (Open hand)

### 2. Test Your Dataset

```bash
# Test all images in your dataset
python test_dataset.py
```

This will:
- Test every image against your service
- Show detection accuracy
- Provide detailed results
- Save test results to JSON file

### 3. Test Individual Images

```bash
# Test a specific image
python test_real_image.py path/to/your/image.jpg

# Examples:
python test_real_image.py hand_gesture_dataset/open/open_20241226_143022.jpg
python test_real_image.py my_hand_photo.jpg
```

## 📁 Dataset Structure

The capture tool creates this directory structure:

```
hand_gesture_dataset/
├── dataset_info.json          # Metadata and statistics
├── close/                     # Closed fist images
│   ├── close_20241226_143022.jpg
│   └── close_20241226_143045.jpg
├── one/                       # One finger images
│   ├── one_20241226_143100.jpg
│   └── one_20241226_143115.jpg
├── two/                       # Two finger images
├── three/                     # Three finger images
├── four/                      # Four finger images
└── open/                      # Open hand images
```

## 💡 Tips for Better Results

### **Lighting:**
- Ensure good, even lighting
- Avoid shadows on your hand
- Use natural light when possible

### **Hand Position:**
- Keep your hand clearly visible
- Avoid overlapping fingers
- Use consistent background
- Try different angles and positions

### **Image Quality:**
- Capture multiple images per gesture
- Vary hand positions slightly
- Include different lighting conditions
- Use consistent hand size in frame

## 🧪 Testing Workflow

1. **Capture Phase:**
   ```bash
   python capture_dataset.py
   # Capture 10-20 images per gesture
   ```

2. **Test Phase:**
   ```bash
   python test_dataset.py
   # See how well your service performs
   ```

3. **Iterate:**
   - Add more images for low-accuracy gestures
   - Improve lighting/positioning
   - Retest until satisfied

## 🔧 Troubleshooting

### **Webcam Issues:**
- Make sure no other application is using the camera
- Check camera permissions
- Try different camera index (change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`)

### **Service Connection:**
- Ensure your service is running: `python app.py`
- Check service health: `curl http://localhost:8000/health`

### **Image Quality:**
- If detection fails, try better lighting
- Ensure hand is clearly visible
- Avoid complex backgrounds

## 📊 Expected Results

With a good dataset, you should see:
- **Success Rate:** 90%+ images processed successfully
- **Accuracy:** 80%+ correct gesture predictions
- **Confidence:** Average confidence > 0.7

## 🎯 Next Steps

After building your dataset:

1. **Test with Swagger UI:**
   - Open `http://localhost:8000/docs`
   - Upload your images manually

2. **Integrate with applications:**
   - Use the API endpoints in your own code
   - Build web/mobile apps that use the service

3. **Improve the model:**
   - Retrain with your specific dataset
   - Fine-tune for your use case

## 📞 Support

If you encounter issues:
1. Check the service is running
2. Verify webcam permissions
3. Ensure good lighting conditions
4. Check the generated error messages

Happy testing! 🎉
