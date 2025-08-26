# 🤖 Hand Gesture Recognition Service

A real-time hand gesture recognition service built with FastAPI, MediaPipe, and TensorFlow Lite. This service can detect and classify hand gestures including Open, Close, One, Two, Three, and Four finger positions.

## ✨ Features

- **Real-time Hand Gesture Recognition** - Detect 6 different hand gestures
- **FastAPI Web Service** - RESTful API with automatic documentation
- **MediaPipe Integration** - Advanced hand landmark detection
- **TensorFlow Lite Models** - Optimized for performance
- **Dataset Management Tools** - Capture and evaluate your own datasets
- **Performance Metrics** - F1 score, precision, recall, and accuracy evaluation
- **Docker Support** - Easy deployment and containerization

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Webcam (for dataset capture)
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd hand-gesture-recognition-service
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv .env
   source .env/bin/activate  # On Windows: .env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the service:**
   ```bash
   python app.py
   ```

The service will be available at `http://localhost:8000`

## 📖 API Documentation

Once the service is running, visit:
- **Interactive API Docs:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health`
- **Finger Detection:** `http://localhost:8000/detect-fingers`

### API Endpoints

#### `POST /detect-fingers`
Detect hand gestures in an image.

**Request Body:**
```json
{
  "image_base64": "base64_encoded_image_string"
}
```

**Response:**
```json
{
  "success": true,
  "detected_gesture": "Open",
  "confidence": 0.95,
  "processing_time_ms": 120,
  "num_fingers": 5
}
```

#### `GET /health`
Service health check.

**Response:**
```json
{
  "status": "healthy",
  "service": "finger-detection-service",
  "version": "1.0.0",
  "models_loaded": true
}
```

## 🎯 Supported Gestures

| Gesture | Description | Fingers |
|---------|-------------|---------|
| **Open** | Open palm | 5 fingers |
| **Close** | Closed fist | 0 fingers |
| **One** | Pointing with index finger | 1 finger |
| **Two** | Peace sign | 2 fingers |
| **Three** | Three fingers extended | 3 fingers |
| **Four** | Four fingers extended | 4 fingers |

## 📸 Dataset Management

### Capture Your Own Dataset

Use the built-in dataset capture tool to create your own hand gesture dataset:

```bash
python capture_dataset.py
```

This tool will:
- Use your webcam to capture images
- Organize images by gesture type
- Create a structured dataset directory
- Generate metadata for each image

### Evaluate Performance

Test your service against the captured dataset:

```bash
python simple_evaluate.py
```

This will provide:
- **F1 Score** - Harmonic mean of precision and recall
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **Accuracy** - Overall correct predictions
- **Per-gesture breakdown** - Performance for each gesture type

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
docker-compose up --build
```

### Manual Docker Build

```bash
docker build -t hand-gesture-service .
docker run -p 8000:8000 hand-gesture-service
```

## 📊 Performance

Based on testing with real hand gesture images:

- **Overall Accuracy:** 79.55%
- **Macro F1 Score:** 88.36%
- **Best Performing Gestures:** Two (100%), Three (100%)
- **Processing Time:** ~100-120ms per image

## 🏗️ Project Structure

```
hand-gesture-recognition-service/
├── app.py                      # Main FastAPI application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image definition
├── docker-compose.yml         # Docker Compose configuration
├── capture_dataset.py         # Dataset capture tool
├── simple_evaluate.py         # Performance evaluation script
├── README.md                  # This file
├── README_DATASET.md          # Dataset management guide
├── model/                     # ML models and classifier
│   ├── keypoint_classifier.py
│   └── keypoint_classifier/   # TensorFlow Lite models
├── hand_gesture_dataset/      # Your captured dataset
│   ├── open/                  # Open gesture images
│   ├── close/                 # Close gesture images
│   ├── one/                   # One finger images
│   ├── two/                   # Two finger images
│   ├── three/                 # Three finger images
│   └── four/                  # Four finger images
└── .gitignore                 # Git ignore rules
```

## 🔧 Configuration

### Environment Variables

- `PORT` - Service port (default: 8000)
- `HOST` - Service host (default: 0.0.0.0)
- `LOG_LEVEL` - Logging level (default: INFO)

### Model Configuration

The service uses pre-trained TensorFlow Lite models for:
- Hand landmark detection (MediaPipe)
- Gesture classification (Custom model)

## 🧪 Testing

### Manual Testing

1. **Start the service:**
   ```bash
   python app.py
   ```

2. **Test with curl:**
   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # Test image (replace with your image)
   curl -X POST "http://localhost:8000/detect-fingers" \
        -H "Content-Type: application/json" \
        -d '{"image_base64": "your_base64_image"}'
   ```

### Automated Testing

Run the evaluation script against your dataset:
```bash
python simple_evaluate.py
```

## 🚨 Troubleshooting

### Common Issues

1. **Service won't start:**
   - Check if port 8000 is free: `lsof -i :8000`
   - Ensure all dependencies are installed
   - Check Python version (3.12+ required)

2. **Model loading errors:**
   - Verify model files exist in `model/keypoint_classifier/`
   - Check file permissions
   - Ensure TensorFlow is properly installed

3. **Webcam issues:**
   - Ensure webcam is not in use by another application
   - Check webcam permissions
   - Try different webcam index in capture script

4. **Performance issues:**
   - Use GPU if available (CUDA-enabled TensorFlow)
   - Optimize image resolution
   - Check system resources

### Debug Mode

Enable debug logging:
```bash
LOG_LEVEL=DEBUG python app.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Hand landmark detection
- [TensorFlow Lite](https://www.tensorflow.org/lite) - Model inference
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [OpenCV](https://opencv.org/) - Computer vision library

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the API documentation at `/docs`
3. Open an issue on GitHub
4. Check the logs for error messages

---

**Happy Gesturing! 🎉**
