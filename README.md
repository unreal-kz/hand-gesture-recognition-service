# 🚀 Production Deployment Guide

This guide covers deploying the enhanced Hand Gesture Recognition Service to production with monitoring, caching, and scalability features.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Scaling](#scaling)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The production service includes:

- **Enhanced FastAPI Application** with structured logging and monitoring
- **Redis Caching** for improved performance
- **Rate Limiting** to prevent abuse
- **Batch Processing** for high-throughput scenarios
- **Prometheus Metrics** for observability
- **Grafana Dashboards** for visualization
- **Docker Containerization** with production optimizations
- **Load Balancing** ready for horizontal scaling

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Nginx Proxy   │    │   Prometheus    │
│   (Optional)    │    │   (Optional)    │    │   + Grafana     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   FastAPI App   │
                    │   (Multiple     │
                    │    Instances)   │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │      Redis      │
                    │   (Cache +      │
                    │   Rate Limit)   │
                    └─────────────────┘
```

## ✅ Prerequisites

- **Docker** 20.10+ and **Docker Compose** 2.0+
- **Linux/macOS** (Windows with WSL2)
- **4GB+ RAM** available
- **2+ CPU cores**
- **10GB+ disk space**

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd hand-gesture-recognition-service

# Make deployment script executable
chmod +x deploy_production.sh

# Copy production environment
cp env.production .env
```

### 2. Deploy with Script

```bash
# Run the automated deployment script
./deploy_production.sh
```

### 3. Manual Deployment

```bash
# Create necessary directories
mkdir -p logs monitoring/grafana/{dashboards,datasources} nginx/ssl

# Start services
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps
```

### 4. Verify Deployment

```bash
# Check service health
curl http://localhost:8000/health

# Check metrics
curl http://localhost:8001/metrics

# Check Prometheus
curl http://localhost:9090/-/healthy

# Check Grafana
curl http://localhost:3000/api/health
```

## ⚙️ Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Service Configuration
APP_NAME="Hand Gesture Recognition Service"
APP_VERSION="2.0.0"
DEBUG=false

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=200
RATE_LIMIT_WINDOW=60

# Performance
BATCH_SIZE=20
MAX_IMAGE_SIZE=20971520  # 20MB

# Redis
REDIS_ENABLED=true
REDIS_HOST=redis
REDIS_PORT=6379
```

### Service Configuration

- **Ports**: 8000 (API), 8001 (Metrics)
- **Workers**: 4 (adjust based on CPU cores)
- **Batch Size**: 20 images per batch
- **Rate Limit**: 200 requests per minute
- **Image Size**: 20MB maximum

## 📊 Monitoring

### Prometheus Metrics

Available metrics:

- **Request Rate**: `hand_gesture_requests_total`
- **Response Time**: `hand_gesture_request_duration_seconds`
- **ML Inference**: `ml_inference_total`, `ml_inference_duration_seconds`
- **System**: `active_requests`, `model_loaded`
- **Errors**: `hand_gesture_errors_total`
- **Batch Processing**: `batch_processing_size`, `batch_processing_duration_seconds`

### Grafana Dashboard

Access: http://localhost:3000 (admin/admin123)

Dashboard includes:
- Request rate and response time
- ML inference metrics
- Error rates and types
- Batch processing performance
- System resource usage

### Health Checks

```bash
# Service health
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/status

# Model information
curl http://localhost:8000/model/info
```

## 📈 Scaling

### Horizontal Scaling

```bash
# Scale the main service
docker-compose -f docker-compose.production.yml up -d --scale hand-gesture-service=3

# Check running instances
docker-compose -f docker-compose.production.yml ps
```

### Load Balancer Configuration

```nginx
# nginx/nginx.conf
upstream hand_gesture_backend {
    server hand-gesture-service:8000;
    # Add more instances as needed
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://hand_gesture_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Performance Tuning

```bash
# Increase worker processes
WORKERS=8

# Adjust batch size
BATCH_SIZE=50

# Optimize Redis
REDIS_MAX_MEMORY=512mb
REDIS_MAX_MEMORY_POLICY=allkeys-lru
```

## 🔒 Security

### Production Security Checklist

- [ ] **HTTPS/SSL**: Configure SSL certificates
- [ ] **Authentication**: Implement API key authentication
- [ ] **Rate Limiting**: Enable and configure rate limits
- [ ] **Input Validation**: Validate all inputs
- [ ] **Logging**: Structured logging with sensitive data filtering
- [ ] **Monitoring**: Set up alerts for security events

### SSL Configuration

```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout nginx/ssl/nginx.key \
    -out nginx/ssl/nginx.crt

# Update nginx configuration
# Enable HTTPS in docker-compose.production.yml
```

### API Authentication

```python
# Add to app_enhanced.py
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(token: str = Security(security)):
    if token.credentials != "your-secret-token":
        raise HTTPException(status_code=403, detail="Invalid token")
    return token.credentials

# Use in endpoints
@app.post("/detect-fingers")
async def detect_fingers(
    request: FingerDetectionRequest,
    token: str = Depends(verify_token)
):
    # ... endpoint logic
```

## 🧪 Testing

### Load Testing

```bash
# Install dependencies
pip install aiohttp

# Run load test
python load_test.py --requests 1000 --concurrency 50

# Test batch processing
python load_test.py --requests 100 --concurrency 10 --type batch
```

### API Testing

```bash
# Test single image
curl -X POST "http://localhost:8000/detect-fingers" \
     -H "Content-Type: application/json" \
     -d '{"image_base64": "your_base64_image"}'

# Test batch processing
curl -X POST "http://localhost:8000/detect-fingers-batch" \
     -H "Content-Type: application/json" \
     -d '{"images": [{"image_id": "1", "image_base64": "..."}]}'

# Test file upload
curl -X POST "http://localhost:8000/detect-fingers-upload" \
     -F "file=@your_image.jpg"
```

## 🚨 Troubleshooting

### Common Issues

#### Service Won't Start

```bash
# Check logs
docker-compose -f docker-compose.production.yml logs hand-gesture-service

# Check resource usage
docker stats

# Verify configuration
docker-compose -f docker-compose.production.yml config
```

#### High Memory Usage

```bash
# Check Redis memory
docker-compose -f docker-compose.production.yml exec redis redis-cli info memory

# Check service memory
docker stats hand-gesture-service

# Adjust batch size and workers
BATCH_SIZE=10
WORKERS=2
```

#### Slow Response Times

```bash
# Check metrics
curl http://localhost:8001/metrics | grep duration

# Check Redis performance
docker-compose -f docker-compose.production.yml exec redis redis-cli info stats

# Monitor system resources
htop
```

### Performance Optimization

```bash
# Enable GPU support (if available)
docker run --gpus all your-image

# Optimize image preprocessing
# Reduce image resolution before processing

# Use Redis clustering for high availability
# Configure Redis sentinel or cluster mode
```

### Monitoring Alerts

Set up alerts for:
- High error rates (>5%)
- Slow response times (>2s p95)
- High memory usage (>80%)
- Service unavailability
- High CPU usage (>80%)

## 📚 Additional Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Redis Documentation](https://redis.io/documentation)

### Monitoring Best Practices
- Set up alerting rules
- Create custom dashboards
- Monitor business metrics
- Set up log aggregation
- Implement distributed tracing

### Production Checklist
- [ ] Load testing completed
- [ ] Monitoring configured
- [ ] Alerts set up
- [ ] Backup strategy implemented
- [ ] Disaster recovery plan
- [ ] Security audit completed
- [ ] Performance benchmarks
- [ ] Documentation updated

## 🤝 Support

For production support:

1. Check the troubleshooting section
2. Review monitoring dashboards
3. Check service logs
4. Verify configuration
5. Test with load testing tools
6. Contact the development team

---

**Happy Deploying! 🎉**
