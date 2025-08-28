#!/bin/bash

# Production Deployment Script for Hand Gesture Recognition Service
# This script automates the deployment process for production

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="hand-gesture-service"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env.example"

echo -e "${BLUE}🚀 Starting Production Deployment...${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install it and try again.${NC}"
    exit 1
fi

# Create .env file from production template
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}📝 Creating .env file from template...${NC}"
    cp "$ENV_FILE" .env
    echo -e "${GREEN}✅ .env file created${NC}"
else
    echo -e "${YELLOW}⚠️  .env file already exists. Using existing configuration.${NC}"
fi

# Create necessary directories
echo -e "${BLUE}📁 Creating necessary directories...${NC}"
mkdir -p logs monitoring/grafana/dashboards monitoring/grafana/datasources nginx/ssl

# Copy monitoring configurations if they don't exist
if [ ! -f "monitoring/grafana/dashboards/hand-gesture-dashboard.json" ]; then
    echo -e "${YELLOW}📊 Setting up monitoring configurations...${NC}"
    # The monitoring files should already be created by the previous steps
fi

# Stop existing services
echo -e "${BLUE}🛑 Stopping existing services...${NC}"
docker-compose -f "$COMPOSE_FILE" down --remove-orphans || true

# Clean up old images (optional)
read -p "Do you want to remove old images? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}🧹 Cleaning up old images...${NC}"
    docker system prune -f
fi

# Build and start services
echo -e "${BLUE}🔨 Building and starting services...${NC}"
docker-compose -f "$COMPOSE_FILE" up --build -d

# Wait for services to be ready
echo -e "${BLUE}⏳ Waiting for services to be ready...${NC}"
sleep 30

# Check service health
echo -e "${BLUE}🏥 Checking service health...${NC}"
for i in {1..10}; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Service is healthy!${NC}"
        break
    else
        echo -e "${YELLOW}⏳ Waiting for service to be ready... (attempt $i/10)${NC}"
        sleep 10
    fi
    
    if [ $i -eq 10 ]; then
        echo -e "${RED}❌ Service failed to become healthy after 10 attempts${NC}"
        echo -e "${YELLOW}📋 Checking service logs...${NC}"
        docker-compose -f "$COMPOSE_FILE" logs "$SERVICE_NAME"
        exit 1
    fi
done

# Check Redis
echo -e "${BLUE}🔍 Checking Redis connection...${NC}"
if docker-compose -f "$COMPOSE_FILE" exec redis redis-cli ping | grep -q "PONG"; then
    echo -e "${GREEN}✅ Redis is running${NC}"
else
    echo -e "${RED}❌ Redis is not responding${NC}"
fi

# Check Prometheus
echo -e "${BLUE}🔍 Checking Prometheus...${NC}"
if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Prometheus is running${NC}"
else
    echo -e "${RED}❌ Prometheus is not responding${NC}"
fi

# Check Grafana
echo -e "${BLUE}🔍 Checking Grafana...${NC}"
if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Grafana is running${NC}"
else
    echo -e "${RED}❌ Grafana is not responding${NC}"
fi

# Display service information
echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo -e "${BLUE}📋 Service Information:${NC}"
echo -e "   🌐 Main Service: http://localhost:8000"
echo -e "   📊 API Docs: http://localhost:8000/docs"
echo -e "   🏥 Health Check: http://localhost:8000/health"
echo -e "   📈 Metrics: http://localhost:8001/metrics"
echo -e "   📊 Prometheus: http://localhost:9090"
echo -e "   📈 Grafana: http://localhost:3000 (admin/admin123)"
echo -e "   🔍 Redis: localhost:6379"

# Show running services
echo -e "${BLUE}🐳 Running Services:${NC}"
docker-compose -f "$COMPOSE_FILE" ps

# Show logs
echo -e "${BLUE}📋 Recent logs:${NC}"
docker-compose -f "$COMPOSE_FILE" logs --tail=20 "$SERVICE_NAME"

echo -e "${GREEN}✅ Production deployment completed!${NC}"
echo -e "${YELLOW}💡 Next steps:${NC}"
echo -e "   1. Test the API endpoints"
echo -e "   2. Configure Grafana dashboards"
echo -e "   3. Set up monitoring alerts"
echo -e "   4. Configure SSL/TLS if needed"
echo -e "   5. Set up backup and recovery procedures"
