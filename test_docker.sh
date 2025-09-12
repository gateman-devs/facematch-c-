#!/bin/bash

# Test Docker deployment script for Gateman Face Service
# This script tests the Docker build and basic functionality

set -e

IMAGE_NAME="gateman-face-test"
CONTAINER_NAME="gateman-face-test-container"

echo "🧪 Testing Gateman Face Service Docker deployment"
echo "==============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    docker rmi $IMAGE_NAME 2>/dev/null || true
}

# Error handling
error_exit() {
    echo -e "\n${RED}❌ Test failed: $1${NC}"
    cleanup
    exit 1
}

trap cleanup EXIT

echo "📦 Building Docker image..."
docker build -t $IMAGE_NAME . || error_exit "Failed to build Docker image"

echo "🐳 Starting container..."
docker run -d --name $CONTAINER_NAME -p 8080:8080 $IMAGE_NAME || error_exit "Failed to start container"

echo "⏳ Waiting for service to be ready..."
sleep 15

# Check if container is running
if ! docker ps | grep -q $CONTAINER_NAME; then
    error_exit "Container is not running"
fi

echo "🏥 Testing health endpoint..."
if ! curl -f -s http://localhost:8080/health > /dev/null; then
    error_exit "Health endpoint is not responding"
fi
echo -e "${GREEN}✓ Health endpoint is working${NC}"

echo "🔍 Testing API endpoints..."

# Test compare-faces endpoint
if curl -s -X POST http://localhost:8080/compare-faces \
    -H "Content-Type: application/json" \
    -d '{"image1": "test", "image2": "test"}' | grep -q '"success":true'; then
    echo -e "${GREEN}✓ Compare-faces endpoint is working${NC}"
else
    error_exit "Compare-faces endpoint failed"
fi

# Test liveness-check endpoint
if curl -s -X POST http://localhost:8080/liveness-check \
    -H "Content-Type: application/json" \
    -d '{"image": "test"}' | grep -q '"success":true'; then
    echo -e "${GREEN}✓ Liveness-check endpoint is working${NC}"
else
    error_exit "Liveness-check endpoint failed"
fi

# Test generate-challenge endpoint
if curl -s -X POST http://localhost:8080/generate-challenge \
    -H "Content-Type: application/json" \
    -d '{}' | grep -q '"success":true'; then
    echo -e "${GREEN}✓ Generate-challenge endpoint is working${NC}"
else
    error_exit "Generate-challenge endpoint failed"
fi

echo -e "\n${GREEN}🎉 All tests passed! Docker deployment is working correctly.${NC}"
echo "📊 Container logs:"
docker logs $CONTAINER_NAME | tail -20

cleanup
echo -e "\n${GREEN}✅ Test completed successfully${NC}"
