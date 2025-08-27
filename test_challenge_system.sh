#!/bin/bash

# Test script for the challenge system
SERVER_URL="http://localhost:8080"

echo "=== Testing Challenge System ==="
echo

# Test 0: Check Redis connection
echo "0. Checking Redis status..."
if command -v redis-cli >/dev/null 2>&1; then
    if redis-cli ping >/dev/null 2>&1; then
        echo "✓ Redis is running and accessible"
    else
        echo "⚠ Redis CLI found but Redis is not responding"
        echo "  Start Redis with: ./setup_redis.sh"
    fi
else
    echo "⚠ Redis CLI not found"
    echo "  Install Redis with: ./setup_redis.sh"
fi
echo

# Test 1: Health check
echo "1. Testing health check..."
HEALTH_RESPONSE=$(curl -s -X GET "${SERVER_URL}/health")
echo "$HEALTH_RESPONSE" | jq '.'

# Check if Redis is connected according to the service
REDIS_CONNECTED=$(echo "$HEALTH_RESPONSE" | jq -r '.redis_connected // false')
CHALLENGE_AVAILABLE=$(echo "$HEALTH_RESPONSE" | jq -r '.challenge_system_available // false')

echo "Redis connected: $REDIS_CONNECTED"
echo "Challenge system available: $CHALLENGE_AVAILABLE"
echo

# Test 2: Generate challenge
echo "2. Generating challenge..."
CHALLENGE_RESPONSE=$(curl -s -X POST "${SERVER_URL}/generate-challenge" \
    -H "Content-Type: application/json" \
    -d '{"ttl_seconds": 600}')

echo "Challenge response:"
echo "$CHALLENGE_RESPONSE" | jq '.'
echo

# Extract challenge ID and directions
CHALLENGE_ID=$(echo "$CHALLENGE_RESPONSE" | jq -r '.challenge_id')
DIRECTIONS=$(echo "$CHALLENGE_RESPONSE" | jq -r '.directions[]')

echo "Challenge ID: $CHALLENGE_ID"
echo "Directions: $DIRECTIONS"
echo

# Test 3: Attempt to verify challenge (this will fail without actual videos)
echo "3. Testing challenge verification (expected to fail without valid videos)..."
curl -s -X POST "${SERVER_URL}/verify-video-liveness" \
    -H "Content-Type: application/json" \
    -d "{
        \"challenge_id\": \"$CHALLENGE_ID\",
        \"video_urls\": [
            \"https://example.com/video1.mp4\",
            \"https://example.com/video2.mp4\",
            \"https://example.com/video3.mp4\",
            \"https://example.com/video4.mp4\"
        ]
    }" | jq '.'
echo

echo "=== Test completed ==="
