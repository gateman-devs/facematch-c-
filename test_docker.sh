#!/bin/bash

# ML Face Service - Docker Test Script
# Tests the Docker deployment of the face recognition service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="gateman-face-service"
REDIS_CONTAINER_NAME="gateman-redis"

echo "=== ML Face Service - Docker Test ==="
echo "Testing Docker deployment..."
echo ""

# Function to cleanup containers
cleanup() {
    echo "Cleaning up containers..."
    docker-compose down -v 2>/dev/null || true
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker stop $REDIS_CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    docker rm $REDIS_CONTAINER_NAME 2>/dev/null || true
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local timeout=60
    local count=0

    echo "Waiting for service at $url..."
    while ! curl -f -s "$url" >/dev/null 2>&1; do
        if [ $count -ge $timeout ]; then
            echo "Timeout waiting for service"
            return 1
        fi
        count=$((count + 1))
        sleep 1
    done
    echo "Service is ready!"
    return 0
}

# Function to test health endpoint
test_health() {
    echo "Testing health endpoint..."
    local response=$(curl -s http://localhost:8080/health)

    if [[ $response == *"healthy"* ]]; then
        echo "✓ Health check passed"
        return 0
    else
        echo "✗ Health check failed. Response: $response"
        return 1
    fi
}

# Function to test face comparison with sample data
test_face_comparison() {
    echo "Testing face comparison endpoint..."

    # Create a simple test image (1x1 pixel PNG in base64)
    local test_image="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77yQAAAABJRU5ErkJggg=="

    local payload=$(cat <<EOF
{
    "image1": "$test_image",
    "image2": "$test_image"
}
EOF
    )

    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        http://localhost:8080/compare-faces)

    if [[ $response == *"error"* ]] && [[ $response == *"face"* ]]; then
        echo "✓ Face comparison endpoint responded (expected error for test image)"
        return 0
    elif [[ $response == *"success"* ]]; then
        echo "✓ Face comparison endpoint working"
        return 0
    else
        echo "✗ Face comparison test failed. Response: $response"
        return 1
    fi
}

# Function to test challenge generation
test_challenge_generation() {
    echo "Testing challenge generation endpoint..."

    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d '{}' \
        http://localhost:8080/generate-challenge)

    if [[ $response == *"success"* ]] && [[ $response == *"challenge_id"* ]]; then
        echo "✓ Challenge generation working"
        return 0
    else
        echo "✗ Challenge generation failed. Response: $response"
        return 1
    fi
}

# Main test execution
main() {
    cd "$SCRIPT_DIR"

    # Cleanup any existing containers
    cleanup

    echo "Starting services with docker-compose..."
    if ! docker-compose up -d; then
        echo "✗ Failed to start services with docker-compose"
        exit 1
    fi

    echo ""
    echo "Waiting for services to start..."

    # Wait for Redis
    if ! docker exec $REDIS_CONTAINER_NAME redis-cli ping >/dev/null 2>&1; then
        echo "Waiting for Redis..."
        sleep 5
    fi

    # Wait for the face service
    if ! wait_for_service "http://localhost:8080/health"; then
        echo "✗ Service failed to start within timeout"
        docker-compose logs
        cleanup
        exit 1
    fi

    echo ""
    echo "Running tests..."

    local tests_passed=0
    local tests_total=0

    # Test health endpoint
    tests_total=$((tests_total + 1))
    if test_health; then
        tests_passed=$((tests_passed + 1))
    fi

    echo ""

    # Test face comparison
    tests_total=$((tests_total + 1))
    if test_face_comparison; then
        tests_passed=$((tests_passed + 1))
    fi

    echo ""

    # Test challenge generation
    tests_total=$((tests_total + 1))
    if test_challenge_generation; then
        tests_passed=$((tests_passed + 1))
    fi

    echo ""
    echo "=== Test Results ==="
    echo "Passed: $tests_passed/$tests_total tests"

    if [ $tests_passed -eq $tests_total ]; then
        echo "✓ All tests passed!"
        echo ""
        echo "Your Docker deployment is working correctly."
        echo "Service is available at: http://localhost:8080"
        echo ""
        echo "To view logs: docker-compose logs -f"
        echo "To stop: docker-compose down"
    else
        echo "✗ Some tests failed. Check the logs above for details."
        echo ""
        echo "To view detailed logs: docker-compose logs"
        docker-compose logs
    fi

    # Cleanup
    cleanup
}

# Handle script interruption
trap cleanup EXIT INT TERM

# Check if docker and docker-compose are available
if ! command -v docker >/dev/null 2>&1; then
    echo "✗ Docker is not installed or not in PATH"
    exit 1
fi

if ! command -v docker-compose >/dev/null 2>&1; then
    echo "✗ docker-compose is not installed or not in PATH"
    exit 1
fi

# Run main function
main "$@"
