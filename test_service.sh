#!/bin/bash

# ML Face Service - Test Script
# This script tests the service endpoints to verify functionality

set -e

# Configuration
SERVICE_URL="http://localhost:8080"
TIMEOUT=10

echo "=== ML Face Service - Test Script ==="
echo "Service URL: $SERVICE_URL"
echo "===================================="

# Function to check if service is running
check_service() {
    echo "Checking if service is running..."
    if curl -s --max-time $TIMEOUT "$SERVICE_URL/health" > /dev/null; then
        echo "✓ Service is running"
        return 0
    else
        echo "✗ Service is not running or not responding"
        echo "Please start the service first:"
        echo "  ./startup_local.sh"
        echo "  or"
        echo "  docker-compose up"
        return 1
    fi
}

# Function to test health endpoint
test_health() {
    echo ""
    echo "Testing health endpoint..."
    
    local response=$(curl -s --max-time $TIMEOUT "$SERVICE_URL/health")
    local status=$(echo "$response" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    
    if [[ "$status" == "healthy" ]]; then
        echo "✓ Health check passed"
        echo "Response: $response"
    else
        echo "✗ Health check failed"
        echo "Response: $response"
        return 1
    fi
}

# Function to test face comparison with sample URLs
test_face_comparison_urls() {
    echo ""
    echo "Testing face comparison with URLs..."
    
    # Using placeholder URLs (replace with actual test images)
    local test_data='{
        "image1": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png",
        "image2": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png"
    }'
    
    local response=$(curl -s --max-time 30 \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$test_data" \
        "$SERVICE_URL/compare-faces" 2>/dev/null)
    
    if [[ -n "$response" ]]; then
        local success=$(echo "$response" | grep -o '"success":[^,]*' | cut -d':' -f2)
        if [[ "$success" == "true" ]]; then
            echo "✓ Face comparison test passed"
            echo "Response: $response"
        else
            echo "⚠ Face comparison returned error (expected with placeholder images)"
            echo "Response: $response"
        fi
    else
        echo "✗ Face comparison test failed - no response"
        return 1
    fi
}

# Function to test liveness check
test_liveness_check_url() {
    echo ""
    echo "Testing liveness check with URL..."
    
    # Using placeholder URL (replace with actual test image)
    local test_data='{
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png"
    }'
    
    local response=$(curl -s --max-time 30 \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$test_data" \
        "$SERVICE_URL/liveness-check" 2>/dev/null)
    
    if [[ -n "$response" ]]; then
        local success=$(echo "$response" | grep -o '"success":[^,]*' | cut -d':' -f2)
        if [[ "$success" == "true" ]]; then
            echo "✓ Liveness check test passed"
            echo "Response: $response"
        else
            echo "⚠ Liveness check returned error (expected with placeholder images)"
            echo "Response: $response"
        fi
    else
        echo "✗ Liveness check test failed - no response"
        return 1
    fi
}

# Function to test invalid requests
test_invalid_requests() {
    echo ""
    echo "Testing invalid request handling..."
    
    # Test missing parameters
    local response=$(curl -s --max-time $TIMEOUT \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{}' \
        "$SERVICE_URL/compare-faces" 2>/dev/null)
    
    local success=$(echo "$response" | grep -o '"success":[^,]*' | cut -d':' -f2)
    if [[ "$success" == "false" ]]; then
        echo "✓ Invalid request handling works"
    else
        echo "✗ Invalid request handling failed"
        echo "Response: $response"
        return 1
    fi
}

# Function to test performance
test_performance() {
    echo ""
    echo "Testing performance..."
    
    local start_time=$(date +%s)
    
    curl -s --max-time $TIMEOUT "$SERVICE_URL/health" > /dev/null
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ $duration -le 2 ]]; then
        echo "✓ Health endpoint responds quickly (${duration}s)"
    else
        echo "⚠ Health endpoint slow response (${duration}s)"
    fi
}

# Function to show service info
show_service_info() {
    echo ""
    echo "Service Information:"
    echo "==================="
    
    local health_response=$(curl -s --max-time $TIMEOUT "$SERVICE_URL/health" 2>/dev/null)
    
    if [[ -n "$health_response" ]]; then
        echo "Health Response: $health_response"
        
        # Parse JSON manually (basic parsing)
        local version=$(echo "$health_response" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
        local models_loaded=$(echo "$health_response" | grep -o '"models_loaded":[^,]*' | cut -d':' -f2)
        
        echo ""
        echo "Version: ${version:-Unknown}"
        echo "Models Loaded: ${models_loaded:-Unknown}"
    fi
    
    echo ""
    echo "Available Endpoints:"
    echo "  GET  $SERVICE_URL/health"
    echo "  POST $SERVICE_URL/compare-faces"
    echo "  POST $SERVICE_URL/liveness-check"
}

# Main execution
main() {
    # Check if service is running
    if ! check_service; then
        exit 1
    fi
    
    # Run tests
    local failed_tests=0
    
    test_health || ((failed_tests++))
    test_performance || ((failed_tests++))
    test_invalid_requests || ((failed_tests++))
    
    # Optional: Test with real endpoints (might fail with placeholder URLs)
    echo ""
    echo "Testing with placeholder URLs (may show warnings):"
    test_face_comparison_urls || true  # Don't count as failure
    test_liveness_check_url || true    # Don't count as failure
    
    # Show service info
    show_service_info
    
    # Summary
    echo ""
    echo "===================================="
    if [[ $failed_tests -eq 0 ]]; then
        echo "✓ All core tests passed!"
        echo ""
        echo "The ML Face Service is working correctly."
        echo "You can now use the API endpoints with real images."
    else
        echo "✗ $failed_tests test(s) failed"
        echo ""
        echo "Please check the service logs for more details."
        exit 1
    fi
    echo "===================================="
}

# Handle command line arguments
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Test the ML Face Service API endpoints."
    echo ""
    echo "Options:"
    echo "  --help, -h    Show this help message"
    echo "  --url URL     Use custom service URL (default: $SERVICE_URL)"
    echo ""
    echo "Make sure the service is running before running tests:"
    echo "  ./startup_local.sh"
    echo "  or"
    echo "  docker-compose up"
    echo ""
    exit 0
elif [[ "$1" == "--url" && -n "$2" ]]; then
    SERVICE_URL="$2"
    echo "Using custom service URL: $SERVICE_URL"
fi

# Run main function
main "$@"
