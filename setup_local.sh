#!/bin/bash

# Full Face Recognition Service - Local Setup Script  
# This script builds and configures the complete ML-based face service with all endpoints

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "=== Full Face Recognition Service - Local Setup ==="
echo "Project directory: $SCRIPT_DIR"
echo "Build directory: $BUILD_DIR" 
echo "===================================================="

# Default configuration
PORT=8080
FORCE_REBUILD=false
VERBOSE=false

# Function to print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --port PORT       Server port (default: 8080)"
    echo "  --rebuild         Force rebuild even if binary exists"
    echo "  --verbose         Enable verbose output"
    echo "  --help, -h        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                        # Build with default settings"
    echo "  $0 --port 9000           # Set port for future runs"
    echo "  $0 --rebuild             # Force rebuild"
    echo ""
}

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to log verbose messages
log_verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo "[VERBOSE] $1"
    fi
}

# Function to check build dependencies
check_build_deps() {
    log "Checking build dependencies..."
    
    local missing_deps=()
    
    if ! command -v cmake >/dev/null 2>&1; then
        missing_deps+=("cmake")
    fi
    
    if ! command -v make >/dev/null 2>&1; then
        missing_deps+=("make")
    fi
    
    if ! command -v pkg-config >/dev/null 2>&1; then
        missing_deps+=("pkg-config")
    fi
    
    # Check for C++ compiler
    if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
        missing_deps+=("g++ or clang++")
    fi
    
    # Check for OpenCV
    if ! pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
        missing_deps+=("opencv")
    fi
    
    # Check for CURL
    if ! pkg-config --exists libcurl; then
        missing_deps+=("libcurl")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log "✗ Missing build dependencies:"
        for dep in "${missing_deps[@]}"; do
            log "  - $dep"
        done
        log ""
        log "Please run './setup_environment.sh' to install dependencies"
        return 1
    else
        log "✓ All build dependencies are available"
        return 0
    fi
}

# Function to build the full web service
build_project() {
    log "Building Full Face Recognition Service..."
    
    # Create build directory
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Run CMake
    log "Running CMake configuration..."
    if [[ "$VERBOSE" == "true" ]]; then
        cmake .. -DCMAKE_BUILD_TYPE=Release
    else
        cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null
    fi
    
    # Build the project
    log "Compiling source code..."
    local cpu_count=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "4")
    
    if [[ "$VERBOSE" == "true" ]]; then
        make -j"$cpu_count"
    else
        make -j"$cpu_count" >/dev/null
    fi
    
    # Check if full web server binary was created
    local build_success=false
    if [[ -f "$BUILD_DIR/MLFaceService" ]]; then
        log "✓ Built: MLFaceService (full web server with all endpoints)"
        build_success=true
    fi
    
    # Also check for lightweight binaries as fallback
    for binary in "test_lightweight" "lightweight_server"; do
        if [[ -f "$BUILD_DIR/$binary" ]]; then
            log "✓ Built: $binary"
        fi
    done
    
    if [[ "$build_success" == "true" ]]; then
        log "✓ Build completed successfully"
        cd "$SCRIPT_DIR"
        return 0
    else
        log "✗ Build failed - MLFaceService binary not found"
        log "Make sure all dependencies are installed (dlib, crow, nlohmann/json, hiredis)"
        cd "$SCRIPT_DIR"
        return 1
    fi
}

# Function to check if rebuild is needed
needs_rebuild() {
    local binary_path="$BUILD_DIR/MLFaceService"
    
    # Force rebuild if requested
    if [[ "$FORCE_REBUILD" == "true" ]]; then
        log "Force rebuild requested"
        return 0
    fi
    
    # Rebuild if binary doesn't exist
    if [[ ! -f "$binary_path" ]]; then
        log "Binary not found, build required"
        return 0
    fi
    
    # Check if source files are newer than binary
    local binary_time=$(stat -f%m "$binary_path" 2>/dev/null || stat -c%Y "$binary_path" 2>/dev/null || echo "0")
    
    while IFS= read -r -d '' source_file; do
        local source_time=$(stat -f%m "$source_file" 2>/dev/null || stat -c%Y "$source_file" 2>/dev/null || echo "0")
        if [[ "$source_time" -gt "$binary_time" ]]; then
            log "Source file updated, rebuild required: $(basename "$source_file")"
            return 0
        fi
    done < <(find "$SCRIPT_DIR/src" -name "*.cpp" -o -name "*.hpp" -print0 2>/dev/null)
    
    # Check if CMakeLists.txt is newer
    if [[ -f "$SCRIPT_DIR/CMakeLists.txt" ]]; then
        local cmake_time=$(stat -f%m "$SCRIPT_DIR/CMakeLists.txt" 2>/dev/null || stat -c%Y "$SCRIPT_DIR/CMakeLists.txt" 2>/dev/null || echo "0")
        if [[ "$cmake_time" -gt "$binary_time" ]]; then
            log "CMakeLists.txt updated, rebuild required"
            return 0
        fi
    fi
    
    log "Binary is up to date, skipping build"
    return 1
}

# Function to run tests
run_tests() {
    log "Running web server tests..."
    cd "$BUILD_DIR"
    
    if [[ -f "./test_lightweight" ]]; then
        log "Running lightweight test suite..."
        ./test_lightweight
    else
        log "⚠ Test binary not found, skipping tests..."
        return 0
    fi
    
    cd "$SCRIPT_DIR"
}

# Function to show final information
show_final_info() {
    log ""
    log "=============================================="
    log "✓ Face Recognition Service Setup Complete!"
    log ""
    log "Built binaries:"
    for binary in "MLFaceService" "test_lightweight" "lightweight_server"; do
        if [[ -f "$BUILD_DIR/$binary" ]]; then
            log "  - $binary"
        fi
    done
    log ""
    log "Key features:"
    log "  ✓ Full ML models (dlib face recognition)"
    log "  ✓ Face comparison with confidence scoring"
    log "  ✓ Liveness detection (anti-spoofing)"
    log "  ✓ Challenge-based video liveness verification"
    log "  ✓ Redis caching for challenges"
    log "  ✓ RESTful API with all endpoints"
    log ""
    log "Available endpoints:"
    log "  GET  /health           - Health check"
    log "  POST /compare-faces    - Compare two faces"
    log "  POST /liveness-check   - Check face liveness"
    log "  POST /generate-challenge - Generate liveness challenge"
    log "  POST /verify-video-liveness - Verify challenge videos"
    log ""
    log "Manual commands:"
    log "  Run tests: cd $BUILD_DIR && ./test_lightweight"
    log "  Start full server: cd $BUILD_DIR && ./MLFaceService --port $PORT --models ../models"
    log "=============================================="
}

# Function to start the server
start_server() {
    local server_path="$BUILD_DIR/MLFaceService"
    local models_path="$SCRIPT_DIR/models"
    
    # Check if server binary exists
    if [[ ! -f "$server_path" ]]; then
        log "✗ Full web server binary not found: $server_path"
        log "Please rebuild with: $0 --rebuild"
        return 1
    fi
    
    if [[ ! -x "$server_path" ]]; then
        log "✗ Server binary is not executable: $server_path"
        return 1
    fi
    
    # Check if models directory exists
    if [[ ! -d "$models_path" ]]; then
        log "⚠ Models directory not found: $models_path"
        log "Downloading models first..."
        cd "$SCRIPT_DIR"
        if [[ -f "./download_models.sh" ]]; then
            chmod +x ./download_models.sh
            ./download_models.sh
        else
            log "✗ download_models.sh script not found"
            log "Please ensure ML models are available in: $models_path"
        fi
    fi
    
    log "Server binary: $server_path"
    log "Models path: $models_path"
    log "Port: $PORT"
    log ""
    log "Starting Full Face Recognition Service with all endpoints..."
    log ""
    
    # Change to build directory and start server
    cd "$BUILD_DIR"
    exec "./MLFaceService" --port "$PORT" --models "$models_path"
}

# Function to handle cleanup on exit
cleanup() {
    cd "$SCRIPT_DIR" 2>/dev/null || true
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                PORT="$2"
                shift 2
            else
                echo "Error: --port requires a numeric value"
                exit 1
            fi
            ;;
        --rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate port range
if [[ "$PORT" -lt 1 || "$PORT" -gt 65535 ]]; then
    log "✗ Invalid port number: $PORT (must be 1-65535)"
    exit 1
fi

# Main execution
main() {
    log "Starting Full Face Recognition Service setup..."
    
    # Check build dependencies
    if ! check_build_deps; then
        exit 1
    fi
    
    # Build if necessary
    if needs_rebuild; then
        if ! build_project; then
            log "✗ Build failed. Cannot proceed."
            exit 1
        fi
    fi
    
    # Run tests to verify functionality
    if ! run_tests; then
        log "⚠ Tests failed, but build completed"
        log "You can still use the implementation for integration"
    fi
    
    # Show final information
    show_final_info
    
    log "✓ Setup completed successfully!"
    log ""
    log "Starting Full Face Recognition Service Server..."
    start_server
}

# Run main function
main "$@"