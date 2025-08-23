#!/bin/bash

# ML Face Service - Local Startup Script
# This script handles model downloading, building, and starting the service locally

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
MODELS_DIR="$SCRIPT_DIR/models"

echo "=== ML Face Service - Local Startup ==="
echo "Project directory: $SCRIPT_DIR"
echo "Build directory: $BUILD_DIR"
echo "Models directory: $MODELS_DIR"
echo "======================================="

# Default configuration
PORT=8080
FORCE_REBUILD=false
SKIP_MODELS=false
VERBOSE=false

# Function to print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --port PORT       Server port (default: 8080)"
    echo "  --rebuild         Force rebuild even if binary exists"
    echo "  --skip-models     Skip model download check"
    echo "  --verbose         Enable verbose output"
    echo "  --help, -h        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                        # Start with default settings"
    echo "  $0 --port 9000           # Start on port 9000"
    echo "  $0 --rebuild             # Force rebuild and start"
    echo "  $0 --skip-models         # Skip model check (faster startup)"
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

# Function to check if models exist and are valid
check_models() {
    log "Checking ML models..."
    
    local required_models=(
        "dlib_face_recognition_resnet_model_v1.dat"
        "shape_predictor_68_face_landmarks.dat"
    )
    
    local missing_models=()
    
    for model in "${required_models[@]}"; do
        local model_path="$MODELS_DIR/$model"
        if [[ ! -f "$model_path" ]]; then
            missing_models+=("$model")
        else
            local model_size=$(stat -f%z "$model_path" 2>/dev/null || stat -c%s "$model_path" 2>/dev/null || echo "0")
            if [[ "$model_size" -lt 1000 ]]; then  # Models should be much larger than 1KB
                log "Warning: Model file seems too small: $model ($model_size bytes)"
                missing_models+=("$model")
            else
                log_verbose "✓ Model found: $model ($(ls -lh "$model_path" | awk '{print $5}'))"
            fi
        fi
    done
    
    if [[ ${#missing_models[@]} -gt 0 ]]; then
        log "Missing or invalid models detected:"
        for model in "${missing_models[@]}"; do
            log "  - $model"
        done
        return 1
    else
        log "✓ All required models are present and valid"
        return 0
    fi
}

# Function to download models
download_models() {
    log "Downloading ML models..."
    
    if [[ ! -x "$MODELS_DIR/download_models.sh" ]]; then
        log "✗ Model download script not found or not executable: $MODELS_DIR/download_models.sh"
        return 1
    fi
    
    cd "$MODELS_DIR"
    if ./download_models.sh; then
        log "✓ Models downloaded successfully"
        cd "$SCRIPT_DIR"
        return 0
    else
        log "✗ Model download failed"
        cd "$SCRIPT_DIR"
        return 1
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

# Function to build the project
build_project() {
    log "Building ML Face Service..."
    
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
    
    # Check if binary was created
    if [[ -f "$BUILD_DIR/MLFaceService" ]]; then
        log "✓ Build completed successfully"
        cd "$SCRIPT_DIR"
        return 0
    else
        log "✗ Build failed - binary not found"
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

# Function to start the service
start_service() {
    local binary_path="$BUILD_DIR/MLFaceService"
    
    log "Starting ML Face Service..."
    log "Port: $PORT"
    log "Models: $MODELS_DIR"
    log "Binary: $binary_path"
    log ""
    
    # Change to build directory to ensure relative paths work
    cd "$BUILD_DIR"
    
    # Start the service
    exec "$binary_path" --port "$PORT" --models "$MODELS_DIR"
}

# Function to handle cleanup on exit
cleanup() {
    log "Shutting down..."
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
        --skip-models)
            SKIP_MODELS=true
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
    log "Starting ML Face Service setup..."
    
    # Check models (unless skipped)
    if [[ "$SKIP_MODELS" != "true" ]]; then
        if ! check_models; then
            log "Attempting to download missing models..."
            if ! download_models; then
                log "✗ Failed to download models. Cannot proceed."
                log "Try running: ./models/download_models.sh"
                exit 1
            fi
        fi
    else
        log "Skipping model check (--skip-models specified)"
    fi
    
    # Check build dependencies
    if ! check_build_deps; then
        exit 1
    fi
    
    # Build if necessary
    if needs_rebuild; then
        if ! build_project; then
            log "✗ Build failed. Cannot start service."
            exit 1
        fi
    fi
    
    # Final validation
    local binary_path="$BUILD_DIR/MLFaceService"
    if [[ ! -f "$binary_path" ]]; then
        log "✗ Binary not found: $binary_path"
        exit 1
    fi
    
    if [[ ! -x "$binary_path" ]]; then
        log "✗ Binary is not executable: $binary_path"
        exit 1
    fi
    
    log "✓ All checks passed. Starting service..."
    log ""
    
    # Start the service
    start_service
}

# Run main function
main "$@"
