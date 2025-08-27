#!/bin/bash

# ML Face Service - Model Download Script
# Downloads required dlib models and sets up MediaPipe dependencies

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

echo "=== ML Face Service - Model Download ==="
echo "Script directory: $SCRIPT_DIR"
echo "Models directory: $MODELS_DIR"
echo "========================================"

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to download file with checksum verification
download_with_checksum() {
    local url="$1"
    local filename="$2"
    local expected_sha256="$3"
    
    echo "Downloading $filename..."
    
    # Skip if file already exists and has correct checksum
    if [[ -f "$filename" ]]; then
        if command_exists sha256sum; then
            local current_checksum=$(sha256sum "$filename" | cut -d' ' -f1)
        elif command_exists shasum; then
            local current_checksum=$(shasum -a 256 "$filename" | cut -d' ' -f1)
        else
            echo "Warning: No checksum utility available, skipping verification"
            return 0
        fi
        
        if [[ "$current_checksum" == "$expected_sha256" ]]; then
            echo "✓ $filename already exists with correct checksum"
            return 0
        else
            echo "✗ $filename exists but checksum doesn't match, re-downloading..."
            rm -f "$filename"
        fi
    fi
    
    # Download the file
    if command_exists curl; then
        curl -L -o "$filename" "$url"
    elif command_exists wget; then
        wget -O "$filename" "$url"
    else
        echo "✗ Error: Neither curl nor wget is available for downloading"
        return 1
    fi
    
    # Verify checksum
    if command_exists sha256sum; then
        local downloaded_checksum=$(sha256sum "$filename" | cut -d' ' -f1)
    elif command_exists shasum; then
        local downloaded_checksum=$(shasum -a 256 "$filename" | cut -d' ' -f1)
    else
        echo "Warning: No checksum utility available, skipping verification"
        return 0
    fi
    
    if [[ "$downloaded_checksum" == "$expected_sha256" ]]; then
        echo "✓ $filename downloaded and verified successfully"
        return 0
    else
        echo "✗ Error: Checksum verification failed for $filename"
        echo "Expected: $expected_sha256"
        echo "Got:      $downloaded_checksum"
        rm -f "$filename"
        return 1
    fi
}

# Download dlib face recognition model
echo "Downloading dlib face recognition model..."
download_with_checksum \
    "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2" \
    "dlib_face_recognition_resnet_model_v1.dat.bz2" \
    "abb1f61041e434465855ce81c2bd546e830d28bcbed8d27ffbe5bb408b11553a"

# Extract if needed
if [[ -f "dlib_face_recognition_resnet_model_v1.dat.bz2" && ! -f "dlib_face_recognition_resnet_model_v1.dat" ]]; then
    echo "Extracting face recognition model..."
    if command_exists bzip2; then
        bzip2 -d "dlib_face_recognition_resnet_model_v1.dat.bz2"
    else
        echo "✗ Error: bzip2 is required to extract the model"
        return 1
    fi
fi

# Download dlib facial landmarks model
echo "Downloading dlib facial landmarks model..."
download_with_checksum \
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" \
    "shape_predictor_68_face_landmarks.dat.bz2" \
    "7d6637b8f34ddb0c1363e09a4628acb34314019ec3566fd66b80c04dda6980f5"

# Extract if needed
if [[ -f "shape_predictor_68_face_landmarks.dat.bz2" && ! -f "shape_predictor_68_face_landmarks.dat" ]]; then
    echo "Extracting facial landmarks model..."
    if command_exists bzip2; then
        bzip2 -d "shape_predictor_68_face_landmarks.dat.bz2"
    else
        echo "✗ Error: bzip2 is required to extract the model"
        return 1
    fi
fi

# Set up MediaPipe models directory
echo "Setting up MediaPipe models..."
mkdir -p mediapipe

# Check if MediaPipe models already exist
mediapipe_models_exist=true
if [[ ! -f "mediapipe/blaze_face_short_range.tflite" || ! -f "mediapipe/face_landmarker.task" ]]; then
    mediapipe_models_exist=false
fi

if [[ "$mediapipe_models_exist" == "false" ]]; then
    echo "MediaPipe models not found, attempting to download..."
    
    # Download MediaPipe BlazeFace model
    echo "Downloading MediaPipe BlazeFace model..."
    if download_with_checksum \
        "https://storage.googleapis.com/mediapipe-models/face_detection/blaze_face_short_range/float16/1/blaze_face_short_range.tflite" \
        "mediapipe/blaze_face_short_range.tflite" \
        ""; then  # Checksum verification skipped for MediaPipe models
        echo "✓ MediaPipe BlazeFace model downloaded"
    else
        echo "⚠ Warning: Failed to download MediaPipe BlazeFace model"
    fi
    
    # Download MediaPipe Face Landmarker model
    echo "Downloading MediaPipe Face Landmarker model..."
    if download_with_checksum \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
        "mediapipe/face_landmarker.task" \
        ""; then  # Checksum verification skipped for MediaPipe models
        echo "✓ MediaPipe Face Landmarker model downloaded"
    else
        echo "⚠ Warning: Failed to download MediaPipe Face Landmarker model"
    fi
else
    echo "✓ MediaPipe models already exist"
fi

# Verify all required models are present
echo ""
echo "Verifying downloaded models..."

required_models=(
    "dlib_face_recognition_resnet_model_v1.dat"
    "shape_predictor_68_face_landmarks.dat"
)

missing_models=()
for model in "${required_models[@]}"; do
    if [[ -f "$model" ]]; then
        size=$(ls -lh "$model" | awk '{print $5}')
        echo "✓ $model ($size)"
    else
        echo "✗ $model (missing)"
        missing_models+=("$model")
    fi
done

# Check MediaPipe models (optional)
mediapipe_models=(
    "mediapipe/blaze_face_short_range.tflite"
    "mediapipe/face_landmarker.task"
)

for model in "${mediapipe_models[@]}"; do
    if [[ -f "$model" ]]; then
        size=$(ls -lh "$model" | awk '{print $5}')
        echo "✓ $model ($size) [optional]"
    else
        echo "⚠ $model (missing) [optional - video liveness may not work]"
    fi
done

if [[ ${#missing_models[@]} -eq 0 ]]; then
    echo ""
    echo "✓ All required models downloaded successfully!"
    echo "✓ Models are ready for use"
    exit 0
else
    echo ""
    echo "✗ Some required models are missing:"
    for model in "${missing_models[@]}"; do
        echo "  - $model"
    done
    exit 1
fi