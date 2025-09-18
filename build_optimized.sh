#!/bin/bash

# Build script for optimized face service
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Building Optimized Face Service${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${YELLOW}Detected macOS${NC}"
    PLATFORM="macos"
else
    echo -e "${YELLOW}Detected Linux${NC}"
    PLATFORM="linux"
fi

# Create build directory
echo -e "\n${YELLOW}Creating build directory...${NC}"
mkdir -p build_optimized
cd build_optimized

# Install TensorFlow Lite if not present
echo -e "\n${YELLOW}Checking for TensorFlow Lite...${NC}"
if [[ "$PLATFORM" == "macos" ]]; then
    # Check if TensorFlow Lite is installed via Homebrew
    if ! brew list tensorflow-lite &>/dev/null; then
        echo -e "${YELLOW}Installing TensorFlow Lite via Homebrew...${NC}"
        brew install tensorflow-lite
    else
        echo -e "${GREEN}TensorFlow Lite already installed${NC}"
    fi
else
    # For Linux, check if TensorFlow Lite is available
    if [ ! -f "/usr/local/lib/libtensorflowlite.so" ]; then
        echo -e "${YELLOW}TensorFlow Lite not found. Building from source...${NC}"

        # Create temp directory for building TFLite
        TFLITE_BUILD_DIR="/tmp/tflite_build_$$"
        mkdir -p $TFLITE_BUILD_DIR
        cd $TFLITE_BUILD_DIR

        # Clone TensorFlow repository
        git clone --depth 1 --branch v2.14.0 https://github.com/tensorflow/tensorflow.git
        cd tensorflow

        # Build TensorFlow Lite
        mkdir -p build
        cd build
        cmake ../tensorflow/lite -DCMAKE_BUILD_TYPE=Release
        make -j$(nproc)

        # Install TensorFlow Lite
        sudo make install

        # Clean up
        cd /
        rm -rf $TFLITE_BUILD_DIR

        # Return to build directory
        cd -
    else
        echo -e "${GREEN}TensorFlow Lite found${NC}"
    fi
fi

# Download BlazeFace model if not present
echo -e "\n${YELLOW}Checking for BlazeFace model...${NC}"
if [ ! -f "../models/blazeface.tflite" ]; then
    echo -e "${YELLOW}Downloading BlazeFace model...${NC}"
    mkdir -p ../models
    curl -L -o ../models/blazeface.tflite \
        https://storage.googleapis.com/mediapipe-models/face_detection/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
    echo -e "${GREEN}Model downloaded successfully${NC}"
else
    echo -e "${GREEN}BlazeFace model already present${NC}"
fi

# Configure CMake
echo -e "\n${YELLOW}Configuring CMake...${NC}"
if [[ "$PLATFORM" == "macos" ]]; then
    # macOS specific configuration
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -mtune=native -ffast-math -funroll-loops" \
          -DTENSORFLOW_SOURCE_DIR=/opt/homebrew/include \
          -DTENSORFLOW_LITE_LIB=/opt/homebrew/lib/libtensorflow-lite.dylib \
          ../CMakeLists_optimized.txt
else
    # Linux specific configuration
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -mtune=native -ffast-math -funroll-loops -flto" \
          ../CMakeLists_optimized.txt
fi

# Build
echo -e "\n${YELLOW}Building project...${NC}"
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu) optimized_server test_directions

# Check if build was successful
if [ -f "optimized_server" ] && [ -f "test_directions" ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Build successful!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "\nExecutables created:"
    echo -e "  - ${GREEN}optimized_server${NC}: Web server with optimized BlazeFace"
    echo -e "  - ${GREEN}test_directions${NC}: Test program for direction detection"

    echo -e "\n${YELLOW}To test the direction detection:${NC}"
    echo -e "  ./test_directions"

    echo -e "\n${YELLOW}To start the web server:${NC}"
    echo -e "  ./optimized_server --port 8080"

    echo -e "\n${YELLOW}To test with Docker:${NC}"
    echo -e "  docker build -f ../Dockerfile.optimized -t gateman-face-optimized .."
    echo -e "  docker run -p 8080:8080 gateman-face-optimized"
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}Build failed!${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "Please check the error messages above."
    exit 1
fi

# Optionally run tests
echo -e "\n${YELLOW}Do you want to run the direction detection test now? (y/n)${NC}"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo -e "\n${YELLOW}Running direction detection test...${NC}"
    ./test_directions
fi
