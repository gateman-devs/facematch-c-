#!/bin/bash

# Build script for optimized face service (OpenCV-only version)
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Building Optimized Face Service${NC}"
echo -e "${GREEN}(OpenCV-only version - no TensorFlow)${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${YELLOW}Detected macOS${NC}"
    PLATFORM="macos"
    NPROC=$(sysctl -n hw.ncpu)
else
    echo -e "${YELLOW}Detected Linux${NC}"
    PLATFORM="linux"
    NPROC=$(nproc)
fi

# Check for required dependencies
echo -e "\n${YELLOW}Checking dependencies...${NC}"

# Check OpenCV
if pkg-config --exists opencv4 2>/dev/null || pkg-config --exists opencv 2>/dev/null; then
    echo -e "${GREEN}✓ OpenCV found${NC}"
    if pkg-config --exists opencv4; then
        OPENCV_VERSION=$(pkg-config --modversion opencv4)
    else
        OPENCV_VERSION=$(pkg-config --modversion opencv)
    fi
    echo -e "  Version: ${OPENCV_VERSION}"
else
    echo -e "${RED}✗ OpenCV not found${NC}"
    echo -e "${YELLOW}Please install OpenCV:${NC}"
    if [[ "$PLATFORM" == "macos" ]]; then
        echo -e "  brew install opencv"
    else
        echo -e "  sudo apt-get install libopencv-dev"
    fi
    exit 1
fi

# Check CURL
if pkg-config --exists libcurl 2>/dev/null; then
    echo -e "${GREEN}✓ CURL found${NC}"
else
    echo -e "${RED}✗ CURL not found${NC}"
    echo -e "${YELLOW}Please install CURL:${NC}"
    if [[ "$PLATFORM" == "macos" ]]; then
        echo -e "  brew install curl"
    else
        echo -e "  sudo apt-get install libcurl4-openssl-dev"
    fi
    exit 1
fi

# Check CMake
if command -v cmake &> /dev/null; then
    echo -e "${GREEN}✓ CMake found${NC}"
    CMAKE_VERSION=$(cmake --version | head -n 1 | cut -d' ' -f3)
    echo -e "  Version: ${CMAKE_VERSION}"
else
    echo -e "${RED}✗ CMake not found${NC}"
    echo -e "${YELLOW}Please install CMake:${NC}"
    if [[ "$PLATFORM" == "macos" ]]; then
        echo -e "  brew install cmake"
    else
        echo -e "  sudo apt-get install cmake"
    fi
    exit 1
fi

# Create build directory
echo -e "\n${YELLOW}Creating build directory...${NC}"
mkdir -p build_simple
cd build_simple

# Download models if not present
echo -e "\n${YELLOW}Checking for face detection models...${NC}"
mkdir -p ../models

# Download OpenCV DNN model
if [ ! -f "../models/deploy.prototxt" ] || [ ! -f "../models/res10_300x300_ssd_iter_140000.caffemodel" ]; then
    echo -e "${YELLOW}Downloading OpenCV face detection model...${NC}"

    # Download prototxt
    curl -L -o ../models/deploy.prototxt \
        https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

    # Download caffemodel
    curl -L -o ../models/res10_300x300_ssd_iter_140000.caffemodel \
        https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

    echo -e "${GREEN}✓ OpenCV DNN model downloaded${NC}"
else
    echo -e "${GREEN}✓ OpenCV DNN model already present${NC}"
fi

# Download Haar cascade as fallback
if [ ! -f "../models/haarcascade_frontalface_default.xml" ]; then
    echo -e "${YELLOW}Downloading Haar cascade model...${NC}"

    curl -L -o ../models/haarcascade_frontalface_default.xml \
        https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

    echo -e "${GREEN}✓ Haar cascade model downloaded${NC}"
else
    echo -e "${GREEN}✓ Haar cascade model already present${NC}"
fi

# Configure CMake
echo -e "\n${YELLOW}Configuring CMake...${NC}"
if [[ "$PLATFORM" == "macos" ]]; then
    # macOS specific configuration
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS_RELEASE="-O3 -ffast-math -funroll-loops" \
          ..
else
    # Linux specific configuration
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -mtune=native -ffast-math -funroll-loops" \
          ..
fi

# Build
echo -e "\n${YELLOW}Building project (using ${NPROC} cores)...${NC}"
make -j${NPROC} optimized_server test_directions

# Check if build was successful
if [ -f "optimized_server" ] && [ -f "test_directions" ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo -e "${GREEN}========================================${NC}"

    echo -e "\n${BLUE}Executables created:${NC}"
    echo -e "  ${GREEN}optimized_server${NC}: Web server with BlazeFace (OpenCV)"
    echo -e "  ${GREEN}test_directions${NC}: Test program for direction detection"

    echo -e "\n${BLUE}Model files:${NC}"
    echo -e "  ${GREEN}../models/deploy.prototxt${NC}: DNN model architecture"
    echo -e "  ${GREEN}../models/res10_300x300_ssd_iter_140000.caffemodel${NC}: DNN model weights"
    echo -e "  ${GREEN}../models/haarcascade_frontalface_default.xml${NC}: Fallback cascade"

    echo -e "\n${YELLOW}To test the direction detection:${NC}"
    echo -e "  cd build_simple"
    echo -e "  ./test_directions"

    echo -e "\n${YELLOW}To start the web server:${NC}"
    echo -e "  cd build_simple"
    echo -e "  ./optimized_server --port 8080"

    echo -e "\n${YELLOW}To test with known videos:${NC}"
    echo -e "  curl -X POST http://localhost:8080/test-directions"

    # Ask to run tests
    echo -e "\n${YELLOW}Do you want to run the direction detection test now? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo -e "\n${YELLOW}Running direction detection test...${NC}"
        echo -e "${YELLOW}This will test with 4 videos from Cloudinary...${NC}\n"
        ./test_directions

        echo -e "\n${YELLOW}Do you want to start the web server now? (y/n)${NC}"
        read -r response2
        if [[ "$response2" =~ ^[Yy]$ ]]; then
            echo -e "\n${YELLOW}Starting web server on port 8080...${NC}"
            echo -e "${YELLOW}Press Ctrl+C to stop${NC}\n"
            ./optimized_server --port 8080
        fi
    fi
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}✗ Build failed!${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "Please check the error messages above."

    # Common troubleshooting tips
    echo -e "\n${YELLOW}Troubleshooting tips:${NC}"
    echo -e "1. Make sure OpenCV is installed with DNN module:"
    if [[ "$PLATFORM" == "macos" ]]; then
        echo -e "   ${BLUE}brew reinstall opencv${NC}"
    else
        echo -e "   ${BLUE}sudo apt-get install libopencv-dev${NC}"
    fi
    echo -e "2. Check that all dependencies are installed"
    echo -e "3. Try cleaning the build directory:"
    echo -e "   ${BLUE}rm -rf build_simple && ./build_simple.sh${NC}"

    exit 1
fi
