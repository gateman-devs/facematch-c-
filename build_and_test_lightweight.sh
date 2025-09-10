#!/bin/bash

# Build and test lightweight video detector
set -e

echo "========================================="
echo "Building Lightweight Video Detector"
echo "========================================="

# Create build directory
mkdir -p build_lightweight
cd build_lightweight

# Create simplified test program
cat > ../test_lightweight_simple.cpp << 'EOF'
#include "src/lightweight/lightweight_video_detector_simple.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace lightweight;

std::string directionToString(Direction dir) {
    switch (dir) {
        case Direction::LEFT: return "LEFT";
        case Direction::RIGHT: return "RIGHT";
        case Direction::UP: return "UP";
        case Direction::DOWN: return "DOWN";
        default: return "NONE";
    }
}

int main() {
    std::cout << "\n==== Lightweight Video Detector Test ====\n" << std::endl;

    // Initialize detector
    LightweightVideoDetector detector;
    detector.setFrameSampleRate(3);
    detector.setMinMovementThreshold(0.08f);
    detector.setMaxFramesToAnalyze(15);

    if (!detector.initialize("./models")) {
        std::cout << "Warning: Could not fully initialize (will use fallback methods)\n" << std::endl;
    }

    // Test videos
    struct Test {
        std::string url;
        std::string expected;
    };

    std::vector<Test> tests = {
        {"https://res.cloudinary.com/themizehq/video/upload/v1755978338/IMG_6552.mov", "DOWN"},
        {"https://res.cloudinary.com/themizehq/video/upload/v1755978338/IMG_6551.mov", "UP"},
        {"https://res.cloudinary.com/themizehq/video/upload/v1755978331/IMG_6553.mov", "LEFT"},
        {"https://res.cloudinary.com/themizehq/video/upload/v1755978327/IMG_6554.mov", "RIGHT"}
    };

    int correct = 0;
    auto start = std::chrono::steady_clock::now();

    for (const auto& test : tests) {
        std::cout << "Testing " << test.expected << " movement..." << std::endl;

        auto analysis = detector.analyzeVideo(test.url);
        std::string detected = directionToString(analysis.primary_direction);

        bool match = (detected == test.expected);
        if (match) correct++;

        std::cout << "  Expected: " << test.expected
                  << ", Detected: " << detected
                  << " [" << (match ? "✓" : "✗") << "]"
                  << " (Time: " << std::fixed << std::setprecision(1)
                  << analysis.total_processing_time_ms << "ms)" << std::endl;
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<float>(end - start).count();

    std::cout << "\n==== Results ====" << std::endl;
    std::cout << "Accuracy: " << correct << "/" << tests.size()
              << " (" << (100.0f * correct / tests.size()) << "%)" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2)
              << duration << "s" << std::endl;
    std::cout << "Avg per video: " << (duration / tests.size()) << "s" << std::endl;

    // Test concurrent processing
    std::cout << "\n==== Testing Concurrent Processing ====" << std::endl;
    std::vector<std::string> urls;
    for (const auto& test : tests) {
        urls.push_back(test.url);
    }

    auto c_start = std::chrono::steady_clock::now();
    auto results = detector.analyzeVideos(urls);
    auto c_end = std::chrono::steady_clock::now();
    auto c_duration = std::chrono::duration<float>(c_end - c_start).count();

    std::cout << "Concurrent time: " << c_duration << "s" << std::endl;
    std::cout << "Speedup: " << (duration / c_duration) << "x" << std::endl;

    // Resource usage estimate
    std::cout << "\n==== Resource Usage ====" << std::endl;
    std::cout << "Model size: ~0 MB (using OpenCV only)" << std::endl;
    std::cout << "Avg frame processing: " << (duration * 1000 / (tests.size() * 15)) << "ms" << std::endl;
    std::cout << "Est. CPU usage: Minimal (compared to original)" << std::endl;

    return (correct == tests.size()) ? 0 : 1;
}
EOF

# Configure with CMake (simplified version without TensorFlow Lite)
cat > ../CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.16)
project(LightweightVideoTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# Find packages
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio video)
find_package(CURL REQUIRED)
find_package(Threads REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})

# Test executable
add_executable(test_lightweight
    test_lightweight_simple.cpp
    src/lightweight/lightweight_video_detector_simple.cpp
)

# Link libraries
target_link_libraries(test_lightweight
    ${OpenCV_LIBS}
    ${CURL_LIBRARIES}
    Threads::Threads
)
EOF

# Build
echo "Configuring CMake..."
cmake -DCMAKE_BUILD_TYPE=Release ..

echo "Building..."
make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# Create models directory
mkdir -p models

echo ""
echo "========================================="
echo "Running Tests"
echo "========================================="
echo ""

# Run the test
./test_lightweight

echo ""
echo "========================================="
echo "Build and test complete!"
echo "========================================="
