#include "src/blazeface/blazeface_opencv.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace blazeface;

// ANSI color codes for terminal output
const std::string RESET = "\033[0m";
const std::string GREEN = "\033[32m";
const std::string RED = "\033[31m";
const std::string YELLOW = "\033[33m";
const std::string BLUE = "\033[34m";
const std::string BOLD = "\033[1m";

void printSeparator() {
    std::cout << "========================================" << std::endl;
}

void printTestHeader() {
    printSeparator();
    std::cout << BOLD << "Direction Detection Test Program" << RESET << std::endl;
    std::cout << "Testing BlazeFace-based video analysis" << std::endl;
    printSeparator();
}

void printTestResult(const std::string& url, const std::string& expected, 
                     const VideoAnalysis& analysis) {
    bool passed = (analysis.primary_direction == expected);
    std::string status_color = passed ? GREEN : RED;
    std::string status_symbol = passed ? "✓" : "✗";
    
    std::cout << "\nVideo Test:" << std::endl;
    std::cout << "  URL: " << BLUE << url.substr(0, 60) << "..." << RESET << std::endl;
    std::cout << "  Expected: " << YELLOW << expected << RESET << std::endl;
    std::cout << "  Detected: " << YELLOW << analysis.primary_direction << RESET << std::endl;
    std::cout << "  Status: " << status_color << status_symbol << " " 
              << (passed ? "PASSED" : "FAILED") << RESET << std::endl;
    std::cout << "  Confidence: " << std::fixed << std::setprecision(2) 
              << analysis.confidence << std::endl;
    std::cout << "  Face Present: " << (analysis.has_face ? "Yes" : "No") << std::endl;
    std::cout << "  Face Ratio: " << std::fixed << std::setprecision(2) 
              << analysis.face_presence_ratio << std::endl;
    std::cout << "  Frames: " << analysis.frames_with_face << "/" 
              << analysis.total_frames << std::endl;
    
    if (analysis.is_live) {
        std::cout << "  Liveness Score: " << std::fixed << std::setprecision(2) 
                  << analysis.liveness_score << std::endl;
    }
}

void testSingleVideo(BlazeFaceDetector& detector, const std::string& url, 
                    const std::string& expected_direction) {
    std::cout << "\n" << BOLD << "Testing " << expected_direction 
              << " movement..." << RESET << std::endl;
    
    auto start = std::chrono::steady_clock::now();
    VideoAnalysis analysis = detector.analyzeVideo(url);
    auto end = std::chrono::steady_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printTestResult(url, expected_direction, analysis);
    std::cout << "  Processing Time: " << duration.count() << " ms" << std::endl;
}

void testConcurrentProcessing(BlazeFaceDetector& detector) {
    printSeparator();
    std::cout << BOLD << "Concurrent Processing Test" << RESET << std::endl;
    std::cout << "Processing 4 videos simultaneously" << std::endl;
    std::cout << "Liveness check on 1 random video only" << std::endl;
    printSeparator();
    
    std::vector<std::string> video_urls = {
        "https://res.cloudinary.com/themizehq/video/upload/v1755978338/IMG_6552.mov",
        "https://res.cloudinary.com/themizehq/video/upload/v1755978338/IMG_6551.mov",
        "https://res.cloudinary.com/themizehq/video/upload/v1755978331/IMG_6553.mov",
        "https://res.cloudinary.com/themizehq/video/upload/v1755978327/IMG_6554.mov"
    };
    
    std::vector<std::string> expected_directions = {"DOWN", "UP", "LEFT", "RIGHT"};
    
    auto start = std::chrono::steady_clock::now();
    auto results = detector.analyzeVideos(video_urls, -1); // -1 for random liveness check
    auto end = std::chrono::steady_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\nResults:" << std::endl;
    int passed = 0;
    int liveness_checked_index = -1;
    
    for (size_t i = 0; i < results.size() && i < expected_directions.size(); ++i) {
        bool matches = (results[i].primary_direction == expected_directions[i]);
        if (matches) passed++;
        
        std::string status_color = matches ? GREEN : RED;
        std::string status_symbol = matches ? "✓" : "✗";
        
        std::cout << "  Video " << (i + 1) << ": ";
        std::cout << "Expected=" << YELLOW << std::setw(5) << expected_directions[i] << RESET;
        std::cout << ", Detected=" << YELLOW << std::setw(5) << results[i].primary_direction << RESET;
        std::cout << " " << status_color << status_symbol << RESET;
        std::cout << " (Conf: " << std::fixed << std::setprecision(2) << results[i].confidence << ")";
        
        // Check if this video had liveness check
        if (results[i].liveness_score < 1.0f) {
            liveness_checked_index = i;
            std::cout << " " << BLUE << "[Liveness: " << results[i].liveness_score << "]" << RESET;
        }
        
        std::cout << std::endl;
    }
    
    std::cout << "\n" << BOLD << "Summary:" << RESET << std::endl;
    std::cout << "  Total Videos: " << results.size() << std::endl;
    std::cout << "  Passed: " << GREEN << passed << "/" << results.size() << RESET << std::endl;
    std::cout << "  Liveness Checked: Video " << (liveness_checked_index + 1) << std::endl;
    std::cout << "  Total Time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Avg Time/Video: " << (duration.count() / results.size()) << " ms" << std::endl;
    
    if (passed == static_cast<int>(results.size())) {
        std::cout << "\n" << GREEN << BOLD << "ALL TESTS PASSED!" << RESET << std::endl;
    } else {
        std::cout << "\n" << RED << BOLD << "SOME TESTS FAILED!" << RESET << std::endl;
    }
}

void testPerformance(BlazeFaceDetector& detector) {
    printSeparator();
    std::cout << BOLD << "Performance Benchmark" << RESET << std::endl;
    printSeparator();
    
    std::string test_url = "https://res.cloudinary.com/themizehq/video/upload/v1755978338/IMG_6552.mov";
    
    std::cout << "Running 3 iterations for benchmark..." << std::endl;
    
    std::vector<long> times;
    for (int i = 0; i < 3; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto analysis = detector.analyzeVideo(test_url);
        auto end = std::chrono::steady_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        times.push_back(duration.count());
        
        std::cout << "  Iteration " << (i + 1) << ": " << duration.count() << " ms" << std::endl;
    }
    
    long total = 0;
    for (long t : times) total += t;
    long avg = total / times.size();
    
    std::cout << "\nAverage Processing Time: " << BOLD << avg << " ms" << RESET << std::endl;
}

int main(int argc, char* argv[]) {
    printTestHeader();
    
    // Initialize detector
    std::cout << "\nInitializing BlazeFace detector..." << std::endl;
    BlazeFaceDetector detector;
    
    std::string model_path = "./models/res10_300x300_ssd_iter_140000.caffemodel";
    if (!detector.initialize(model_path)) {
        std::cerr << RED << "Failed to initialize detector!" << RESET << std::endl;
        return 1;
    }
    
    // Configure detector
    detector.setConfidenceThreshold(0.5f);
    detector.setNMSThreshold(0.3f);
    
    std::cout << GREEN << "✓ Detector initialized successfully" << RESET << std::endl;
    
    // Parse command line arguments
    bool run_individual = false;
    bool run_concurrent = true;
    bool run_benchmark = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--individual" || arg == "-i") {
            run_individual = true;
        } else if (arg == "--benchmark" || arg == "-b") {
            run_benchmark = true;
        } else if (arg == "--all" || arg == "-a") {
            run_individual = true;
            run_concurrent = true;
            run_benchmark = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "\nUsage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -i, --individual  Run individual video tests" << std::endl;
            std::cout << "  -b, --benchmark   Run performance benchmark" << std::endl;
            std::cout << "  -a, --all         Run all tests" << std::endl;
            std::cout << "  -h, --help        Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Test data
    std::vector<std::pair<std::string, std::string>> test_videos = {
        {"https://res.cloudinary.com/themizehq/video/upload/v1755978338/IMG_6552.mov", "DOWN"},
        {"https://res.cloudinary.com/themizehq/video/upload/v1755978338/IMG_6551.mov", "UP"},
        {"https://res.cloudinary.com/themizehq/video/upload/v1755978331/IMG_6553.mov", "LEFT"},
        {"https://res.cloudinary.com/themizehq/video/upload/v1755978327/IMG_6554.mov", "RIGHT"}
    };
    
    // Run individual tests
    if (run_individual) {
        printSeparator();
        std::cout << BOLD << "Individual Video Tests" << RESET << std::endl;
        printSeparator();
        
        for (const auto& [url, direction] : test_videos) {
            testSingleVideo(detector, url, direction);
        }
    }
    
    // Run concurrent processing test
    if (run_concurrent) {
        testConcurrentProcessing(detector);
    }
    
    // Run performance benchmark
    if (run_benchmark) {
        testPerformance(detector);
    }
    
    printSeparator();
    std::cout << BOLD << "Test Complete!" << RESET << std::endl;
    printSeparator();
    
    return 0;
}