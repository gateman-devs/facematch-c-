#include "src/lightweight/lightweight_video_detector.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>

using namespace lightweight;

// Helper function to convert Direction enum to string
std::string directionToString(Direction dir) {
    switch (dir) {
        case Direction::LEFT: return "LEFT";
        case Direction::RIGHT: return "RIGHT";
        case Direction::UP: return "UP";
        case Direction::DOWN: return "DOWN";
        case Direction::NONE: return "NONE";
        default: return "UNKNOWN";
    }
}

// Helper function to print analysis results
void printAnalysis(const std::string& expected, const VideoAnalysis& analysis) {
    std::cout << "\n=== Video Analysis Result ===" << std::endl;
    std::cout << "Expected Direction: " << expected << std::endl;
    std::cout << "Detected Direction: " << directionToString(analysis.primary_direction) << std::endl;
    std::cout << "Success: " << (analysis.success ? "YES" : "NO") << std::endl;
    std::cout << "Is Live: " << (analysis.is_live ? "YES" : "NO") << std::endl;
    std::cout << "Frames Analyzed: " << analysis.total_frames_analyzed << std::endl;
    std::cout << "Frames with Face: " << analysis.frames_with_face << std::endl;
    std::cout << "Average Spoof Score: " << std::fixed << std::setprecision(2) << analysis.average_spoof_score << std::endl;
    std::cout << "Total Movements Detected: " << analysis.movements.size() << std::endl;
    std::cout << "Processing Time: " << std::fixed << std::setprecision(1) << analysis.total_processing_time_ms << " ms" << std::endl;
    std::cout << "Avg Frame Time: " << std::fixed << std::setprecision(2) << analysis.avg_frame_time_ms << " ms" << std::endl;
    
    if (!analysis.error_message.empty()) {
        std::cout << "Error: " << analysis.error_message << std::endl;
    }
    
    // Print movement breakdown
    if (!analysis.movements.empty()) {
        int left = 0, right = 0, up = 0, down = 0;
        for (const auto& movement : analysis.movements) {
            switch (movement.direction) {
                case Direction::LEFT: left++; break;
                case Direction::RIGHT: right++; break;
                case Direction::UP: up++; break;
                case Direction::DOWN: down++; break;
                default: break;
            }
        }
        std::cout << "Movement Breakdown: "
                  << "Left=" << left << ", "
                  << "Right=" << right << ", "
                  << "Up=" << up << ", "
                  << "Down=" << down << std::endl;
    }
    
    // Check if detection matches expectation
    bool matches = false;
    if (expected == "DOWN" && analysis.primary_direction == Direction::DOWN) matches = true;
    else if (expected == "UP" && analysis.primary_direction == Direction::UP) matches = true;
    else if (expected == "LEFT" && analysis.primary_direction == Direction::LEFT) matches = true;
    else if (expected == "RIGHT" && analysis.primary_direction == Direction::RIGHT) matches = true;
    
    std::cout << "RESULT: " << (matches ? "✓ CORRECT" : "✗ INCORRECT") << std::endl;
    std::cout << "=============================" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "Lightweight Video Detector Test" << std::endl;
    std::cout << "================================\n" << std::endl;
    
    // Initialize detector
    LightweightVideoDetector detector;
    
    // Configure for optimal performance
    detector.setFrameSampleRate(3);        // Process every 3rd frame
    detector.setMinMovementThreshold(0.10f); // 10% movement threshold
    detector.setSpoofThreshold(0.4f);      // Lower threshold since we might not have anti-spoof model
    detector.setMaxFramesToAnalyze(20);    // Limit to 20 frames max
    
    // Create models directory if it doesn't exist
    system("mkdir -p ./models");
    
    std::cout << "Initializing detector..." << std::endl;
    if (!detector.initialize("./models")) {
        std::cerr << "Failed to initialize detector. Continuing anyway..." << std::endl;
        // Continue anyway as we have fallback implementations
    }
    
    // Test videos with expected directions
    struct TestCase {
        std::string url;
        std::string expected_direction;
    };
    
    std::vector<TestCase> test_cases = {
        {"https://res.cloudinary.com/themizehq/video/upload/v1755978338/IMG_6552.mov", "DOWN"},
        {"https://res.cloudinary.com/themizehq/video/upload/v1755978338/IMG_6551.mov", "UP"},
        {"https://res.cloudinary.com/themizehq/video/upload/v1755978331/IMG_6553.mov", "LEFT"},
        {"https://res.cloudinary.com/themizehq/video/upload/v1755978327/IMG_6554.mov", "RIGHT"}
    };
    
    std::cout << "\nProcessing " << test_cases.size() << " test videos...\n" << std::endl;
    
    // Track overall performance
    auto overall_start = std::chrono::steady_clock::now();
    int correct_detections = 0;
    
    // Process videos sequentially for clear output
    for (size_t i = 0; i < test_cases.size(); i++) {
        std::cout << "Processing video " << (i + 1) << "/" << test_cases.size() 
                  << " (" << test_cases[i].expected_direction << ")..." << std::endl;
        
        VideoAnalysis analysis = detector.analyzeVideo(test_cases[i].url);
        printAnalysis(test_cases[i].expected_direction, analysis);
        
        // Check if correct
        bool correct = false;
        if (test_cases[i].expected_direction == "DOWN" && analysis.primary_direction == Direction::DOWN) correct = true;
        else if (test_cases[i].expected_direction == "UP" && analysis.primary_direction == Direction::UP) correct = true;
        else if (test_cases[i].expected_direction == "LEFT" && analysis.primary_direction == Direction::LEFT) correct = true;
        else if (test_cases[i].expected_direction == "RIGHT" && analysis.primary_direction == Direction::RIGHT) correct = true;
        
        if (correct) correct_detections++;
    }
    
    auto overall_end = std::chrono::steady_clock::now();
    auto overall_duration = std::chrono::duration<float>(overall_end - overall_start).count();
    
    // Print summary
    std::cout << "\n========== SUMMARY ==========" << std::endl;
    std::cout << "Total Videos: " << test_cases.size() << std::endl;
    std::cout << "Correct Detections: " << correct_detections << "/" << test_cases.size() << std::endl;
    std::cout << "Accuracy: " << std::fixed << std::setprecision(1) 
              << (100.0f * correct_detections / test_cases.size()) << "%" << std::endl;
    std::cout << "Total Processing Time: " << std::fixed << std::setprecision(2) 
              << overall_duration << " seconds" << std::endl;
    std::cout << "Average Time per Video: " << std::fixed << std::setprecision(2) 
              << (overall_duration / test_cases.size()) << " seconds" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Test concurrent processing
    std::cout << "\n========== CONCURRENT PROCESSING TEST ==========" << std::endl;
    std::cout << "Processing all videos concurrently..." << std::endl;
    
    std::vector<std::string> urls;
    for (const auto& test : test_cases) {
        urls.push_back(test.url);
    }
    
    auto concurrent_start = std::chrono::steady_clock::now();
    std::vector<VideoAnalysis> concurrent_results = detector.analyzeVideos(urls);
    auto concurrent_end = std::chrono::steady_clock::now();
    auto concurrent_duration = std::chrono::duration<float>(concurrent_end - concurrent_start).count();
    
    std::cout << "Concurrent processing completed in " << std::fixed << std::setprecision(2) 
              << concurrent_duration << " seconds" << std::endl;
    std::cout << "Speed improvement: " << std::fixed << std::setprecision(1) 
              << (overall_duration / concurrent_duration) << "x faster" << std::endl;
    
    // Verify concurrent results
    std::cout << "\nConcurrent Results:" << std::endl;
    for (size_t i = 0; i < concurrent_results.size(); i++) {
        std::cout << "  Video " << (i + 1) << ": " 
                  << directionToString(concurrent_results[i].primary_direction)
                  << " (Expected: " << test_cases[i].expected_direction << ")" << std::endl;
    }
    
    std::cout << "================================================\n" << std::endl;
    
    return (correct_detections == test_cases.size()) ? 0 : 1;
}