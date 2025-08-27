#include "challenge_verifier.hpp"
#include <iostream>
#include <thread>
#include <algorithm>

ChallengeVerifier::ChallengeVerifier() : initialized_(false) {
}

bool ChallengeVerifier::initialize(std::shared_ptr<VideoLivenessDetector> detector) {
    if (!detector || !detector->isInitialized()) {
        std::cerr << "Invalid or uninitialized video liveness detector" << std::endl;
        return false;
    }
    
    video_detector_ = detector;
    initialized_ = true;
    
    std::cout << "Challenge verifier initialized successfully" << std::endl;
    return true;
}

ChallengeVerificationResult ChallengeVerifier::verifyChallenge(
    const Challenge& challenge, 
    const std::vector<std::string>& video_urls) {
    
    ChallengeVerificationResult result;
    
    if (!initialized_) {
        result.error_message = "Challenge verifier not initialized";
        return result;
    }
    
    // Validate input
    if (challenge.directions.size() != 4) {
        result.error_message = "Challenge must have exactly 4 directions";
        return result;
    }
    
    if (video_urls.size() != 4) {
        result.error_message = "Must provide exactly 4 video URLs";
        return result;
    }
    
    // Check if challenge has expired
    if (challenge.isExpired()) {
        result.error_message = "Challenge has expired";
        return result;
    }
    
    // Store expected directions
    result.expected_directions = challenge.directions;
    
    std::cout << "Verifying challenge " << challenge.id << " with " << video_urls.size() << " videos" << std::endl;
    std::cout << "Expected directions: ";
    for (size_t i = 0; i < challenge.directions.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << Challenge::directionToString(challenge.directions[i]);
    }
    std::cout << std::endl;
    
    // Analyze all videos concurrently
    result.video_results = analyzeVideosConcurrently(video_urls);
    
    // Extract detected directions and check for success
    bool all_videos_analyzed = true;
    result.detected_directions.reserve(4);
    
    for (const auto& video_result : result.video_results) {
        result.detected_directions.push_back(video_result.detected_direction);
        if (!video_result.success) {
            all_videos_analyzed = false;
        }
    }
    
    if (!all_videos_analyzed) {
        result.error_message = "Failed to analyze one or more videos";
        std::cout << "Challenge verification failed: " << result.error_message << std::endl;
        return result;
    }
    
    // Compare expected vs detected directions
    bool all_directions_match = true;
    std::cout << "Direction comparison:" << std::endl;
    
    for (size_t i = 0; i < 4; ++i) {
        ChallengeDirection expected = challenge.directions[i];
        MovementDirection detected = result.detected_directions[i];
        bool matches = directionsMatch(expected, detected);
        
        std::cout << "Video " << i << ": Expected " 
                  << Challenge::directionToString(expected) 
                  << ", Detected " << (detected == MovementDirection::NONE ? "none" : 
                     (detected == MovementDirection::LEFT ? "left" :
                      detected == MovementDirection::RIGHT ? "right" :
                      detected == MovementDirection::UP ? "up" :
                      detected == MovementDirection::DOWN ? "down" : "unknown"))
                  << " - " << (matches ? "MATCH" : "MISMATCH") << std::endl;
        
        if (!matches) {
            all_directions_match = false;
        }
    }
    
    result.passed = all_directions_match;
    
    if (result.passed) {
        std::cout << "Challenge verification PASSED - all directions match!" << std::endl;
    } else {
        std::cout << "Challenge verification FAILED - direction mismatch detected" << std::endl;
        result.error_message = "One or more video directions do not match the challenge";
    }
    
    return result;
}

std::vector<VideoAnalysisResult> ChallengeVerifier::analyzeVideosConcurrently(
    const std::vector<std::string>& video_urls) {
    
    std::vector<std::future<VideoAnalysisResult>> futures;
    futures.reserve(video_urls.size());
    
    std::cout << "Starting concurrent analysis of " << video_urls.size() << " videos..." << std::endl;
    
    // Launch async tasks for each video
    for (size_t i = 0; i < video_urls.size(); ++i) {
        futures.emplace_back(std::async(std::launch::async, [this, video_url = video_urls[i], index = i]() {
            std::cout << "Starting analysis of video " << index << ": " << video_url << std::endl;
            auto start_time = std::chrono::steady_clock::now();
            
            VideoAnalysisResult result = analyzeSingleVideo(video_url);
            result.video_url = video_url;
            
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            std::cout << "Completed analysis of video " << index << " in " 
                      << duration.count() << "ms - Result: " 
                      << (result.success ? "SUCCESS" : "FAILED")
                      << ", Direction: " << (result.detected_direction == MovementDirection::NONE ? "none" :
                         (result.detected_direction == MovementDirection::LEFT ? "left" :
                          result.detected_direction == MovementDirection::RIGHT ? "right" :
                          result.detected_direction == MovementDirection::UP ? "up" :
                          result.detected_direction == MovementDirection::DOWN ? "down" : "unknown"))
                      << std::endl;
            
            return result;
        }));
    }
    
    // Collect results
    std::vector<VideoAnalysisResult> results;
    results.reserve(video_urls.size());
    
    for (auto& future : futures) {
        try {
            results.push_back(future.get());
        } catch (const std::exception& e) {
            std::cerr << "Exception during video analysis: " << e.what() << std::endl;
            VideoAnalysisResult error_result;
            error_result.success = false;
            error_result.error_message = std::string("Exception: ") + e.what();
            results.push_back(error_result);
        }
    }
    
    std::cout << "Completed concurrent analysis of all videos" << std::endl;
    return results;
}

VideoAnalysisResult ChallengeVerifier::analyzeSingleVideo(const std::string& video_url) {
    VideoAnalysisResult result;
    result.video_url = video_url;
    
    try {
        // Use MediaPipe if available, otherwise fallback to OpenCV
#ifdef MEDIAPIPE_AVAILABLE
        VideoLivenessAnalysis analysis = video_detector_->analyzeSingleVideoWithMediaPipe(video_url);
#else
        VideoLivenessAnalysis analysis = video_detector_->analyzeSingleVideoWithOpenCV(video_url);
#endif
        
        if (analysis.has_failure) {
            result.success = false;
            result.error_message = std::string(analysis.failure_reason);
            return result;
        }
        
        if (!analysis.is_live) {
            result.success = false;
            result.error_message = "Video failed liveness detection";
            return result;
        }
        
        // Extract the primary movement direction
        if (analysis.directional_movements.empty()) {
            result.success = false;
            result.error_message = "No significant movement detected in video";
            return result;
        }
        
        // Use the first (and typically only) detected movement
        result.detected_direction = analysis.directional_movements[0].direction;
        result.confidence = analysis.confidence;
        result.success = true;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Analysis exception: ") + e.what();
    }
    
    return result;
}

MovementDirection ChallengeVerifier::challengeDirectionToMovementDirection(ChallengeDirection dir) const {
    switch (dir) {
        case ChallengeDirection::UP: return MovementDirection::UP;
        case ChallengeDirection::DOWN: return MovementDirection::DOWN;
        case ChallengeDirection::LEFT: return MovementDirection::LEFT;
        case ChallengeDirection::RIGHT: return MovementDirection::RIGHT;
        default: return MovementDirection::NONE;
    }
}

bool ChallengeVerifier::directionsMatch(ChallengeDirection expected, MovementDirection detected) const {
    // Convert challenge direction to movement direction and compare
    MovementDirection expected_movement = challengeDirectionToMovementDirection(expected);
    return expected_movement == detected;
}
