#ifndef CHALLENGE_VERIFIER_HPP
#define CHALLENGE_VERIFIER_HPP

#include "challenge_generator.hpp"
#include "video_liveness_detector.hpp"
#include <vector>
#include <string>
#include <future>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct VideoAnalysisResult {
    bool success;
    MovementDirection detected_direction;
    std::string video_url;
    std::string error_message;
    float confidence;
    
    VideoAnalysisResult() : success(false), detected_direction(MovementDirection::NONE), confidence(0.0f) {}
};

struct ChallengeVerificationResult {
    bool passed;
    std::vector<ChallengeDirection> expected_directions;
    std::vector<MovementDirection> detected_directions;
    std::vector<VideoAnalysisResult> video_results;
    std::string error_message;
    
    ChallengeVerificationResult() : passed(false) {}
    
    // Convert to JSON for API response
    json toJson() const {
        json j;
        j["passed"] = passed;
        j["error_message"] = error_message;
        
        // Expected directions
        j["expected_directions"] = json::array();
        for (const auto& dir : expected_directions) {
            j["expected_directions"].push_back(Challenge::directionToString(dir));
        }
        
        // Detected directions
        j["detected_directions"] = json::array();
        for (const auto& dir : detected_directions) {
            j["detected_directions"].push_back(movementDirectionToString(dir));
        }
        
        // Video analysis details
        j["video_analysis"] = json::array();
        for (const auto& result : video_results) {
            json video_json;
            video_json["video_url"] = result.video_url;
            video_json["success"] = result.success;
            video_json["detected_direction"] = movementDirectionToString(result.detected_direction);
            video_json["confidence"] = result.confidence;
            video_json["error_message"] = result.error_message;
            j["video_analysis"].push_back(video_json);
        }
        
        return j;
    }
    
private:
    static std::string movementDirectionToString(MovementDirection dir) {
        switch (dir) {
            case MovementDirection::LEFT: return "left";
            case MovementDirection::RIGHT: return "right";
            case MovementDirection::UP: return "up";
            case MovementDirection::DOWN: return "down";
            case MovementDirection::NONE:
            default: return "none";
        }
    }
};

class ChallengeVerifier {
public:
    ChallengeVerifier();
    ~ChallengeVerifier() = default;
    
    // Initialize the verifier with a video liveness detector
    bool initialize(std::shared_ptr<VideoLivenessDetector> detector);
    
    // Verify a challenge by analyzing 4 videos concurrently
    ChallengeVerificationResult verifyChallenge(
        const Challenge& challenge, 
        const std::vector<std::string>& video_urls);
    
    // Check if verifier is initialized
    bool isInitialized() const { return initialized_; }
    
private:
    bool initialized_;
    std::shared_ptr<VideoLivenessDetector> video_detector_;
    
    // Analyze a single video and extract the primary movement direction
    VideoAnalysisResult analyzeSingleVideo(const std::string& video_url);
    
    // Convert ChallengeDirection to MovementDirection for comparison
    MovementDirection challengeDirectionToMovementDirection(ChallengeDirection dir) const;
    
    // Check if two directions match (with some tolerance for similar movements)
    bool directionsMatch(ChallengeDirection expected, MovementDirection detected) const;
    
    // Analyze videos concurrently using futures
    std::vector<VideoAnalysisResult> analyzeVideosConcurrently(const std::vector<std::string>& video_urls);
};

#endif // CHALLENGE_VERIFIER_HPP
