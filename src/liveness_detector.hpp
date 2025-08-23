#ifndef LIVENESS_DETECTOR_HPP
#define LIVENESS_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <vector>
#include <memory>
#include <string>
#include <cstring>

struct LivenessAnalysis {
    bool is_live;
    float confidence;
    float texture_score;
    float landmark_consistency;
    float image_quality;
    float overall_score;
    float sharpness_score;
    float lighting_score;
    float spoof_detection_score;
    bool has_failure;
    char failure_reason[256];  // Use fixed-size array instead of std::string
    
    LivenessAnalysis() : is_live(false), confidence(0.0f), texture_score(0.0f), 
                        landmark_consistency(0.0f), image_quality(0.0f), overall_score(0.0f),
                        sharpness_score(0.0f), lighting_score(0.0f), spoof_detection_score(0.0f),
                        has_failure(false) {
        std::memset(failure_reason, 0, sizeof(failure_reason));  // Initialize all bytes to 0
    }
};

class LivenessDetector {
public:
    LivenessDetector();
    ~LivenessDetector() = default;
    
    // Initialize with shape predictor for landmark detection
    bool initialize(const std::string& shape_predictor_path);
    
    // Main liveness detection method
    LivenessAnalysis checkLiveness(const cv::Mat& image);
    
    // Individual analysis methods
    float analyzeTexture(const cv::Mat& face_region);
    float checkLandmarkConsistency(const cv::Mat& image);
    float analyzeImageQuality(const cv::Mat& image);
    float analyzeSharpness(const cv::Mat& image);
    float analyzeLightingQuality(const cv::Mat& image);
    float detectSpoofing(const cv::Mat& image);
    
    // Check if detector is initialized
    bool isInitialized() const { return initialized; }

private:
    // Models and detectors
    dlib::frontal_face_detector face_detector;
    dlib::shape_predictor shape_predictor;
    bool initialized;
    
    // Production-level filtering thresholds (derived from test case analysis)
    static constexpr float LIVENESS_THRESHOLD = 0.6f;
    static constexpr float MIN_SHARPNESS_THRESHOLD = 0.05f;  // Spoofed images have very low sharpness
    static constexpr float MIN_TEXTURE_THRESHOLD = 0.4f;     // Minimum for real faces
    static constexpr float MIN_LBP_UNIFORMITY = 0.6f;       // Spoofed images show high uniformity
    static constexpr float MIN_LIGHTING_QUALITY = 0.3f;     // Poor lighting threshold
    static constexpr float MAX_REFLECTION_RATIO = 0.08f;    // Screen reflections indicator
    
    // Scoring weights
    static constexpr float TEXTURE_WEIGHT = 0.3f;
    static constexpr float LANDMARK_WEIGHT = 0.2f;
    static constexpr float QUALITY_WEIGHT = 0.2f;
    static constexpr float SHARPNESS_WEIGHT = 0.15f;
    static constexpr float LIGHTING_WEIGHT = 0.1f;
    static constexpr float SPOOF_WEIGHT = 0.05f;
    
    // Helper methods
    cv::Mat extractFaceRegion(const cv::Mat& image);
    float calculateLBPUniformity(const cv::Mat& region);
    float analyzeFrequencyDomain(const cv::Mat& region);
    float checkEdgeConsistency(const cv::Mat& region);
    std::vector<dlib::point> detectLandmarks(const cv::Mat& image);
    float calculateLandmarkStability(const std::vector<dlib::point>& landmarks);
    float analyzeEyeRegion(const cv::Mat& image, const std::vector<dlib::point>& landmarks);
    float calculateMoirePatterns(const cv::Mat& image);
    float analyzeColorDistribution(const cv::Mat& image);
    float detectReflections(const cv::Mat& image);
};

#endif // LIVENESS_DETECTOR_HPP
