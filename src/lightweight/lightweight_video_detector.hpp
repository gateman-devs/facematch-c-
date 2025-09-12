#ifndef LIGHTWEIGHT_VIDEO_DETECTOR_HPP
#define LIGHTWEIGHT_VIDEO_DETECTOR_HPP

#include <string>
#include <vector>
#include <memory>
#include <array>
#include <optional>
#include <chrono>
#include <opencv2/opencv.hpp>

// TensorFlow Lite headers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

namespace lightweight {

// Direction enum matching the existing system
enum class Direction {
    NONE = 0,
    LEFT = 1,
    RIGHT = 2,
    UP = 3,
    DOWN = 4
};

// Face detection result from BlazeFace
struct FaceDetection {
    cv::Rect2f bbox;                    // Face bounding box
    std::array<cv::Point2f, 6> keypoints; // 6 facial keypoints
    float confidence;
    
    // Keypoint indices:
    // 0: Right eye
    // 1: Left eye  
    // 2: Nose tip
    // 3: Mouth center
    // 4: Right ear
    // 5: Left ear
    
    cv::Point2f getCenter() const {
        return cv::Point2f(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
    }
};

// Frame analysis result
struct FrameAnalysis {
    bool has_face = false;
    FaceDetection face;
    float spoof_score = 0.0f;  // 0.0 = likely fake, 1.0 = likely real
    cv::Point2f face_center;
    float timestamp = 0.0f;
};

// Movement tracking between frames
struct Movement {
    Direction direction = Direction::NONE;
    float confidence = 0.0f;
    float delta_x = 0.0f;  // Normalized movement in X (-1 to 1)
    float delta_y = 0.0f;  // Normalized movement in Y (-1 to 1)
};

// Video analysis result
struct VideoAnalysis {
    bool success = false;
    bool is_live = false;
    Direction primary_direction = Direction::NONE;
    float average_spoof_score = 0.0f;
    int total_frames_analyzed = 0;
    int frames_with_face = 0;
    std::vector<Movement> movements;
    std::string error_message;
    
    // Performance metrics
    float total_processing_time_ms = 0.0f;
    float avg_frame_time_ms = 0.0f;
};

class LightweightVideoDetector {
public:
    LightweightVideoDetector();
    ~LightweightVideoDetector();
    
    // Initialize models
    bool initialize(const std::string& models_dir = "./models");
    
    // Analyze video for liveness and head movement
    VideoAnalysis analyzeVideo(const std::string& video_path_or_url);
    
    // Analyze multiple videos concurrently
    std::vector<VideoAnalysis> analyzeVideos(const std::vector<std::string>& video_paths_or_urls);
    
    // Configuration
    void setFrameSampleRate(int rate) { frame_sample_rate_ = rate; }
    void setMinMovementThreshold(float threshold) { min_movement_threshold_ = threshold; }
    void setSpoofThreshold(float threshold) { spoof_threshold_ = threshold; }
    void setMaxFramesToAnalyze(int max_frames) { max_frames_to_analyze_ = max_frames; }
    
private:
    // TensorFlow Lite models
    std::unique_ptr<tflite::FlatBufferModel> blazeface_model_;
    std::unique_ptr<tflite::Interpreter> blazeface_interpreter_;
    
    std::unique_ptr<tflite::FlatBufferModel> antispoof_model_;
    std::unique_ptr<tflite::Interpreter> antispoof_interpreter_;
    
    // Model input/output details
    struct ModelInfo {
        int input_width = 0;
        int input_height = 0;
        int input_channels = 0;
        bool input_normalized = false;  // If true, expects [0,1] range
        std::vector<int> input_tensor_indices;
        std::vector<int> output_tensor_indices;
    };
    
    ModelInfo blazeface_info_;
    ModelInfo antispoof_info_;
    
    // Configuration
    int frame_sample_rate_ = 3;        // Analyze every Nth frame
    float min_movement_threshold_ = 0.15f; // 15% of frame width/height
    float spoof_threshold_ = 0.5f;     // Threshold for real vs fake
    int max_frames_to_analyze_ = 30;   // Max frames to process per video
    
    // Initialization flag
    bool initialized_ = false;
    
    // Model loading
    bool loadBlazeFaceModel(const std::string& model_path);
    bool loadAntiSpoofModel(const std::string& model_path);
    
    // Video processing
    std::vector<cv::Mat> extractFrames(const std::string& video_path_or_url);
    cv::Mat downloadVideo(const std::string& url, std::string& temp_path);
    
    // URL decoding utility
    std::string decodeUrl(const std::string& encoded_url);
    
    // Face detection with BlazeFace
    std::optional<FaceDetection> detectFace(const cv::Mat& frame);
    cv::Mat preprocessForBlazeFace(const cv::Mat& frame);
    FaceDetection postprocessBlazeFace(float* output_locations, float* output_scores);
    
    // Anti-spoofing
    float checkAntiSpoof(const cv::Mat& frame, const FaceDetection& face);
    cv::Mat preprocessForAntiSpoof(const cv::Mat& frame, const cv::Rect2f& face_rect);
    
    // Movement analysis
    Movement calculateMovement(const FrameAnalysis& prev, const FrameAnalysis& curr);
    Direction determineDirection(float delta_x, float delta_y);
    Direction getPrimaryDirection(const std::vector<Movement>& movements);
    
    // Utility functions
    cv::Mat resizeKeepAspectRatio(const cv::Mat& input, int target_width, int target_height);
    float calculateIOU(const cv::Rect2f& box1, const cv::Rect2f& box2);
    void applyNMS(std::vector<FaceDetection>& detections, float iou_threshold = 0.5f);
    
    // Performance monitoring
    std::chrono::steady_clock::time_point start_time_;
    void startTimer() { start_time_ = std::chrono::steady_clock::now(); }
    float getElapsedMs() {
        auto end_time = std::chrono::steady_clock::now();
        return std::chrono::duration<float, std::milli>(end_time - start_time_).count();
    }
};

// Utility class for concurrent video processing
class ConcurrentVideoProcessor {
public:
    static std::vector<VideoAnalysis> processVideos(
        LightweightVideoDetector& detector,
        const std::vector<std::string>& video_urls,
        int max_threads = 4
    );
};

} // namespace lightweight

#endif // LIGHTWEIGHT_VIDEO_DETECTOR_HPP