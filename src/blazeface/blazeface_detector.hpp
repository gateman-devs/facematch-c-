#pragma once

#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <vector>
#include <memory>
#include <optional>
#include <string>

namespace blazeface {

// Face detection result
struct FaceDetection {
    cv::Rect2f bbox;                    // Face bounding box (normalized coordinates)
    float confidence;                    // Detection confidence score
    std::vector<cv::Point2f> keypoints; // 6 facial keypoints from BlazeFace
    
    // Convert normalized bbox to pixel coordinates
    cv::Rect toPixelCoords(int img_width, int img_height) const {
        return cv::Rect(
            static_cast<int>(bbox.x * img_width),
            static_cast<int>(bbox.y * img_height),
            static_cast<int>(bbox.width * img_width),
            static_cast<int>(bbox.height * img_height)
        );
    }
    
    cv::Point2f getCenter() const {
        return cv::Point2f(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);
    }
};

// Head pose angles
struct HeadPose {
    float yaw;   // Left/Right rotation
    float pitch; // Up/Down rotation
    float roll;  // Tilt rotation
    
    HeadPose() : yaw(0), pitch(0), roll(0) {}
    HeadPose(float y, float p, float r) : yaw(y), pitch(p), roll(r) {}
};

// Video analysis result
struct VideoAnalysis {
    bool has_face;
    float face_presence_ratio;
    std::string primary_direction;  // "LEFT", "RIGHT", "UP", "DOWN", "NONE"
    float confidence;
    float avg_face_size;
    int total_frames;
    int frames_with_face;
    std::vector<HeadPose> poses;
    bool is_live;
    float liveness_score;
    
    VideoAnalysis() : has_face(false), face_presence_ratio(0), primary_direction("NONE"),
                      confidence(0), avg_face_size(0), total_frames(0), 
                      frames_with_face(0), is_live(false), liveness_score(0) {}
};

class BlazeFaceDetector {
public:
    BlazeFaceDetector();
    ~BlazeFaceDetector();
    
    // Initialize the detector with model path
    bool initialize(const std::string& model_path);
    
    // Detect faces in an image
    std::vector<FaceDetection> detectFaces(const cv::Mat& image);
    
    // Get face with highest confidence
    std::optional<FaceDetection> detectBestFace(const cv::Mat& image);
    
    // Calculate head pose from face keypoints
    HeadPose calculateHeadPose(const FaceDetection& face, const cv::Size& image_size);
    
    // Analyze video for movement direction
    VideoAnalysis analyzeVideo(const std::string& video_path_or_url);
    
    // Analyze multiple videos concurrently (for challenge verification)
    std::vector<VideoAnalysis> analyzeVideos(const std::vector<std::string>& video_paths_or_urls, 
                                            int liveness_check_index = -1);
    
    // Check if model is loaded
    bool isInitialized() const { return interpreter_ != nullptr; }
    
    // Configuration
    void setConfidenceThreshold(float threshold) { confidence_threshold_ = threshold; }
    void setNMSThreshold(float threshold) { nms_threshold_ = threshold; }
    
private:
    // TensorFlow Lite components
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    
    // Model parameters
    static constexpr int INPUT_WIDTH = 128;
    static constexpr int INPUT_HEIGHT = 128;
    static constexpr int NUM_KEYPOINTS = 6;
    static constexpr int MAX_DETECTIONS = 100;
    
    // Detection thresholds
    float confidence_threshold_ = 0.5f;
    float nms_threshold_ = 0.3f;
    
    // Preprocessing
    cv::Mat preprocessImage(const cv::Mat& image);
    
    // Postprocessing
    std::vector<FaceDetection> postprocessDetections(float* raw_boxes, float* raw_scores, 
                                                     int num_boxes, const cv::Size& original_size);
    
    // Non-maximum suppression
    std::vector<FaceDetection> applyNMS(const std::vector<FaceDetection>& detections);
    
    // Decode anchors (BlazeFace specific)
    std::vector<cv::Rect2f> decodeBoxes(float* raw_boxes, int num_boxes);
    
    // Extract frames from video
    std::vector<cv::Mat> extractFrames(const std::string& video_path_or_url, int max_frames = 30);
    
    // Download video from URL
    std::string downloadVideo(const std::string& url);
    
    // Analyze head movement direction
    std::string detectMovementDirection(const std::vector<HeadPose>& poses);
    
    // Perform liveness detection on video
    float performLivenessCheck(const std::vector<cv::Mat>& frames, const std::vector<FaceDetection>& detections);
    
    // Calculate movement statistics
    void calculateMovementStats(const std::vector<HeadPose>& poses, 
                               float& yaw_range, float& pitch_range, float& roll_range);
    
    // Helper to calculate IOU for NMS
    float calculateIOU(const cv::Rect2f& box1, const cv::Rect2f& box2);
};

// Concurrent video processor
class ConcurrentVideoProcessor {
public:
    static std::vector<VideoAnalysis> processVideos(
        BlazeFaceDetector& detector,
        const std::vector<std::string>& video_urls,
        int liveness_check_index = -1,  // -1 means random selection
        int max_threads = 4
    );
};

} // namespace blazeface