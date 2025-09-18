#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <memory>
#include <optional>
#include <string>
#include <array>

namespace blazeface {

// Face detection result
struct FaceDetection {
    cv::Rect2f bbox;                    // Face bounding box (normalized coordinates)
    float confidence;                    // Detection confidence score
    std::vector<cv::Point2f> keypoints; // Facial keypoints
    
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
    
    // Initialize the detector with model path or use fallback
    bool initialize(const std::string& model_path = "");
    
    // Detect faces in an image
    std::vector<FaceDetection> detectFaces(const cv::Mat& image);
    
    // Get face with highest confidence
    std::optional<FaceDetection> detectBestFace(const cv::Mat& image);
    
    // Calculate head pose from face detection
    HeadPose calculateHeadPose(const FaceDetection& face, const cv::Size& image_size);
    
    // Calculate head pose using facial landmarks (more accurate)
    HeadPose calculateHeadPoseFromLandmarks(const cv::Mat& image, const cv::Rect& face_rect);
    
    // Analyze video for movement direction
    VideoAnalysis analyzeVideo(const std::string& video_path_or_url);
    
    // Analyze multiple videos concurrently (for challenge verification)
    std::vector<VideoAnalysis> analyzeVideos(const std::vector<std::string>& video_paths_or_urls, 
                                            int liveness_check_index = -1);
    
    // Check if model is loaded
    bool isInitialized() const { return initialized_; }
    
    // Configuration
    void setConfidenceThreshold(float threshold) { confidence_threshold_ = threshold; }
    void setNMSThreshold(float threshold) { nms_threshold_ = threshold; }
    
private:
    // OpenCV DNN network
    cv::dnn::Net net_;
    bool initialized_;
    bool use_dnn_model_;
    
    // Cascade classifier fallback
    cv::CascadeClassifier face_cascade_;
    cv::Ptr<cv::face::Facemark> facemark_;
    
    // Detection thresholds
    float confidence_threshold_ = 0.5f;
    float nms_threshold_ = 0.3f;
    
    // Model parameters
    static constexpr int INPUT_WIDTH = 300;
    static constexpr int INPUT_HEIGHT = 300;
    
    // Initialize DNN model
    bool initializeDNNModel(const std::string& model_path);
    
    // Initialize cascade classifier fallback
    bool initializeCascadeClassifier();
    
    // Detect faces using DNN
    std::vector<FaceDetection> detectFacesDNN(const cv::Mat& image);
    
    // Detect faces using cascade classifier
    std::vector<FaceDetection> detectFacesCascade(const cv::Mat& image);
    
    // Extract facial landmarks
    std::vector<cv::Point2f> extractLandmarks(const cv::Mat& image, const cv::Rect& face_rect);
    
    // Extract frames from video
    std::vector<cv::Mat> extractFrames(const std::string& video_path_or_url, int max_frames = 30);
    
    // Download video from URL
    std::string downloadVideo(const std::string& url);
    
    // Analyze head movement direction
    std::string detectMovementDirection(const std::vector<HeadPose>& poses);
    
    // Perform liveness detection on video
    float performLivenessCheck(const std::vector<cv::Mat>& frames, 
                               const std::vector<FaceDetection>& detections);
    
    // Calculate movement statistics
    void calculateMovementStats(const std::vector<HeadPose>& poses, 
                               float& yaw_range, float& pitch_range, float& roll_range);
    
    // Helper to calculate IOU for NMS
    float calculateIOU(const cv::Rect2f& box1, const cv::Rect2f& box2);
    
    // Apply non-maximum suppression
    std::vector<FaceDetection> applyNMS(const std::vector<FaceDetection>& detections);
    
    // 3D model points for pose estimation
    std::vector<cv::Point3f> get3DModelPoints();
    
    // Get 2D image points from landmarks
    std::vector<cv::Point2f> get2DImagePoints(const std::vector<cv::Point2f>& landmarks);
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