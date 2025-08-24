#ifndef VIDEO_LIVENESS_DETECTOR_HPP
#define VIDEO_LIVENESS_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <string>
#include <cstring>

#ifdef MEDIAPIPE_AVAILABLE
// MediaPipe includes
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#endif

enum class MovementDirection {
    NONE = 0,
    LEFT = 1,
    RIGHT = 2,
    UP = 3,
    DOWN = 4
};

struct HeadPoseMovement {
    float yaw_angle;           // Rotation around Y-axis (left-right head turn)
    float pitch_angle;         // Rotation around X-axis (up-down head movement)
                                   // Positive = looking up (nose tilts up)
                                   // Negative = looking down (nose tilts down)
    float roll_angle;          // Rotation around Z-axis (head tilt)
    float timestamp;           // Time in seconds
    
    HeadPoseMovement() : yaw_angle(0.0f), pitch_angle(0.0f), roll_angle(0.0f), timestamp(0.0f) {}
};

struct DirectionalMovement {
    MovementDirection direction;
    float magnitude;           // Degrees of movement
    float start_time;          // When movement started
    float end_time;            // When movement ended
    
    DirectionalMovement() : direction(MovementDirection::NONE), magnitude(0.0f), start_time(0.0f), end_time(0.0f) {}
    DirectionalMovement(MovementDirection dir, float mag, float start, float end) 
        : direction(dir), magnitude(mag), start_time(start), end_time(end) {}
};

struct VideoLivenessAnalysis {
    bool is_live;
    float confidence;
    std::vector<HeadPoseMovement> pose_movements;
    std::vector<DirectionalMovement> directional_movements;
    float yaw_range;           // Maximum yaw movement range detected
    float pitch_range;         // Maximum pitch movement range detected
    int frame_count;
    float duration_seconds;
    bool has_sufficient_movement;
    int left_movements;        // Count of significant left movements
    int right_movements;       // Count of significant right movements
    int up_movements;          // Count of significant up movements
    int down_movements;        // Count of significant down movements
    bool has_failure;
    char failure_reason[256];
    
    VideoLivenessAnalysis() : is_live(false), confidence(0.0f), yaw_range(0.0f), 
                             pitch_range(0.0f), frame_count(0), duration_seconds(0.0f),
                             has_sufficient_movement(false), left_movements(0), 
                             right_movements(0), up_movements(0), down_movements(0),
                             has_failure(false) {
        std::memset(failure_reason, 0, sizeof(failure_reason));
    }
};

class VideoLivenessDetector {
public:
    VideoLivenessDetector();
    ~VideoLivenessDetector();
    
    // Initialize the detector
    bool initialize();
    
    // Process video from URL
    VideoLivenessAnalysis analyzeVideoFromUrl(const std::string& video_url);
    
    // Process video from local file
    VideoLivenessAnalysis analyzeVideoFromFile(const std::string& video_path);
    
    // Process video from OpenCV VideoCapture
    VideoLivenessAnalysis analyzeVideo(cv::VideoCapture& cap);
    
    // Check if detector is initialized
    bool isInitialized() const { return initialized; }

private:
    bool initialized;
    
#ifdef MEDIAPIPE_AVAILABLE
    std::unique_ptr<mediapipe::CalculatorGraph> graph;
    std::string graph_config;
#endif

    // Reference video metrics for movement patterns
    static constexpr float DOWN_YAW_REF = 1.52f;       // DOWN: yaw_range: 1.52°, pitch_range: 13.64°
    static constexpr float DOWN_PITCH_REF = 13.64f;
    static constexpr float UP_YAW_REF = 1.06f;         // UP: yaw_range: 1.06°, pitch_range: 7.90°
    static constexpr float UP_PITCH_REF = 7.90f;
    static constexpr float LEFT_YAW_REF = 8.41f;       // LEFT: yaw_range: 8.41°, pitch_range: 1.70°
    static constexpr float LEFT_PITCH_REF = 1.70f;
    static constexpr float RIGHT_YAW_REF = 7.90f;      // RIGHT: yaw_range: 7.90°, pitch_range: 3.81°
    static constexpr float RIGHT_PITCH_REF = 3.81f;
    
    // Movement detection thresholds based on reference videos
    static constexpr float MIN_HORIZONTAL_RANGE = 3.0f;  // Minimum yaw range for left/right detection
    static constexpr float MIN_VERTICAL_RANGE = 3.0f;    // Minimum pitch range for up/down detection
    static constexpr float DOMINANCE_RATIO = 1.5f;       // Ratio for determining dominant axis
    static constexpr float PATTERN_SIMILARITY = 0.6f;    // Similarity threshold for pattern matching
    static constexpr float MIN_DURATION = 1.0f;          // Minimum video duration in seconds
    static constexpr int MIN_FRAMES = 30;                // Minimum number of frames
    static constexpr float SEGMENT_DURATION = 1.0f;      // Duration of each analysis segment in seconds
    static constexpr float MIN_MOVEMENT_MAGNITUDE = 2.0f; // Minimum movement to consider significant
    
    // Helper methods
    cv::Mat downloadVideoFromUrl(const std::string& url, const std::string& temp_filename);
    HeadPoseMovement calculateHeadPose(const cv::Mat& frame, float timestamp);
    HeadPoseMovement calculateHeadPoseOpenCV(const cv::Mat& frame, float timestamp);
    std::vector<cv::Point3f> getModel3DPoints();
    cv::Mat getCameraMatrix(const cv::Size& image_size);
    cv::Mat getDistortionCoefficients();
    std::vector<cv::Point2f> getFaceLandmarks(const cv::Mat& frame);
    bool analyzeMovementPatterns(const std::vector<HeadPoseMovement>& movements, 
                                float& yaw_range, float& pitch_range);
    std::vector<DirectionalMovement> extractDirectionalMovements(const std::vector<HeadPoseMovement>& movements);
    std::vector<DirectionalMovement> analyzeMovementSegments(const std::vector<HeadPoseMovement>& movements);
    MovementDirection classifyMovementFromRanges(float yaw_range, float pitch_range);
    MovementDirection detectMovementInSegment(const std::vector<HeadPoseMovement>& segment);
    float calculatePatternSimilarity(float yaw_range, float pitch_range, MovementDirection expected_direction);
    std::string directionToString(MovementDirection direction);
    float calculateConfidence(const VideoLivenessAnalysis& analysis);
    void setFailureReason(VideoLivenessAnalysis& analysis, const std::string& reason);
    
#ifdef MEDIAPIPE_AVAILABLE
    HeadPoseMovement calculateHeadPoseMediaPipe(const cv::Mat& frame, float timestamp);
    bool setupMediaPipeGraph();
#endif
};

#endif // VIDEO_LIVENESS_DETECTOR_HPP
