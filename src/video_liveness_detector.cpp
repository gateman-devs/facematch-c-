#include "video_liveness_detector.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <curl/curl.h>
#include <filesystem>
#include <thread>
#include <functional>

VideoLivenessDetector::VideoLivenessDetector() : initialized(false), mediapipe_models_loaded(false) {
#ifdef MEDIAPIPE_AVAILABLE
    graph = nullptr;
#endif
}

VideoLivenessDetector::~VideoLivenessDetector() {
#ifdef MEDIAPIPE_AVAILABLE
    if (graph) {
        graph->CloseInputStream("input_video").IgnoreError();
        graph->WaitUntilDone().IgnoreError();
    }
#endif
}

bool VideoLivenessDetector::initialize() {
    try {
        // Try to initialize MediaPipe models using OpenCV DNN first
        if (initializeMediaPipeModels()) {
            std::cout << "Video liveness detector initialized with MediaPipe models" << std::endl;
            initialized = true;
            return true;
        }

#ifdef MEDIAPIPE_AVAILABLE
        if (setupMediaPipeGraph()) {
            std::cout << "Video liveness detector initialized with MediaPipe framework" << std::endl;
            initialized = true;
            return true;
        } else {
            std::cout << "MediaPipe setup failed, falling back to OpenCV" << std::endl;
        }
#endif

        // Fallback to OpenCV-based implementation
        std::cout << "Video liveness detector initialized with OpenCV fallback" << std::endl;
        initialized = true;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error initializing video liveness detector: " << e.what() << std::endl;
        return false;
    }
}

bool VideoLivenessDetector::initializeMediaPipeModels() {
    try {
        // Try to load MediaPipe BlazeFace model for face detection
        std::string face_detection_model = "./models/mediapipe/blaze_face_short_range.tflite";
        if (std::filesystem::exists(face_detection_model)) {
            face_detection_net = cv::dnn::readNetFromTFLite(face_detection_model);
            if (face_detection_net.empty()) {
                std::cerr << "Failed to load MediaPipe BlazeFace model" << std::endl;
                return false;
            }
            std::cout << "Loaded MediaPipe BlazeFace model for face detection" << std::endl;
        } else {
            std::cerr << "MediaPipe BlazeFace model not found at: " << face_detection_model << std::endl;
            return false;
        }

        // Try to load MediaPipe Face Landmarker model
        std::string face_landmark_model = "./models/mediapipe/face_landmarker.task";
        if (std::filesystem::exists(face_landmark_model)) {
            // Note: .task files are MediaPipe's format and can't be loaded directly with OpenCV DNN
            // We'll use the BlazeFace for detection and estimate landmarks from face bounding box
            std::cout << "MediaPipe Face Landmarker model found (will use BlazeFace + estimation)" << std::endl;
        }

        mediapipe_models_loaded = true;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error initializing MediaPipe models: " << e.what() << std::endl;
        return false;
    }
}

VideoLivenessAnalysis VideoLivenessDetector::analyzeVideoFromUrl(const std::string& video_url) {
    VideoLivenessAnalysis analysis;
    
    if (!initialized) {
        setFailureReason(analysis, "detector_not_initialized");
        return analysis;
    }
    
    try {
        // Create temporary filename with thread ID to avoid collisions
        std::hash<std::thread::id> hasher;
        std::string temp_filename = "/tmp/gateman_video_" + std::to_string(std::time(nullptr)) + "_" + std::to_string(hasher(std::this_thread::get_id())) + ".mp4";
        
        // Download video
        cv::Mat downloaded = downloadVideoFromUrl(video_url, temp_filename);
        if (downloaded.empty()) {
            setFailureReason(analysis, "failed_to_download_video");
            return analysis;
        }
        
        // Analyze the downloaded video
        analysis = analyzeVideoFromFile(temp_filename);
        
        // Clean up temporary file
        std::filesystem::remove(temp_filename);
        
        return analysis;
        
    } catch (const std::exception& e) {
        std::cerr << "Error analyzing video from URL: " << e.what() << std::endl;
        setFailureReason(analysis, "error_processing_video_url");
        return analysis;
    }
}

VideoLivenessAnalysis VideoLivenessDetector::analyzeVideoFromFile(const std::string& video_path) {
    VideoLivenessAnalysis analysis;
    
    if (!initialized) {
        setFailureReason(analysis, "detector_not_initialized");
        return analysis;
    }
    
    try {
        // Validate file exists and has reasonable size
        if (!std::filesystem::exists(video_path)) {
            setFailureReason(analysis, "video_file_not_found");
            return analysis;
        }
        
        auto file_size = std::filesystem::file_size(video_path);
        if (file_size < 1024) {
            setFailureReason(analysis, "video_file_too_small");
            return analysis;
        }
        
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            setFailureReason(analysis, "failed_to_open_video_file");
            return analysis;
        }
        
        // Additional validation - try to read basic video properties
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        
        if (fps <= 0 || total_frames <= 0) {
            setFailureReason(analysis, "invalid_video_properties");
            return analysis;
        }
        
        std::cout << "Video file validated successfully: " << video_path 
                  << " (" << file_size << " bytes, " << total_frames << " frames, " << fps << " FPS)" << std::endl;
        
        return analyzeVideo(cap);
        
    } catch (const std::exception& e) {
        std::cerr << "Error analyzing video from file: " << e.what() << std::endl;
        setFailureReason(analysis, "error_processing_video_file");
        return analysis;
    }
}

VideoLivenessAnalysis VideoLivenessDetector::analyzeVideo(cv::VideoCapture& cap) {
    VideoLivenessAnalysis analysis;
    
    if (!initialized) {
        setFailureReason(analysis, "detector_not_initialized");
        return analysis;
    }
    
    try {
        // Get video properties
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        analysis.duration_seconds = static_cast<float>(total_frames / fps);
        
        std::cout << "Analyzing video: " << total_frames << " frames, " 
                  << analysis.duration_seconds << " seconds, " << fps << " FPS" << std::endl;
        
        // Check minimum duration
        if (analysis.duration_seconds < MIN_DURATION) {
            setFailureReason(analysis, "video_too_short");
            return analysis;
        }
        
        // Process frames
        cv::Mat frame;
        int frame_count = 0;
        const int skip_frames = std::max(1, static_cast<int>(fps / 10)); // Process ~10 FPS
        
        while (cap.read(frame) && frame_count < total_frames) {
            if (frame_count % skip_frames == 0) {
                float timestamp = static_cast<float>(frame_count) / fps;
                
                HeadPoseMovement movement = calculateHeadPose(frame, timestamp);
                
                // Only add valid movements
                if (movement.yaw_angle != 0.0f || movement.pitch_angle != 0.0f) {
                    analysis.pose_movements.push_back(movement);
                }
            }
            frame_count++;
        }
        
        analysis.frame_count = frame_count;
        
        // Check if we have enough data
        if (static_cast<int>(analysis.pose_movements.size()) < MIN_FRAMES / skip_frames) {
            setFailureReason(analysis, "insufficient_face_detection");
            return analysis;
        }
        
        // Analyze movement patterns
        analysis.has_sufficient_movement = analyzeMovementPatterns(
            analysis.pose_movements, analysis.yaw_range, analysis.pitch_range);
        
        // Extract directional movements using segment-based analysis
        analysis.directional_movements = analyzeMovementSegments(analysis.pose_movements);
        
        // Count significant movements by direction
        for (const auto& movement : analysis.directional_movements) {
            switch (movement.direction) {
                case MovementDirection::LEFT:
                    analysis.left_movements++;
                    break;
                case MovementDirection::RIGHT:
                    analysis.right_movements++;
                    break;
                case MovementDirection::UP:
                    analysis.up_movements++;
                    break;
                case MovementDirection::DOWN:
                    analysis.down_movements++;
                    break;
                default:
                    break;
            }
        }
        
        // Calculate confidence and determine liveness
        analysis.confidence = calculateConfidence(analysis);
        analysis.is_live = analysis.has_sufficient_movement && 
                          (analysis.yaw_range >= MIN_HORIZONTAL_RANGE || analysis.pitch_range >= MIN_VERTICAL_RANGE) &&
                          analysis.directional_movements.size() >= 1;
        
        std::cout << "Video analysis complete: " << analysis.pose_movements.size() 
                  << " pose samples, yaw range: " << analysis.yaw_range 
                  << "°, pitch range: " << analysis.pitch_range << "°" << std::endl;
        std::cout << "Directional movements detected: " << analysis.directional_movements.size()
                  << " (Left: " << analysis.left_movements 
                  << ", Right: " << analysis.right_movements
                  << ", Up: " << analysis.up_movements
                  << ", Down: " << analysis.down_movements << ")" << std::endl;
        
        return analysis;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during video analysis: " << e.what() << std::endl;
        setFailureReason(analysis, "error_during_analysis");
        return analysis;
    }
}

HeadPoseMovement VideoLivenessDetector::calculateHeadPose(const cv::Mat& frame, float timestamp) {
    // Try MediaPipe models first (using OpenCV DNN)
    if (mediapipe_models_loaded) {
        HeadPoseMovement movement = calculateHeadPoseMediaPipeDNN(frame, timestamp);
        if (movement.yaw_angle != 0.0f || movement.pitch_angle != 0.0f) {
            return movement;
        }
    }

#ifdef MEDIAPIPE_AVAILABLE
    if (graph) {
        return calculateHeadPoseMediaPipe(frame, timestamp);
    }
#endif
    // Fallback to OpenCV implementation
    return calculateHeadPoseOpenCV(frame, timestamp);
}

HeadPoseMovement VideoLivenessDetector::calculateHeadPoseOpenCV(const cv::Mat& frame, float timestamp) {
    HeadPoseMovement movement;
    movement.timestamp = timestamp;
    
    try {
        // Get facial landmarks
        std::vector<cv::Point2f> landmarks = getFaceLandmarks(frame);
        
        if (landmarks.size() < 6) {
            return movement; // Return zeros if insufficient landmarks
        }
        
        // 3D model points (in arbitrary units)
        std::vector<cv::Point3f> model_points = getModel3DPoints();
        
        // Camera parameters
        cv::Mat camera_matrix = getCameraMatrix(frame.size());
        cv::Mat dist_coeffs = getDistortionCoefficients();
        
        // Solve PnP to get rotation and translation vectors
        cv::Mat rotation_vector, translation_vector;
        bool success = cv::solvePnP(model_points, landmarks, camera_matrix, dist_coeffs,
                                   rotation_vector, translation_vector);
        
        if (success) {
            // Convert rotation vector to Euler angles
            cv::Mat rotation_matrix;
            cv::Rodrigues(rotation_vector, rotation_matrix);
            
            // Extract Euler angles from rotation matrix
            double sy = sqrt(rotation_matrix.at<double>(0,0) * rotation_matrix.at<double>(0,0) +
                            rotation_matrix.at<double>(1,0) * rotation_matrix.at<double>(1,0));
            
            bool singular = sy < 1e-6;
            
            double x, y, z;
            if (!singular) {
                x = atan2(rotation_matrix.at<double>(2,1), rotation_matrix.at<double>(2,2));
                y = atan2(-rotation_matrix.at<double>(2,0), sy);
                z = atan2(rotation_matrix.at<double>(1,0), rotation_matrix.at<double>(0,0));
            } else {
                x = atan2(-rotation_matrix.at<double>(1,2), rotation_matrix.at<double>(1,1));
                y = atan2(-rotation_matrix.at<double>(2,0), sy);
                z = 0;
            }
            
            // Convert radians to degrees
            movement.pitch_angle = static_cast<float>(x * 180.0 / CV_PI);
            movement.yaw_angle = static_cast<float>(y * 180.0 / CV_PI);
            movement.roll_angle = static_cast<float>(z * 180.0 / CV_PI);
            
            // Normalize angles to [-180, 180] range
            auto normalizeAngle = [](float angle) {
                while (angle > 180.0f) angle -= 360.0f;
                while (angle < -180.0f) angle += 360.0f;
                return angle;
            };
            
            movement.pitch_angle = normalizeAngle(movement.pitch_angle);
            movement.yaw_angle = normalizeAngle(movement.yaw_angle);
            movement.roll_angle = normalizeAngle(movement.roll_angle);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error calculating head pose: " << e.what() << std::endl;
    }
    
    return movement;
}

std::vector<cv::Point3f> VideoLivenessDetector::getModel3DPoints() {
    // 3D model points for key facial landmarks (nose tip, chin, left eye corner,
    // right eye corner, left mouth corner, right mouth corner)
    return {
        cv::Point3f(0.0f, 0.0f, 0.0f),        // Nose tip
        cv::Point3f(0.0f, -330.0f, -65.0f),   // Chin
        cv::Point3f(-225.0f, 170.0f, -135.0f), // Left eye left corner
        cv::Point3f(225.0f, 170.0f, -135.0f),  // Right eye right corner
        cv::Point3f(-150.0f, -150.0f, -125.0f), // Left mouth corner
        cv::Point3f(150.0f, -150.0f, -125.0f)   // Right mouth corner
    };
}

cv::Mat VideoLivenessDetector::getCameraMatrix(const cv::Size& image_size) {
    // Approximate camera intrinsic parameters
    double focal_length = image_size.width;
    cv::Point2d center = cv::Point2d(image_size.width / 2, image_size.height / 2);
    
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
        focal_length, 0, center.x,
        0, focal_length, center.y,
        0, 0, 1);
    
    return camera_matrix;
}

cv::Mat VideoLivenessDetector::getDistortionCoefficients() {
    // Assume no lens distortion
    return cv::Mat::zeros(4, 1, cv::DataType<double>::type);
}

std::vector<cv::Point2f> VideoLivenessDetector::getFaceLandmarks(const cv::Mat& frame) {
    std::vector<cv::Point2f> landmarks;

    try {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        // Try multiple face detection approaches for robustness

        // First try Haar cascades with different parameters
        cv::CascadeClassifier face_cascade;
        std::vector<std::string> cascade_paths = {
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
            "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml",
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
        };

        std::vector<cv::Rect> faces;
        for (const auto& path : cascade_paths) {
            if (face_cascade.load(path)) {
                // Try with different parameters for better detection
                std::vector<cv::Rect> detected_faces;
                face_cascade.detectMultiScale(gray, detected_faces, 1.1, 2, 0, cv::Size(30, 30));
                faces.insert(faces.end(), detected_faces.begin(), detected_faces.end());

                // Also try with different scale factor and min neighbors
                face_cascade.detectMultiScale(gray, detected_faces, 1.2, 3, 0, cv::Size(40, 40));
                faces.insert(faces.end(), detected_faces.begin(), detected_faces.end());
            }
        }

        // Remove duplicate detections (simple approach)
        std::sort(faces.begin(), faces.end(), [](const cv::Rect& a, const cv::Rect& b) {
            return a.x < b.x;
        });
        auto last = std::unique(faces.begin(), faces.end(), [](const cv::Rect& a, const cv::Rect& b) {
            return std::abs(a.x - b.x) < 20 && std::abs(a.y - b.y) < 20;
        });
        faces.erase(last, faces.end());

        if (!faces.empty()) {
            // Use the largest face or the one closest to center
            cv::Rect best_face = faces[0];
            cv::Point center(frame.cols / 2, frame.rows / 2);

            for (const auto& face : faces) {
                cv::Point face_center(face.x + face.width/2, face.y + face.height/2);
                cv::Point best_center(best_face.x + best_face.width/2, best_face.y + best_face.height/2);

                double dist1 = cv::norm(center - face_center);
                double dist2 = cv::norm(center - best_center);

                if (dist1 < dist2) {
                    best_face = face;
                }
            }

            cv::Rect face = best_face;

            // Improved landmark estimation based on facial proportions
            // These ratios are based on average facial measurements
            float cx = face.x + face.width * 0.5f;
            float cy = face.y + face.height * 0.5f;
            float w = face.width;
            float h = face.height;

            // More accurate landmark positions based on facial anatomy
            landmarks = {
                cv::Point2f(cx, cy - h * 0.12f),           // Nose tip (slightly above center)
                cv::Point2f(cx, cy + h * 0.35f),           // Chin (below center)
                cv::Point2f(cx - w * 0.22f, cy - h * 0.08f), // Left eye corner
                cv::Point2f(cx + w * 0.22f, cy - h * 0.08f), // Right eye corner
                cv::Point2f(cx - w * 0.18f, cy + h * 0.15f), // Left mouth corner
                cv::Point2f(cx + w * 0.18f, cy + h * 0.15f)  // Right mouth corner
            };
        }

    } catch (const std::exception& e) {
        std::cerr << "Error detecting face landmarks: " << e.what() << std::endl;
    }

    return landmarks;
}

bool VideoLivenessDetector::analyzeMovementPatterns(const std::vector<HeadPoseMovement>& movements,
                                                   float& yaw_range, float& pitch_range) {
    if (movements.empty()) {
        yaw_range = 0.0f;
        pitch_range = 0.0f;
        return false;
    }
    
    // Find min/max angles
    float min_yaw = movements[0].yaw_angle;
    float max_yaw = movements[0].yaw_angle;
    float min_pitch = movements[0].pitch_angle;
    float max_pitch = movements[0].pitch_angle;
    
    for (const auto& movement : movements) {
        min_yaw = std::min(min_yaw, movement.yaw_angle);
        max_yaw = std::max(max_yaw, movement.yaw_angle);
        min_pitch = std::min(min_pitch, movement.pitch_angle);
        max_pitch = std::max(max_pitch, movement.pitch_angle);
    }
    
    yaw_range = max_yaw - min_yaw;
    pitch_range = max_pitch - min_pitch;
    
    // Check for sufficient movement
    return (yaw_range >= MIN_HORIZONTAL_RANGE || pitch_range >= MIN_VERTICAL_RANGE);
}

std::vector<DirectionalMovement> VideoLivenessDetector::analyzeMovementSegments(const std::vector<HeadPoseMovement>& movements) {
    std::vector<DirectionalMovement> directional_movements;
    
    if (movements.empty()) {
        return directional_movements;
    }
    
    float video_duration = movements.back().timestamp - movements.front().timestamp;
    
    // If video is short, analyze as single segment
    if (video_duration < SEGMENT_DURATION * 2) {
        return extractDirectionalMovements(movements);
    }
    
    // Divide video into overlapping segments for analysis
    float current_time = movements.front().timestamp;
    float end_time = movements.back().timestamp;
    
    std::cout << "Analyzing video in segments for multiple movements..." << std::endl;
    
    while (current_time + SEGMENT_DURATION <= end_time) {
        // Extract segment
        std::vector<HeadPoseMovement> segment;
        float segment_end = current_time + SEGMENT_DURATION;
        
        for (const auto& movement : movements) {
            if (movement.timestamp >= current_time && movement.timestamp <= segment_end) {
                segment.push_back(movement);
            }
        }
        
        if (segment.size() >= 10) { // Minimum samples per segment
            MovementDirection detected_direction = detectMovementInSegment(segment);
            
            if (detected_direction != MovementDirection::NONE) {
                // Calculate movement magnitude for this segment
                float min_yaw = segment[0].yaw_angle, max_yaw = segment[0].yaw_angle;
                float min_pitch = segment[0].pitch_angle, max_pitch = segment[0].pitch_angle;
                
                for (const auto& m : segment) {
                    min_yaw = std::min(min_yaw, m.yaw_angle);
                    max_yaw = std::max(max_yaw, m.yaw_angle);
                    min_pitch = std::min(min_pitch, m.pitch_angle);
                    max_pitch = std::max(max_pitch, m.pitch_angle);
                }
                
                float yaw_range = max_yaw - min_yaw;
                float pitch_range = max_pitch - min_pitch;
                float magnitude = (detected_direction == MovementDirection::LEFT || detected_direction == MovementDirection::RIGHT) 
                                 ? yaw_range : pitch_range;
                
                // Only add if this is a significant movement
                if (magnitude >= MIN_MOVEMENT_MAGNITUDE) {
                    // Check if this extends a previous movement or is a new one
                    bool is_continuation = false;
                    if (!directional_movements.empty()) {
                        auto& last_movement = directional_movements.back();
                        if (last_movement.direction == detected_direction && 
                            (current_time - last_movement.end_time) < 0.5f) {
                            // Extend existing movement
                            last_movement.end_time = segment_end;
                            last_movement.magnitude = std::max(last_movement.magnitude, magnitude);
                            is_continuation = true;
                        }
                    }
                    
                    if (!is_continuation) {
                        // Create new movement
                        DirectionalMovement new_movement(detected_direction, magnitude, current_time, segment_end);
                        directional_movements.push_back(new_movement);
                        
                        float similarity = calculatePatternSimilarity(yaw_range, pitch_range, detected_direction);
                        std::cout << "Segment " << current_time << "s-" << segment_end << "s: Detected " 
                                  << directionToString(detected_direction) << " movement (" 
                                  << magnitude << "°, " << similarity * 100 << "% similarity)" << std::endl;
                    }
                }
            }
        }
        
        // Move to next segment with 50% overlap
        current_time += SEGMENT_DURATION * 0.5f;
    }
    
    // Filter out very short movements and merge similar consecutive movements
    auto it = directional_movements.begin();
    while (it != directional_movements.end()) {
        if ((it->end_time - it->start_time) < 0.3f) {
            it = directional_movements.erase(it);
        } else {
            ++it;
        }
    }
    
    std::cout << "Total movements detected: " << directional_movements.size() << std::endl;
    return directional_movements;
}

MovementDirection VideoLivenessDetector::detectMovementInSegment(const std::vector<HeadPoseMovement>& segment) {
    if (segment.size() < 5) {
        return MovementDirection::NONE;
    }
    
    // Calculate movement ranges in this segment
    float min_yaw = segment[0].yaw_angle, max_yaw = segment[0].yaw_angle;
    float min_pitch = segment[0].pitch_angle, max_pitch = segment[0].pitch_angle;
    
    for (const auto& movement : segment) {
        min_yaw = std::min(min_yaw, movement.yaw_angle);
        max_yaw = std::max(max_yaw, movement.yaw_angle);
        min_pitch = std::min(min_pitch, movement.pitch_angle);
        max_pitch = std::max(max_pitch, movement.pitch_angle);
    }
    
    float yaw_range = max_yaw - min_yaw;
    float pitch_range = max_pitch - min_pitch;
    
    // Also analyze movement direction from start to end
    float yaw_delta = segment.back().yaw_angle - segment.front().yaw_angle;
    float pitch_delta = segment.back().pitch_angle - segment.front().pitch_angle;
    
    // Determine if there's significant movement in this segment
    if (yaw_range < MIN_MOVEMENT_MAGNITUDE && pitch_range < MIN_MOVEMENT_MAGNITUDE) {
        return MovementDirection::NONE;
    }
    
    // Classify based on dominant axis and direction
    if (yaw_range >= MIN_HORIZONTAL_RANGE && yaw_range > pitch_range * DOMINANCE_RATIO) {
        // Horizontal movement dominates
        float left_similarity = calculatePatternSimilarity(yaw_range, pitch_range, MovementDirection::LEFT);
        float right_similarity = calculatePatternSimilarity(yaw_range, pitch_range, MovementDirection::RIGHT);
        
        // Also consider the direction of movement
        if (yaw_delta < -1.0f && left_similarity > PATTERN_SIMILARITY) {
            return MovementDirection::LEFT;
        } else if (yaw_delta > 1.0f && right_similarity > PATTERN_SIMILARITY) {
            return MovementDirection::RIGHT;
        } else if (left_similarity > right_similarity && left_similarity > PATTERN_SIMILARITY) {
            return MovementDirection::LEFT;
        } else if (right_similarity > PATTERN_SIMILARITY) {
            return MovementDirection::RIGHT;
        }
    } else if (pitch_range >= MIN_VERTICAL_RANGE) {
        // Vertical movement dominates
        float up_similarity = calculatePatternSimilarity(yaw_range, pitch_range, MovementDirection::UP);
        float down_similarity = calculatePatternSimilarity(yaw_range, pitch_range, MovementDirection::DOWN);
        
        // Enhanced pitch direction detection with improved thresholds
        // COORDINATE SYSTEM: Pitch angle interpretation
        // Positive pitch_delta = looking down (nose tilts down, head moves down)
        // Negative pitch_delta = looking up (nose tilts up, head moves up)
        
        // Primary detection: Use pitch delta with improved thresholds
        const float MIN_PITCH_DELTA_DOWN = 2.5f;   // Minimum delta for down movement
        const float MIN_PITCH_DELTA_UP = -2.5f;    // Minimum delta for up movement
        
        if (pitch_delta > MIN_PITCH_DELTA_DOWN && pitch_range >= MIN_VERTICAL_RANGE && down_similarity > PATTERN_SIMILARITY) {
            return MovementDirection::DOWN;
        } else if (pitch_delta < MIN_PITCH_DELTA_UP && pitch_range >= MIN_VERTICAL_RANGE && up_similarity > PATTERN_SIMILARITY) {
            return MovementDirection::UP;
        }
        
        // Fallback detection: Use pattern similarity when delta is ambiguous
        else if (pitch_range >= MIN_VERTICAL_RANGE) {
            if (down_similarity > up_similarity && down_similarity > PATTERN_SIMILARITY) {
                return MovementDirection::DOWN;
            } else if (up_similarity > PATTERN_SIMILARITY) {
                return MovementDirection::UP;
            }
        }
    }
    
    return MovementDirection::NONE;
}

std::vector<DirectionalMovement> VideoLivenessDetector::extractDirectionalMovements(const std::vector<HeadPoseMovement>& movements) {
    std::vector<DirectionalMovement> directional_movements;
    
    if (movements.empty()) {
        return directional_movements;
    }
    
    // Calculate overall movement ranges
    float min_yaw = movements[0].yaw_angle;
    float max_yaw = movements[0].yaw_angle;
    float min_pitch = movements[0].pitch_angle;
    float max_pitch = movements[0].pitch_angle;
    
    for (const auto& movement : movements) {
        min_yaw = std::min(min_yaw, movement.yaw_angle);
        max_yaw = std::max(max_yaw, movement.yaw_angle);
        min_pitch = std::min(min_pitch, movement.pitch_angle);
        max_pitch = std::max(max_pitch, movement.pitch_angle);
    }
    
    float yaw_range = max_yaw - min_yaw;
    float pitch_range = max_pitch - min_pitch;
    
    // Classify movement based on overall ranges using reference patterns
    MovementDirection primary_direction = classifyMovementFromRanges(yaw_range, pitch_range);
    
    if (primary_direction != MovementDirection::NONE) {
        // Calculate similarity to reference pattern
        float similarity = calculatePatternSimilarity(yaw_range, pitch_range, primary_direction);
        
        // Create a single primary movement for the entire video
        float start_time = movements.front().timestamp;
        float end_time = movements.back().timestamp;
        float magnitude = (primary_direction == MovementDirection::LEFT || primary_direction == MovementDirection::RIGHT) 
                         ? yaw_range : pitch_range;
        
        DirectionalMovement primary_movement(primary_direction, magnitude, start_time, end_time);
        directional_movements.push_back(primary_movement);
        
        std::cout << "Detected " << directionToString(primary_direction) 
                  << " movement with " << similarity * 100 << "% similarity to reference pattern" << std::endl;
    }
    
    return directional_movements;
}

MovementDirection VideoLivenessDetector::classifyMovementFromRanges(float yaw_range, float pitch_range) {
    // Check if movement meets minimum thresholds
    bool has_horizontal = yaw_range >= MIN_HORIZONTAL_RANGE;
    bool has_vertical = pitch_range >= MIN_VERTICAL_RANGE;
    
    if (!has_horizontal && !has_vertical) {
        return MovementDirection::NONE;
    }
    
    // Determine dominant axis
    if (has_horizontal && (!has_vertical || yaw_range > pitch_range * DOMINANCE_RATIO)) {
        // Horizontal movement dominates - classify as LEFT or RIGHT
        // Compare to reference patterns to determine direction
        float left_similarity = calculatePatternSimilarity(yaw_range, pitch_range, MovementDirection::LEFT);
        float right_similarity = calculatePatternSimilarity(yaw_range, pitch_range, MovementDirection::RIGHT);
        
        if (left_similarity > right_similarity && left_similarity > PATTERN_SIMILARITY) {
            return MovementDirection::LEFT;
        } else if (right_similarity > PATTERN_SIMILARITY) {
            return MovementDirection::RIGHT;
        }
    } else if (has_vertical) {
        // Vertical movement dominates - classify as UP or DOWN
        float up_similarity = calculatePatternSimilarity(yaw_range, pitch_range, MovementDirection::UP);
        float down_similarity = calculatePatternSimilarity(yaw_range, pitch_range, MovementDirection::DOWN);
        
        if (down_similarity > up_similarity && down_similarity > PATTERN_SIMILARITY) {
            return MovementDirection::DOWN;
        } else if (up_similarity > PATTERN_SIMILARITY) {
            return MovementDirection::UP;
        }
    }
    
    return MovementDirection::NONE;
}

float VideoLivenessDetector::calculatePatternSimilarity(float yaw_range, float pitch_range, MovementDirection expected_direction) {
    float ref_yaw, ref_pitch;
    
    switch (expected_direction) {
        case MovementDirection::DOWN:
            ref_yaw = DOWN_YAW_REF;
            ref_pitch = DOWN_PITCH_REF;
            break;
        case MovementDirection::UP:
            ref_yaw = UP_YAW_REF;
            ref_pitch = UP_PITCH_REF;
            break;
        case MovementDirection::LEFT:
            ref_yaw = LEFT_YAW_REF;
            ref_pitch = LEFT_PITCH_REF;
            break;
        case MovementDirection::RIGHT:
            ref_yaw = RIGHT_YAW_REF;
            ref_pitch = RIGHT_PITCH_REF;
            break;
        default:
            return 0.0f;
    }
    
    // Calculate similarity based on ratio comparison
    float yaw_ratio = std::min(yaw_range, ref_yaw) / std::max(yaw_range, ref_yaw);
    float pitch_ratio = std::min(pitch_range, ref_pitch) / std::max(pitch_range, ref_pitch);
    
    // Weight the similarity based on which axis should be dominant for this movement
    float similarity;
    if (expected_direction == MovementDirection::LEFT || expected_direction == MovementDirection::RIGHT) {
        // For horizontal movements, yaw should dominate
        similarity = 0.7f * yaw_ratio + 0.3f * pitch_ratio;
    } else {
        // For vertical movements, pitch should dominate
        similarity = 0.3f * yaw_ratio + 0.7f * pitch_ratio;
    }
    
    return similarity;
}

std::string VideoLivenessDetector::directionToString(MovementDirection direction) {
    switch (direction) {
        case MovementDirection::LEFT: return "left";
        case MovementDirection::RIGHT: return "right";
        case MovementDirection::UP: return "up";
        case MovementDirection::DOWN: return "down";
        case MovementDirection::NONE:
        default: return "none";
    }
}

float VideoLivenessDetector::calculateConfidence(const VideoLivenessAnalysis& analysis) {
    float confidence = 0.0f;
    
    // Base confidence from movement range
    float yaw_score = std::min(analysis.yaw_range / MIN_HORIZONTAL_RANGE, 1.0f);
    float pitch_score = std::min(analysis.pitch_range / MIN_VERTICAL_RANGE, 1.0f);
    
    // Duration score
    float duration_score = std::min(analysis.duration_seconds / (MIN_DURATION * 2), 1.0f);
    
    // Frame count score
    float frame_score = std::min(static_cast<float>(analysis.pose_movements.size()) / MIN_FRAMES, 1.0f);
    
    // Pattern similarity score from detected movements
    float pattern_score = 0.0f;
    if (!analysis.directional_movements.empty()) {
        // Use the similarity score from the detected movement
        const auto& movement = analysis.directional_movements[0];
        pattern_score = calculatePatternSimilarity(analysis.yaw_range, analysis.pitch_range, movement.direction);
    }
    
    // Combined confidence with emphasis on pattern matching
    confidence = (0.25f * yaw_score + 0.25f * pitch_score + 0.3f * pattern_score + 
                 0.1f * duration_score + 0.1f * frame_score);
    
    return std::max(0.0f, std::min(1.0f, confidence));
}

void VideoLivenessDetector::setFailureReason(VideoLivenessAnalysis& analysis, const std::string& reason) {
    analysis.has_failure = true;
    strncpy(analysis.failure_reason, reason.c_str(), sizeof(analysis.failure_reason) - 1);
    analysis.failure_reason[sizeof(analysis.failure_reason) - 1] = '\0';
}

// Callback function for writing downloaded data
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::ofstream* userp) {
    size_t total_size = size * nmemb;
    userp->write(static_cast<char*>(contents), total_size);
    return total_size;
}

cv::Mat VideoLivenessDetector::downloadVideoFromUrl(const std::string& url, const std::string& temp_filename) {
    try {
        CURL* curl;
        CURLcode res;
        
        curl = curl_easy_init();
        if (!curl) {
            std::cerr << "Failed to initialize CURL" << std::endl;
            return cv::Mat();
        }
        
        std::ofstream outfile(temp_filename, std::ios::binary);
        if (!outfile.is_open()) {
            std::cerr << "Failed to open temporary file: " << temp_filename << std::endl;
            curl_easy_cleanup(curl);
            return cv::Mat();
        }
        
        // Set CURL options
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &outfile);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L); // 30 second timeout
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "GatemanFace/1.0");
        
        // Perform the request
        res = curl_easy_perform(curl);
        
        // Cleanup
        curl_easy_cleanup(curl);
        outfile.close();
        
        if (res != CURLE_OK) {
            std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
            std::filesystem::remove(temp_filename);
            return cv::Mat();
        }
        
        // Validate downloaded file size and format
        if (!std::filesystem::exists(temp_filename)) {
            std::cerr << "Downloaded file does not exist: " << temp_filename << std::endl;
            return cv::Mat();
        }
        
        auto file_size = std::filesystem::file_size(temp_filename);
        if (file_size < 1024) { // Less than 1KB is likely not a valid video
            std::cerr << "Downloaded file too small (" << file_size << " bytes): " << temp_filename << std::endl;
            std::filesystem::remove(temp_filename);
            return cv::Mat();
        }
        
        // Quick validation that OpenCV can open the file before proceeding
        cv::VideoCapture test_cap(temp_filename);
        if (!test_cap.isOpened()) {
            std::cerr << "OpenCV cannot open downloaded video file: " << temp_filename << std::endl;
            std::filesystem::remove(temp_filename);
            return cv::Mat();
        }
        test_cap.release();
        
        std::cout << "Successfully downloaded and validated video file: " << temp_filename 
                  << " (" << file_size << " bytes)" << std::endl;
        
        // Return a dummy Mat to indicate success
        // The actual video will be read from the file
        return cv::Mat::ones(1, 1, CV_8UC1);
        
    } catch (const std::exception& e) {
        std::cerr << "Error downloading video: " << e.what() << std::endl;
        return cv::Mat();
    }
}

#ifdef MEDIAPIPE_AVAILABLE
VideoLivenessAnalysis VideoLivenessDetector::analyzeSingleVideoWithMediaPipe(const std::string& video_url) {
    VideoLivenessAnalysis analysis;

    if (!initialized) {
        setFailureReason(analysis, "detector_not_initialized");
        return analysis;
    }

    try {
        // Create temporary filename with thread ID to avoid collisions
        std::hash<std::thread::id> hasher;
        std::string temp_filename = "/tmp/gateman_mediapipe_video_" + std::to_string(std::time(nullptr)) + "_" + std::to_string(hasher(std::this_thread::get_id())) + ".mp4";

        // Download video
        cv::Mat downloaded = downloadVideoFromUrl(video_url, temp_filename);
        if (downloaded.empty()) {
            setFailureReason(analysis, "failed_to_download_video");
            return analysis;
        }

        // Analyze the downloaded video using MediaPipe FaceMesh only
        analysis = analyzeVideoWithMediaPipeFaceMesh(temp_filename);

        // Clean up temporary file
        std::filesystem::remove(temp_filename);

        return analysis;

    } catch (const std::exception& e) {
        std::cerr << "Error analyzing video with MediaPipe FaceMesh: " << e.what() << std::endl;
        setFailureReason(analysis, "error_processing_video_mediapipe");
        return analysis;
    }
}
#endif // MEDIAPIPE_AVAILABLE

#ifdef MEDIAPIPE_AVAILABLE
VideoLivenessAnalysis VideoLivenessDetector::analyzeVideoWithMediaPipeFaceMesh(const std::string& video_path) {
    VideoLivenessAnalysis analysis;

    if (!initialized) {
        setFailureReason(analysis, "detector_not_initialized");
        return analysis;
    }

#ifndef MEDIAPIPE_AVAILABLE
    setFailureReason(analysis, "mediapipe_not_available");
    return analysis;
#endif

    try {
        // Open video file
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            setFailureReason(analysis, "failed_to_open_video_file");
            return analysis;
        }

        // Get video properties
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        analysis.duration_seconds = static_cast<float>(total_frames / fps);

        std::cout << "Analyzing video with MediaPipe FaceMesh: " << total_frames << " frames, "
                  << analysis.duration_seconds << " seconds, " << fps << " FPS" << std::endl;

        // Check minimum duration (1.5-5 seconds to accommodate various videos)
        if (analysis.duration_seconds < 1.5f) {
            setFailureReason(analysis, "video_too_short_minimum_1_5_seconds");
            return analysis;
        }
        if (analysis.duration_seconds > 5.0f) {
            setFailureReason(analysis, "video_too_long_maximum_5_seconds");
            return analysis;
        }

        // Initialize MediaPipe graph for this analysis session
        if (!setupMediaPipeGraphForAnalysis()) {
            setFailureReason(analysis, "failed_to_initialize_mediapipe_graph");
            return analysis;
        }

        // Process frames with MediaPipe FaceMesh
        cv::Mat frame;
        int frame_count = 0;
        const int skip_frames = std::max(1, static_cast<int>(fps / 15)); // Process ~15 FPS for better accuracy
        std::vector<HeadPoseMovement> pose_movements;
        bool first_major_movement_found = false;
        DirectionalMovement first_major_movement;

        // Anti-spoofing checks
        int consecutive_face_detections = 0;
        int total_frames_with_face = 0;
        float average_face_size = 0.0f;
        int face_size_samples = 0;

        while (cap.read(frame) && frame_count < total_frames && !first_major_movement_found) {
            if (frame_count % skip_frames == 0) {
                float timestamp = static_cast<float>(frame_count) / fps;

                // Calculate head pose using MediaPipe FaceMesh
                HeadPoseMovement movement = calculateHeadPoseMediaPipeFaceMesh(frame, timestamp);

                if (movement.yaw_angle != 0.0f || movement.pitch_angle != 0.0f) {
                    pose_movements.push_back(movement);
                    total_frames_with_face++;

                    // Anti-spoofing: Check for consistent face detection
                    consecutive_face_detections++;

                    // Anti-spoofing: Check for reasonable face size (not too small/large)
                    // This is a simplified check - in production you'd want more sophisticated spoofing detection
                    cv::Mat gray;
                    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                    cv::equalizeHist(gray, gray);

                    // Simple face detection for size estimation
                    cv::CascadeClassifier face_cascade;
                    std::string cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml";
                    if (face_cascade.load(cascade_path)) {
                        std::vector<cv::Rect> faces;
                        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));

                        if (!faces.empty()) {
                            cv::Rect face = faces[0];
                            float face_size_ratio = static_cast<float>(face.width * face.height) / (frame.cols * frame.rows);
                            average_face_size = (average_face_size * face_size_samples + face_size_ratio) / (face_size_samples + 1);
                            face_size_samples++;
                        }
                    }
                } else {
                    // Reset consecutive detection counter
                    consecutive_face_detections = 0;
                }
            }
            frame_count++;
        }

        analysis.frame_count = frame_count;

        // Anti-spoofing check: Ensure face was detected consistently
        float face_detection_rate = static_cast<float>(total_frames_with_face) / (frame_count / skip_frames);
        if (face_detection_rate < 0.7f) {
            setFailureReason(analysis, "insufficient_face_detection_rate");
            return analysis;
        }

        // Anti-spoofing check: Face size should be reasonable (not too small or too large)
        if (face_size_samples > 0 && (average_face_size < 0.01f || average_face_size > 0.5f)) {
            setFailureReason(analysis, "face_size_unusual");
            return analysis;
        }

        // Check if we have enough pose data
        if (pose_movements.size() < 10) {
            setFailureReason(analysis, "insufficient_pose_data");
            return analysis;
        }

        // Find the FIRST MAJOR MOVEMENT only
        detectFirstMajorMovement(pose_movements, first_major_movement, analysis);

        // Set analysis results
        if (first_major_movement.direction != MovementDirection::NONE) {
            analysis.directional_movements.push_back(first_major_movement);
            analysis.has_sufficient_movement = true;
            analysis.is_live = true;
            analysis.confidence = 0.95f; // High confidence for MediaPipe-based detection

            // Count movements (only one in this case)
            switch (first_major_movement.direction) {
                case MovementDirection::LEFT: analysis.left_movements = 1; break;
                case MovementDirection::RIGHT: analysis.right_movements = 1; break;
                case MovementDirection::UP: analysis.up_movements = 1; break;
                case MovementDirection::DOWN: analysis.down_movements = 1; break;
                default: break;
            }
        } else {
            analysis.has_sufficient_movement = false;
            analysis.is_live = false;
            analysis.confidence = 0.1f;
        }

        // Calculate overall ranges for reporting
        if (!pose_movements.empty()) {
            float min_yaw = pose_movements[0].yaw_angle, max_yaw = pose_movements[0].yaw_angle;
            float min_pitch = pose_movements[0].pitch_angle, max_pitch = pose_movements[0].pitch_angle;

            for (const auto& movement : pose_movements) {
                min_yaw = std::min(min_yaw, movement.yaw_angle);
                max_yaw = std::max(max_yaw, movement.yaw_angle);
                min_pitch = std::min(min_pitch, movement.pitch_angle);
                max_pitch = std::max(max_pitch, movement.pitch_angle);
            }

            analysis.yaw_range = max_yaw - min_yaw;
            analysis.pitch_range = max_pitch - min_pitch;
        }

        std::cout << "MediaPipe FaceMesh analysis complete: " << pose_movements.size()
                  << " pose samples, first major movement: "
                  << (first_major_movement.direction != MovementDirection::NONE ? directionToString(first_major_movement.direction) : "none")
                  << ", confidence: " << analysis.confidence << std::endl;

        return analysis;

    } catch (const std::exception& e) {
        std::cerr << "Error during MediaPipe FaceMesh video analysis: " << e.what() << std::endl;
        setFailureReason(analysis, "error_during_mediapipe_analysis");
        return analysis;
    }
}
#endif // MEDIAPIPE_AVAILABLE

#ifdef MEDIAPIPE_AVAILABLE
void VideoLivenessDetector::detectFirstMajorMovement(const std::vector<HeadPoseMovement>& movements,
                                                    DirectionalMovement& first_movement,
                                                    VideoLivenessAnalysis& analysis) {
    if (movements.size() < 5) {
        return;
    }

    // Calculate baseline pose from first few frames
    float baseline_yaw = 0.0f, baseline_pitch = 0.0f;
    const int baseline_frames = std::min(5, static_cast<int>(movements.size()));

    for (int i = 0; i < baseline_frames; ++i) {
        baseline_yaw += movements[i].yaw_angle;
        baseline_pitch += movements[i].pitch_angle;
    }
    baseline_yaw /= baseline_frames;
    baseline_pitch /= baseline_frames;

    // Look for first major movement deviation from baseline - Enhanced for up/down detection
    const float MAJOR_MOVEMENT_THRESHOLD_YAW = 6.0f;   // degrees (slightly reduced for better sensitivity)
    const float MAJOR_MOVEMENT_THRESHOLD_PITCH = 5.0f; // degrees (reduced for up/down movements)
    const int MIN_MOVEMENT_DURATION = 3; // frames

    for (size_t i = baseline_frames; i < movements.size() - MIN_MOVEMENT_DURATION; ++i) {
        float yaw_deviation = std::abs(movements[i].yaw_angle - baseline_yaw);
        float pitch_deviation = std::abs(movements[i].pitch_angle - baseline_pitch);

        // Check if this is a major movement
        if (yaw_deviation >= MAJOR_MOVEMENT_THRESHOLD_YAW || pitch_deviation >= MAJOR_MOVEMENT_THRESHOLD_PITCH) {
            // Verify movement is sustained for minimum duration
            bool sustained_movement = true;
            MovementDirection direction = MovementDirection::NONE;
            float max_magnitude = 0.0f;

            for (int j = 0; j < MIN_MOVEMENT_DURATION; ++j) {
                float current_yaw_dev = std::abs(movements[i + j].yaw_angle - baseline_yaw);
                float current_pitch_dev = std::abs(movements[i + j].pitch_angle - baseline_pitch);

                if (current_yaw_dev < MAJOR_MOVEMENT_THRESHOLD_YAW / 2 &&
                    current_pitch_dev < MAJOR_MOVEMENT_THRESHOLD_PITCH / 2) {
                    sustained_movement = false;
                    break;
                }

                // Determine dominant direction
                if (current_yaw_dev > current_pitch_dev && current_yaw_dev > max_magnitude) {
                    max_magnitude = current_yaw_dev;
                    direction = (movements[i + j].yaw_angle > baseline_yaw) ? MovementDirection::RIGHT : MovementDirection::LEFT;
                } else if (current_pitch_dev > max_magnitude) {
                    max_magnitude = current_pitch_dev;
                    direction = (movements[i + j].pitch_angle > baseline_pitch) ? MovementDirection::UP : MovementDirection::DOWN;
                }
            }

            if (sustained_movement && direction != MovementDirection::NONE) {
                // Found first major movement
                first_movement.direction = direction;
                first_movement.magnitude = max_magnitude;
                first_movement.start_time = movements[i].timestamp;
                first_movement.end_time = movements[i + MIN_MOVEMENT_DURATION - 1].timestamp;
                break;
            }
        }
    }
}
#endif // MEDIAPIPE_AVAILABLE

#ifdef MEDIAPIPE_AVAILABLE
HeadPoseMovement VideoLivenessDetector::calculateHeadPoseMediaPipeFaceMesh(const cv::Mat& frame, float timestamp) {
    HeadPoseMovement movement;
    movement.timestamp = timestamp;

    try {
        if (!graph) {
            return movement;
        }

        // Convert OpenCV Mat to MediaPipe ImageFrame
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, frame.cols, frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);

        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        frame.copyTo(input_frame_mat);

        // Add packet to the graph
        auto packet = mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(timestamp * 1000000));
        auto status = graph->AddPacketToInputStream("input_video", packet);

        if (!status.ok()) {
            std::cerr << "Failed to add packet to MediaPipe graph: " << status.message() << std::endl;
            return movement;
        }

        // Wait for output and get landmarks
        mediapipe::Packet output_packet;
        status = graph->GetOutputSidePacket("multi_face_landmarks", &output_packet);

        if (status.ok() && !output_packet.IsEmpty()) {
            auto& landmarks_list = output_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

            if (!landmarks_list.empty() && landmarks_list[0].landmark_size() > 0) {
                // Use MediaPipe face landmarks to calculate accurate head pose
                movement = calculateHeadPoseFromFaceMesh(landmarks_list[0], frame.size());
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error in MediaPipe FaceMesh head pose calculation: " << e.what() << std::endl;
    }

    return movement;
}
#endif // MEDIAPIPE_AVAILABLE

#ifdef MEDIAPIPE_AVAILABLE
HeadPoseMovement VideoLivenessDetector::calculateHeadPoseFromFaceMesh(const mediapipe::NormalizedLandmarkList& landmarks,
                                                                       const cv::Size& image_size) {
    HeadPoseMovement movement;

    try {
        if (landmarks.landmark_size() < 468) { // MediaPipe FaceMesh has 468 landmarks
            return movement;
        }

        // Key landmarks for head pose estimation
        // Nose tip (landmark 1)
        // Chin (landmark 152)
        // Left eye corner (landmark 33)
        // Right eye corner (landmark 263)
        // Left mouth corner (landmark 61)
        // Right mouth corner (landmark 291)

        std::vector<cv::Point3f> model_points = {
            cv::Point3f(0.0f, 0.0f, 0.0f),        // Nose tip
            cv::Point3f(0.0f, -330.0f, -65.0f),   // Chin
            cv::Point3f(-225.0f, 170.0f, -135.0f), // Left eye left corner
            cv::Point3f(225.0f, 170.0f, -135.0f),  // Right eye right corner
            cv::Point3f(-150.0f, -150.0f, -125.0f), // Left mouth corner
            cv::Point3f(150.0f, -150.0f, -125.0f)   // Right mouth corner
        };

        std::vector<cv::Point2f> image_points;
        std::vector<int> landmark_indices = {1, 152, 33, 263, 61, 291};

        for (int idx : landmark_indices) {
            if (idx < landmarks.landmark_size()) {
                const auto& landmark = landmarks.landmark(idx);
                float x = landmark.x() * image_size.width;
                float y = landmark.y() * image_size.height;
                image_points.push_back(cv::Point2f(x, y));
            }
        }

        if (image_points.size() == 6) {
            // Camera intrinsic parameters
            double focal_length = image_size.width;
            cv::Point2d center = cv::Point2d(image_size.width / 2, image_size.height / 2);

            cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
                focal_length, 0, center.x,
                0, focal_length, center.y,
                0, 0, 1);

            cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);

            // Solve PnP
            cv::Mat rotation_vector, translation_vector;
            bool success = cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                                       rotation_vector, translation_vector);

            if (success) {
                // Convert to Euler angles
                cv::Mat rotation_matrix;
                cv::Rodrigues(rotation_vector, rotation_matrix);

                double sy = sqrt(rotation_matrix.at<double>(0,0) * rotation_matrix.at<double>(0,0) +
                                rotation_matrix.at<double>(1,0) * rotation_matrix.at<double>(1,0));

                double x, y, z;
                if (sy < 1e-6) {
                    x = atan2(-rotation_matrix.at<double>(1,2), rotation_matrix.at<double>(1,1));
                    y = atan2(-rotation_matrix.at<double>(2,0), sy);
                    z = 0;
                } else {
                    x = atan2(rotation_matrix.at<double>(2,1), rotation_matrix.at<double>(2,2));
                    y = atan2(-rotation_matrix.at<double>(2,0), sy);
                    z = atan2(rotation_matrix.at<double>(1,0), rotation_matrix.at<double>(0,0));
                }

                // Convert to degrees
                movement.pitch_angle = static_cast<float>(x * 180.0 / CV_PI);
                movement.yaw_angle = static_cast<float>(y * 180.0 / CV_PI);
                movement.roll_angle = static_cast<float>(z * 180.0 / CV_PI);

                // Normalize angles
                auto normalizeAngle = [](float angle) {
                    while (angle > 180.0f) angle -= 360.0f;
                    while (angle < -180.0f) angle += 360.0f;
                    return angle;
                };

                movement.pitch_angle = normalizeAngle(movement.pitch_angle);
                movement.yaw_angle = normalizeAngle(movement.yaw_angle);
                movement.roll_angle = normalizeAngle(movement.roll_angle);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error calculating head pose from FaceMesh landmarks: " << e.what() << std::endl;
    }

    return movement;
}
#endif // MEDIAPIPE_AVAILABLE

#ifdef MEDIAPIPE_AVAILABLE
bool VideoLivenessDetector::setupMediaPipeGraphForAnalysis() {
    try {
        if (graph) {
            graph->CloseInputStream("input_video").IgnoreError();
            graph->WaitUntilDone().IgnoreError();
        }

        // MediaPipe graph configuration for face mesh
        graph_config = R"(
            input_stream: "input_video"
            output_stream: "multi_face_landmarks"

            node {
              calculator: "FaceMeshCpu"
              input_stream: "IMAGE:input_video"
              output_stream: "MULTI_FACE_LANDMARKS:multi_face_landmarks"
            }
        )";

        // Initialize the graph
        graph = std::make_unique<mediapipe::CalculatorGraph>();

        mediapipe::CalculatorGraphConfig config;
        if (!mediapipe::ParseTextProto<mediapipe::CalculatorGraphConfig>(graph_config, &config)) {
            std::cerr << "Failed to parse MediaPipe graph config" << std::endl;
            return false;
        }

        auto status = graph->Initialize(config);
        if (!status.ok()) {
            std::cerr << "Failed to initialize MediaPipe graph: " << status.message() << std::endl;
            return false;
        }

        status = graph->StartRun({});
        if (!status.ok()) {
            std::cerr << "Failed to start MediaPipe graph: " << status.message() << std::endl;
            return false;
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception setting up MediaPipe graph: " << e.what() << std::endl;
        return false;
    }
}
#endif // MEDIAPIPE_AVAILABLE

#ifdef MEDIAPIPE_AVAILABLE
bool VideoLivenessDetector::setupMediaPipeGraph() {
    try {
        // MediaPipe graph configuration for face mesh
        graph_config = R"(
            input_stream: "input_video"
            output_stream: "multi_face_landmarks"
            
            node {
              calculator: "FaceMeshCpu"
              input_stream: "IMAGE:input_video"
              output_stream: "MULTI_FACE_LANDMARKS:multi_face_landmarks"
            }
        )";
        
        // Initialize the graph
        graph = std::make_unique<mediapipe::CalculatorGraph>();
        
        mediapipe::CalculatorGraphConfig config;
        if (!mediapipe::ParseTextProto<mediapipe::CalculatorGraphConfig>(graph_config, &config)) {
            std::cerr << "Failed to parse MediaPipe graph config" << std::endl;
            return false;
        }
        
        auto status = graph->Initialize(config);
        if (!status.ok()) {
            std::cerr << "Failed to initialize MediaPipe graph: " << status.message() << std::endl;
            return false;
        }
        
        status = graph->StartRun({});
        if (!status.ok()) {
            std::cerr << "Failed to start MediaPipe graph: " << status.message() << std::endl;
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception setting up MediaPipe graph: " << e.what() << std::endl;
        return false;
    }
}
#endif // MEDIAPIPE_AVAILABLE

VideoLivenessAnalysis VideoLivenessDetector::analyzeSingleVideoWithOpenCV(const std::string& video_url) {
    VideoLivenessAnalysis analysis;

    if (!initialized) {
        setFailureReason(analysis, "detector_not_initialized");
        return analysis;
    }

    try {
        // Create temporary filename with thread ID to avoid collisions
        std::hash<std::thread::id> hasher;
        std::string temp_filename = "/tmp/gateman_opencv_video_" + std::to_string(std::time(nullptr)) + "_" + std::to_string(hasher(std::this_thread::get_id())) + ".mp4";

        // Download video
        cv::Mat downloaded = downloadVideoFromUrl(video_url, temp_filename);
        if (downloaded.empty()) {
            setFailureReason(analysis, "failed_to_download_video");
            return analysis;
        }

        // Analyze the downloaded video using OpenCV-based approach
        analysis = analyzeVideoWithOpenCVOptimized(temp_filename);

        // Clean up temporary file
        std::filesystem::remove(temp_filename);

        return analysis;

    } catch (const std::exception& e) {
        std::cerr << "Error analyzing video with OpenCV: " << e.what() << std::endl;
        setFailureReason(analysis, "error_processing_video_opencv");
        return analysis;
    }
}

VideoLivenessAnalysis VideoLivenessDetector::analyzeVideoWithOpenCVOptimized(const std::string& video_path) {
    VideoLivenessAnalysis analysis;

    if (!initialized) {
        setFailureReason(analysis, "detector_not_initialized");
        return analysis;
    }

    try {
        // Open video file
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            setFailureReason(analysis, "failed_to_open_video_file");
            return analysis;
        }

        // Get video properties
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        analysis.duration_seconds = static_cast<float>(total_frames / fps);

        std::cout << "Analyzing video with OpenCV: " << total_frames << " frames, "
                  << analysis.duration_seconds << " seconds, " << fps << " FPS" << std::endl;

        // Check minimum duration (1.5-5 seconds to accommodate various videos)
        if (analysis.duration_seconds < 1.5f) {
            setFailureReason(analysis, "video_too_short_minimum_1_5_seconds");
            return analysis;
        }
        if (analysis.duration_seconds > 5.0f) {
            setFailureReason(analysis, "video_too_long_maximum_5_seconds");
            return analysis;
        }

        // Process frames with OpenCV
        cv::Mat frame;
        int frame_count = 0;
        const int skip_frames = std::max(1, static_cast<int>(fps / 15)); // Process ~15 FPS for better accuracy
        std::vector<HeadPoseMovement> pose_movements;
        DirectionalMovement first_major_movement;

        // Anti-spoofing checks
        int total_frames_with_face = 0;
        float average_face_size = 0.0f;
        int face_size_samples = 0;

        while (cap.read(frame) && frame_count < total_frames) {
            if (frame_count % skip_frames == 0) {
                float timestamp = static_cast<float>(frame_count) / fps;

                // Calculate head pose using OpenCV
                HeadPoseMovement movement = calculateHeadPoseOpenCV(frame, timestamp);

                if (movement.yaw_angle != 0.0f || movement.pitch_angle != 0.0f) {
                    pose_movements.push_back(movement);
                    total_frames_with_face++;

                    // Anti-spoofing: Check for reasonable face size
                    cv::Mat gray;
                    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                    cv::equalizeHist(gray, gray);

                    // Simple face detection for size estimation
                    cv::CascadeClassifier face_cascade;
                    std::string cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml";
                    if (!face_cascade.load(cascade_path)) {
                        cascade_path = "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml";
                        face_cascade.load(cascade_path);
                    }

                    if (!face_cascade.empty()) {
                        std::vector<cv::Rect> faces;
                        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));

                        if (!faces.empty()) {
                            cv::Rect face = faces[0];
                            float face_size_ratio = static_cast<float>(face.width * face.height) / (frame.cols * frame.rows);
                            average_face_size = (average_face_size * face_size_samples + face_size_ratio) / (face_size_samples + 1);
                            face_size_samples++;
                        }
                    }
                }
            }
            frame_count++;
        }

        analysis.frame_count = frame_count;

        // Anti-spoofing check: Ensure face was detected consistently
        float face_detection_rate = static_cast<float>(total_frames_with_face) / (frame_count / skip_frames);
        if (face_detection_rate < 0.5f) {
            setFailureReason(analysis, "insufficient_face_detection_rate");
            return analysis;
        }

        // Anti-spoofing check: Face size should be reasonable
        if (face_size_samples > 0 && (average_face_size < 0.005f || average_face_size > 0.7f)) {
            setFailureReason(analysis, "face_size_unusual");
            return analysis;
        }

        // Check if we have enough pose data
        if (pose_movements.size() < 8) {
            setFailureReason(analysis, "insufficient_pose_data");
            return analysis;
        }

        // Find the FIRST MAJOR MOVEMENT only
        detectFirstMajorMovementOpenCV(pose_movements, first_major_movement, analysis);

        // Set analysis results
        if (first_major_movement.direction != MovementDirection::NONE) {
            analysis.directional_movements.push_back(first_major_movement);
            analysis.has_sufficient_movement = true;
            analysis.is_live = true;
            analysis.confidence = 0.85f; // Good confidence for OpenCV-based detection

            // Count movements (only one in this case)
            switch (first_major_movement.direction) {
                case MovementDirection::LEFT: analysis.left_movements = 1; break;
                case MovementDirection::RIGHT: analysis.right_movements = 1; break;
                case MovementDirection::UP: analysis.up_movements = 1; break;
                case MovementDirection::DOWN: analysis.down_movements = 1; break;
                default: break;
            }
        } else {
            analysis.has_sufficient_movement = false;
            analysis.is_live = false;
            analysis.confidence = 0.2f;
        }

        // Calculate overall ranges for reporting
        if (!pose_movements.empty()) {
            float min_yaw = pose_movements[0].yaw_angle, max_yaw = pose_movements[0].yaw_angle;
            float min_pitch = pose_movements[0].pitch_angle, max_pitch = pose_movements[0].pitch_angle;

            for (const auto& movement : pose_movements) {
                min_yaw = std::min(min_yaw, movement.yaw_angle);
                max_yaw = std::max(max_yaw, movement.yaw_angle);
                min_pitch = std::min(min_pitch, movement.pitch_angle);
                max_pitch = std::max(max_pitch, movement.pitch_angle);
            }

            analysis.yaw_range = max_yaw - min_yaw;
            analysis.pitch_range = max_pitch - min_pitch;
        }

        std::cout << "OpenCV analysis complete: " << pose_movements.size()
                  << " pose samples, first major movement: "
                  << (first_major_movement.direction != MovementDirection::NONE ? directionToString(first_major_movement.direction) : "none")
                  << ", confidence: " << analysis.confidence << std::endl;

        return analysis;

    } catch (const std::exception& e) {
        std::cerr << "Error during OpenCV video analysis: " << e.what() << std::endl;
        setFailureReason(analysis, "error_during_opencv_analysis");
        return analysis;
    }
}

void VideoLivenessDetector::detectFirstMajorMovementOpenCV(const std::vector<HeadPoseMovement>& movements,
                                                          DirectionalMovement& first_movement,
                                                          VideoLivenessAnalysis& /* analysis */) {
    if (movements.size() < 5) {
        return;
    }

    // Calculate baseline pose from first few frames
    float baseline_yaw = 0.0f, baseline_pitch = 0.0f;
    const int baseline_frames = std::min(3, static_cast<int>(movements.size()));

    for (int i = 0; i < baseline_frames; ++i) {
        baseline_yaw += movements[i].yaw_angle;
        baseline_pitch += movements[i].pitch_angle;
    }
    baseline_yaw /= baseline_frames;
    baseline_pitch /= baseline_frames;

    // Look for first major movement deviation from baseline - Enhanced for up/down detection
    const float MAJOR_MOVEMENT_THRESHOLD_YAW = 5.0f;   // degrees (optimized for OpenCV)
    const float MAJOR_MOVEMENT_THRESHOLD_PITCH = 4.0f; // degrees (reduced for better up/down sensitivity)
    const int MIN_MOVEMENT_DURATION = 2; // frames (shorter for OpenCV)

    for (size_t i = baseline_frames; i < movements.size() - MIN_MOVEMENT_DURATION; ++i) {
        float yaw_deviation = std::abs(movements[i].yaw_angle - baseline_yaw);
        float pitch_deviation = std::abs(movements[i].pitch_angle - baseline_pitch);

        // Check if this is a major movement
        if (yaw_deviation >= MAJOR_MOVEMENT_THRESHOLD_YAW || pitch_deviation >= MAJOR_MOVEMENT_THRESHOLD_PITCH) {
            // Verify movement is sustained for minimum duration
            bool sustained_movement = true;
            MovementDirection direction = MovementDirection::NONE;
            float max_magnitude = 0.0f;

            for (int j = 0; j < MIN_MOVEMENT_DURATION; ++j) {
                if (i + j >= movements.size()) break;

                float current_yaw_dev = std::abs(movements[i + j].yaw_angle - baseline_yaw);
                float current_pitch_dev = std::abs(movements[i + j].pitch_angle - baseline_pitch);

                if (current_yaw_dev < MAJOR_MOVEMENT_THRESHOLD_YAW / 3 &&
                    current_pitch_dev < MAJOR_MOVEMENT_THRESHOLD_PITCH / 3) {
                    sustained_movement = false;
                    break;
                }

                // Determine dominant direction
                if (current_yaw_dev > current_pitch_dev && current_yaw_dev > max_magnitude) {
                    max_magnitude = current_yaw_dev;
                    direction = (movements[i + j].yaw_angle > baseline_yaw) ? MovementDirection::RIGHT : MovementDirection::LEFT;
                } else if (current_pitch_dev > max_magnitude) {
                    max_magnitude = current_pitch_dev;
                    direction = (movements[i + j].pitch_angle > baseline_pitch) ? MovementDirection::UP : MovementDirection::DOWN;
                }
            }

            if (sustained_movement && direction != MovementDirection::NONE) {
                // Found first major movement
                first_movement.direction = direction;
                first_movement.magnitude = max_magnitude;
                first_movement.start_time = movements[i].timestamp;
                first_movement.end_time = movements[std::min(i + MIN_MOVEMENT_DURATION - 1, movements.size() - 1)].timestamp;
                break;
            }
        }
    }
}

HeadPoseMovement VideoLivenessDetector::calculateHeadPoseMediaPipe(const cv::Mat& frame, float timestamp) {
#ifdef MEDIAPIPE_AVAILABLE
    HeadPoseMovement movement;
    movement.timestamp = timestamp;
    
    try {
        // Convert OpenCV Mat to MediaPipe ImageFrame
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, frame.cols, frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        frame.copyTo(input_frame_mat);
        
        // Add packet to the graph
        auto packet = mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(timestamp * 1000000));
        auto status = graph->AddPacketToInputStream("input_video", packet);
        
        if (!status.ok()) {
            std::cerr << "Failed to add packet to MediaPipe graph: " << status.message() << std::endl;
            return movement;
        }
        
        // Get output packet
        mediapipe::Packet output_packet;
        status = graph->GetOutputSidePacket("multi_face_landmarks", &output_packet);
        
        if (status.ok() && !output_packet.IsEmpty()) {
            auto& landmarks = output_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
            
            if (!landmarks.empty() && landmarks[0].landmark_size() > 0) {
                // Use MediaPipe landmarks to calculate more accurate head pose
                // This would require implementing the full MediaPipe face mesh processing
                // For now, fall back to the OpenCV method
                return calculateHeadPoseOpenCV(frame, timestamp);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in MediaPipe head pose calculation: " << e.what() << std::endl;
    }

    return movement;
#else
    // Fallback to OpenCV when MediaPipe is not available
    return calculateHeadPoseOpenCV(frame, timestamp);
#endif
}

HeadPoseMovement VideoLivenessDetector::calculateHeadPoseMediaPipeDNN(const cv::Mat& frame, float timestamp) {
    HeadPoseMovement movement;
    movement.timestamp = timestamp;

    try {
        // For now, use improved OpenCV face detection with better landmark estimation
        // This is a simpler approach that works reliably
        std::vector<cv::Point2f> landmarks = getFaceLandmarks(frame);

        if (landmarks.size() >= 6) {
            // 3D model points (same as before)
            std::vector<cv::Point3f> model_points = getModel3DPoints();

            // Camera parameters
            cv::Mat camera_matrix = getCameraMatrix(frame.size());
            cv::Mat dist_coeffs = getDistortionCoefficients();

            // Solve PnP to get rotation and translation vectors
            cv::Mat rotation_vector, translation_vector;
            bool success = cv::solvePnP(model_points, landmarks, camera_matrix, dist_coeffs,
                                       rotation_vector, translation_vector);

            if (success) {
                // Convert rotation vector to Euler angles
                cv::Mat rotation_matrix;
                cv::Rodrigues(rotation_vector, rotation_matrix);

                // Extract Euler angles from rotation matrix
                double sy = sqrt(rotation_matrix.at<double>(0,0) * rotation_matrix.at<double>(0,0) +
                                rotation_matrix.at<double>(1,0) * rotation_matrix.at<double>(1,0));

                bool singular = sy < 1e-6;

                double x, y, z;
                if (!singular) {
                    x = atan2(rotation_matrix.at<double>(2,1), rotation_matrix.at<double>(2,2));
                    y = atan2(-rotation_matrix.at<double>(2,0), sy);
                    z = atan2(rotation_matrix.at<double>(1,0), rotation_matrix.at<double>(0,0));
                } else {
                    x = atan2(-rotation_matrix.at<double>(1,2), rotation_matrix.at<double>(1,1));
                    y = atan2(-rotation_matrix.at<double>(2,0), sy);
                    z = 0;
                }

                // Convert radians to degrees
                movement.pitch_angle = static_cast<float>(x * 180.0 / CV_PI);
                movement.yaw_angle = static_cast<float>(y * 180.0 / CV_PI);
                movement.roll_angle = static_cast<float>(z * 180.0 / CV_PI);

                // Normalize angles to [-180, 180] range
                auto normalizeAngle = [](float angle) {
                    while (angle > 180.0f) angle -= 360.0f;
                    while (angle < -180.0f) angle += 360.0f;
                    return angle;
                };

                movement.pitch_angle = normalizeAngle(movement.pitch_angle);
                movement.yaw_angle = normalizeAngle(movement.yaw_angle);
                movement.roll_angle = normalizeAngle(movement.roll_angle);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error in MediaPipe DNN head pose calculation: " << e.what() << std::endl;
    }

    return movement;
}
