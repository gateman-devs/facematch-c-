#include "video_liveness_detector.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <curl/curl.h>
#include <filesystem>

VideoLivenessDetector::VideoLivenessDetector() : initialized(false) {
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
#ifdef MEDIAPIPE_AVAILABLE
        if (setupMediaPipeGraph()) {
            std::cout << "Video liveness detector initialized with MediaPipe" << std::endl;
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

VideoLivenessAnalysis VideoLivenessDetector::analyzeVideoFromUrl(const std::string& video_url) {
    VideoLivenessAnalysis analysis;
    
    if (!initialized) {
        setFailureReason(analysis, "detector_not_initialized");
        return analysis;
    }
    
    try {
        // Create temporary filename
        std::string temp_filename = "/tmp/gateman_video_" + std::to_string(std::time(nullptr)) + ".mp4";
        
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
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            setFailureReason(analysis, "failed_to_open_video_file");
            return analysis;
        }
        
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
        if (analysis.pose_movements.size() < MIN_FRAMES / skip_frames) {
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
        // Simple face detection using Haar cascades
        cv::CascadeClassifier face_cascade;
        
        // Try to load face cascade (this is a fallback implementation)
        std::string cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml";
        if (!face_cascade.load(cascade_path)) {
            // Try alternative paths
            cascade_path = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml";
            if (!face_cascade.load(cascade_path)) {
                cascade_path = "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml";
                face_cascade.load(cascade_path);
            }
        }
        
        if (face_cascade.empty()) {
            return landmarks; // Return empty if cascade not found
        }
        
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));
        
        if (!faces.empty()) {
            cv::Rect face = faces[0]; // Use first detected face
            
            // Estimate landmark positions based on face rectangle
            // This is a simplified approach - in a real implementation,
            // you would use a proper landmark detector like dlib
            float cx = face.x + face.width * 0.5f;
            float cy = face.y + face.height * 0.5f;
            float w = face.width;
            float h = face.height;
            
            // Approximate landmark positions
            landmarks = {
                cv::Point2f(cx, cy - h * 0.1f),           // Nose tip
                cv::Point2f(cx, cy + h * 0.3f),           // Chin
                cv::Point2f(cx - w * 0.2f, cy - h * 0.1f), // Left eye corner
                cv::Point2f(cx + w * 0.2f, cy - h * 0.1f), // Right eye corner
                cv::Point2f(cx - w * 0.15f, cy + h * 0.1f), // Left mouth corner
                cv::Point2f(cx + w * 0.15f, cy + h * 0.1f)  // Right mouth corner
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
        
        // Consider the direction of movement
        // FIXED: Corrected pitch angle interpretation for coordinate system
        // Negative pitch_delta = looking down (nose tilts down)
        // Positive pitch_delta = looking up (nose tilts up)
        if (pitch_delta < -1.0f && down_similarity > PATTERN_SIMILARITY) {
            return MovementDirection::DOWN;
        } else if (pitch_delta > 1.0f && up_similarity > PATTERN_SIMILARITY) {
            return MovementDirection::UP;
        } else if (down_similarity > up_similarity && down_similarity > PATTERN_SIMILARITY) {
            return MovementDirection::DOWN;
        } else if (up_similarity > PATTERN_SIMILARITY) {
            return MovementDirection::UP;
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
        
        // Return a dummy Mat to indicate success
        // The actual video will be read from the file
        return cv::Mat::ones(1, 1, CV_8UC1);
        
    } catch (const std::exception& e) {
        std::cerr << "Error downloading video: " << e.what() << std::endl;
        return cv::Mat();
    }
}

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

HeadPoseMovement VideoLivenessDetector::calculateHeadPoseMediaPipe(const cv::Mat& frame, float timestamp) {
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
}
#endif
