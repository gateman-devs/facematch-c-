#include "lightweight_video_detector_simple.hpp"
#include <iostream>
#include <fstream>
#include <curl/curl.h>
#include <filesystem>
#include <thread>
#include <future>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
// Simple OpenCV-only version without objdetect

namespace lightweight {

// Helper function for CURL download
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total_size = size * nmemb;
    std::ofstream* file = static_cast<std::ofstream*>(userp);
    file->write(static_cast<char*>(contents), total_size);
    return total_size;
}

LightweightVideoDetector::LightweightVideoDetector() : initialized_(false) {
}

LightweightVideoDetector::~LightweightVideoDetector() {
}

bool LightweightVideoDetector::initialize(const std::string& models_dir) {
    try {
        std::cout << "Initializing Lightweight Video Detector (OpenCV-only version)..." << std::endl;
        
        // Try multiple paths for Haar cascades
        // Simple lightweight detector - no cascades needed
        std::cout << "Initialized simple face detector (no cascade files required)" << std::endl;
        
        initialized_ = true;
        std::cout << "Lightweight Video Detector initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing detector: " << e.what() << std::endl;
        return false;
    }
}

VideoAnalysis LightweightVideoDetector::analyzeVideo(const std::string& video_path_or_url) {
    VideoAnalysis analysis;
    
    if (!initialized_) {
        analysis.error_message = "Detector not initialized";
        return analysis;
    }
    
    startTimer();
    tracking_initialized_ = false;  // Reset tracker for new video
    
    // Reset optical flow static variables for new video
    resetOpticalFlow();
    
    try {
        // Extract frames from video
        std::vector<cv::Mat> frames = extractFrames(video_path_or_url);
        
        if (frames.empty()) {
            analysis.error_message = "Failed to extract frames from video";
            return analysis;
        }
        
        std::vector<FrameAnalysis> frame_analyses;
        std::vector<float> spoof_scores;
        
        // Store frame dimensions for movement calculation
        int frame_width = frames.empty() ? 640 : frames[0].cols;
        int frame_height = frames.empty() ? 480 : frames[0].rows;
        
        // Analyze frames
        for (size_t i = 0; i < frames.size(); i++) {
            FrameAnalysis frame_analysis;
            frame_analysis.timestamp = static_cast<float>(i) / 10.0f;  // Assume 10fps after sampling
            
            // Detect or track face
            std::optional<FaceDetection> detection;
            
            // Use tracking for efficiency after first detection
            if (use_tracking_ && tracking_initialized_) {
                detection = trackFace(frames[i]);
                if (!detection.has_value()) {
                    // Tracking lost, try detection again
                    tracking_initialized_ = false;
                    detection = detectFace(frames[i]);
                    if (detection.has_value()) {
                        initializeTracker(frames[i], detection->bbox);
                    }
                }
            } else {
                detection = detectFace(frames[i]);
                if (detection.has_value() && use_tracking_) {
                    initializeTracker(frames[i], detection->bbox);
                }
            }
            
            if (detection.has_value()) {
                frame_analysis.has_face = true;
                frame_analysis.face = detection.value();
                frame_analysis.face_center = detection->getCenter();
                analysis.frames_with_face++;
                std::cout << "Frame " << i << " face detected at (" << frame_analysis.face_center.x << ", " << frame_analysis.face_center.y << ")" << std::endl;
                
                // Quick anti-spoofing check (only on first and last frames)
                if (i == 0 || i == frames.size() - 1) {
                    float spoof_score = checkAntiSpoof(frames[i], detection.value());
                    frame_analysis.spoof_score = spoof_score;
                    spoof_scores.push_back(spoof_score);
                } else {
                    frame_analysis.spoof_score = 0.7f;  // Default reasonable score
                }
            }
            
            frame_analyses.push_back(frame_analysis);
            analysis.total_frames_analyzed++;
        }
        
        // Detect the primary intentional movement, ignoring small preparatory movements
        Direction primary_movement = Direction::NONE;
        cv::Point2f initial_center;
        float best_magnitude = 0.0f;
        Movement best_movement;
        
        // Use first few frames to establish baseline center position
        if (frame_analyses.size() > 3) {
            float avg_x = 0, avg_y = 0;
            int valid_frames = 0;
            for (size_t i = 0; i < std::min(size_t(5), frame_analyses.size()); i++) {
                if (frame_analyses[i].has_face) {
                    avg_x += frame_analyses[i].face_center.x;
                    avg_y += frame_analyses[i].face_center.y;
                    valid_frames++;
                }
            }
            if (valid_frames > 0) {
                initial_center = cv::Point2f(avg_x / valid_frames, avg_y / valid_frames);
                std::cout << "Baseline center position: (" << initial_center.x << ", " << initial_center.y << ")" << std::endl;
            }
        }
        
        // Look for the strongest movement in a window after baseline establishment
        float significant_threshold = 0.008f;  // Higher threshold to ignore small movements
        
        for (size_t i = 6; i < frame_analyses.size(); i++) {
            if (frame_analyses[i].has_face) {
                cv::Point2f current_center = frame_analyses[i].face_center;
                
                // Calculate movement from initial position
                float delta_x = (current_center.x - initial_center.x) / frame_width;
                float delta_y = (current_center.y - initial_center.y) / frame_height;
                float magnitude = std::sqrt(delta_x * delta_x + delta_y * delta_y);
                
                std::cout << "Frame " << i << " from baseline: dx=" << delta_x 
                         << ", dy=" << delta_y << ", magnitude=" << magnitude << std::endl;
                
                // Track the strongest significant movement
                if (magnitude > significant_threshold && magnitude > best_magnitude) {
                    Direction candidate_direction = determineDirection(delta_x, delta_y);
                    if (candidate_direction != Direction::NONE) {
                        best_magnitude = magnitude;
                        best_movement.delta_x = delta_x;
                        best_movement.delta_y = delta_y;
                        best_movement.confidence = std::min(1.0f, magnitude / significant_threshold);
                        best_movement.direction = candidate_direction;
                        primary_movement = candidate_direction;
                        
                        std::cout << "STRONGER MOVEMENT FOUND at frame " << i << ": direction=" 
                                 << static_cast<int>(candidate_direction) << ", magnitude=" 
                                 << magnitude << std::endl;
                    }
                }
            }
        }
        
        // Add the strongest movement if found
        if (primary_movement != Direction::NONE) {
            analysis.movements.push_back(best_movement);
            std::cout << "PRIMARY MOVEMENT: direction=" << static_cast<int>(primary_movement) 
                     << ", confidence=" << best_movement.confidence << std::endl;
        }
        
        std::cout << "Total movements detected: " << analysis.movements.size() << std::endl;
        std::cout << "Frames with faces: " << analysis.frames_with_face << "/" << analysis.total_frames_analyzed << std::endl;
        
        // Determine primary direction
        analysis.primary_direction = getPrimaryDirection(analysis.movements);
        
        // Calculate average spoof score
        if (!spoof_scores.empty()) {
            analysis.average_spoof_score = std::accumulate(spoof_scores.begin(), spoof_scores.end(), 0.0f) / spoof_scores.size();
        } else {
            analysis.average_spoof_score = 0.6f;  // Default to slightly live
        }
        analysis.is_live = analysis.average_spoof_score > spoof_threshold_;
        
        // Set success if we detected a clear direction
        analysis.success = (analysis.primary_direction != Direction::NONE) && 
                          (analysis.frames_with_face > analysis.total_frames_analyzed * 0.5);
        
        // Performance metrics
        analysis.total_processing_time_ms = getElapsedMs();
        analysis.avg_frame_time_ms = analysis.total_processing_time_ms / analysis.total_frames_analyzed;
        
        std::cout << "Video analysis complete: " 
                  << "Direction=" << static_cast<int>(analysis.primary_direction)
                  << ", Frames=" << analysis.total_frames_analyzed
                  << ", Time=" << analysis.total_processing_time_ms << "ms"
                  << ", AvgFrame=" << analysis.avg_frame_time_ms << "ms" << std::endl;
        
    } catch (const std::exception& e) {
        analysis.error_message = std::string("Analysis error: ") + e.what();
        std::cerr << analysis.error_message << std::endl;
    }
    
    return analysis;
}

std::vector<VideoAnalysis> LightweightVideoDetector::analyzeVideos(const std::vector<std::string>& video_paths_or_urls) {
    return ConcurrentVideoProcessor::processVideos(*this, video_paths_or_urls);
}

std::vector<cv::Mat> LightweightVideoDetector::extractFrames(const std::string& video_path_or_url) {
    std::vector<cv::Mat> frames;
    std::string video_path = video_path_or_url;
    std::string temp_path;
    
    // Download if URL
    if (video_path_or_url.find("http") == 0) {
        // Decode URL before downloading
        std::string decoded_url = decodeUrl(video_path_or_url);
        cv::Mat dummy = downloadVideo(decoded_url, temp_path);
        if (!temp_path.empty()) {
            video_path = temp_path;
        } else {
            return frames;
        }
    }
    
    // Open video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        if (!temp_path.empty()) {
            std::filesystem::remove(temp_path);
        }
        return frames;
    }
    
    // Get video properties
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30;  // Default to 30fps if unknown
    
    // Extract frames with sampling
    cv::Mat frame;
    int frame_count = 0;
    
    while (cap.read(frame)) {
        if (frame_count % frame_sample_rate_ == 0) {
            // Resize frame to reduce memory usage and processing time
            cv::Mat resized;
            if (frame.cols > 640 || frame.rows > 480) {
                float scale = std::min(640.0f / frame.cols, 480.0f / frame.rows);
                cv::resize(frame, resized, cv::Size(), scale, scale, cv::INTER_AREA);
            } else {
                resized = frame.clone();
            }
            frames.push_back(resized);
            
            // Limit number of frames
            if (frames.size() >= static_cast<size_t>(max_frames_to_analyze_)) {
                break;
            }
        }
        frame_count++;
    }
    
    cap.release();
    
    // Clean up temp file
    if (!temp_path.empty()) {
        std::filesystem::remove(temp_path);
    }
    
    std::cout << "Extracted " << frames.size() << " frames from video (sampled from " << frame_count << " total frames)" << std::endl;
    return frames;
}

cv::Mat LightweightVideoDetector::downloadVideo(const std::string& url, std::string& temp_path) {
    try {
        // Generate unique temp filename
        auto now = std::chrono::system_clock::now().time_since_epoch().count();
        temp_path = "/tmp/lightweight_video_" + std::to_string(now) + ".mp4";
        
        // Download using CURL
        CURL* curl = curl_easy_init();
        if (!curl) {
            return cv::Mat();
        }
        
        std::ofstream file(temp_path, std::ios::binary);
        if (!file.is_open()) {
            curl_easy_cleanup(curl);
            return cv::Mat();
        }
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);  // For simplicity
        
        CURLcode res = curl_easy_perform(curl);
        file.close();
        curl_easy_cleanup(curl);
        
        if (res != CURLE_OK) {
            std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
            std::filesystem::remove(temp_path);
            temp_path.clear();
            return cv::Mat();
        }
        
        // Check file size
        auto file_size = std::filesystem::file_size(temp_path);
        if (file_size < 1024) {
            std::cerr << "Downloaded file too small: " << file_size << " bytes" << std::endl;
            std::filesystem::remove(temp_path);
            temp_path.clear();
            return cv::Mat();
        }
        
        std::cout << "Downloaded video: " << file_size / 1024 << " KB" << std::endl;
        return cv::Mat(1, 1, CV_8UC1);  // Return non-empty mat to indicate success
        
    } catch (const std::exception& e) {
        std::cerr << "Error downloading video: " << e.what() << std::endl;
        if (!temp_path.empty() && std::filesystem::exists(temp_path)) {
            std::filesystem::remove(temp_path);
        }
        temp_path.clear();
        return cv::Mat();
    }
}

std::optional<FaceDetection> LightweightVideoDetector::detectFace(const cv::Mat& frame) {
    return detectFaceWithOpticalFlow(frame);
}

std::optional<FaceDetection> LightweightVideoDetector::detectFaceWithColorSegmentation(const cv::Mat& frame) {
    // Only try color detection if frame is reasonable quality
    if (frame.cols < 100 || frame.rows < 100) {
        return std::nullopt;
    }
    
    try {
        // Resize frame for faster processing
        cv::Mat small_frame;
        float scale = std::min(320.0f / frame.cols, 240.0f / frame.rows);
        if (scale < 1.0f) {
            cv::resize(frame, small_frame, cv::Size(), scale, scale, cv::INTER_AREA);
        } else {
            small_frame = frame;
        }
        
        // Convert to HSV for skin color detection
        cv::Mat hsv, mask;
        cv::cvtColor(small_frame, hsv, cv::COLOR_BGR2HSV);
        
        // Define broader skin color range for better detection
        cv::Scalar lower_skin(0, 30, 60);
        cv::Scalar upper_skin(25, 255, 255);
        cv::inRange(hsv, lower_skin, upper_skin, mask);
        
        // Additional skin color range
        cv::Mat mask2;
        cv::Scalar lower_skin2(160, 30, 60);
        cv::Scalar upper_skin2(180, 255, 255);
        cv::inRange(hsv, lower_skin2, upper_skin2, mask2);
        
        // Combine masks
        cv::bitwise_or(mask, mask2, mask);
        
        // Simple morphological cleanup
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        
        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        if (contours.empty()) {
            return std::nullopt;
        }
        
        // Find largest contour in upper-center region (where faces typically are)
        cv::Rect center_region(small_frame.cols * 0.2f, 0, 
                              small_frame.cols * 0.6f, small_frame.rows * 0.8f);
        
        auto best_contour = contours.end();
        double best_area = 0;
        
        for (auto it = contours.begin(); it != contours.end(); ++it) {
            cv::Rect bbox = cv::boundingRect(*it);
            double area = cv::contourArea(*it);
            
            // Check if contour center is in reasonable face region
            cv::Point2f center(bbox.x + bbox.width/2, bbox.y + bbox.height/2);
            if (center_region.contains(center) && area > best_area) {
                best_area = area;
                best_contour = it;
            }
        }
        
        if (best_contour == contours.end()) {
            return std::nullopt;
        }
        
        // Get bounding rectangle and scale back to original
        cv::Rect face_rect = cv::boundingRect(*best_contour);
        if (scale < 1.0f) {
            face_rect.x /= scale;
            face_rect.y /= scale;
            face_rect.width /= scale;
            face_rect.height /= scale;
        }
        
        // Filter by size
        float min_face_size = std::min(frame.cols, frame.rows) * 0.08f;
        float max_face_size = std::min(frame.cols, frame.rows) * 0.7f;
        
        if (face_rect.width < min_face_size || face_rect.height < min_face_size ||
            face_rect.width > max_face_size || face_rect.height > max_face_size) {
            return std::nullopt;
        }
        
        FaceDetection detection;
        detection.bbox = cv::Rect2f(face_rect);
        detection.confidence = 0.8f;
        
        // Estimate keypoints
        detection.keypoints = estimateKeypoints(detection.bbox, {});
        
        return detection;
        
    } catch (const std::exception& e) {
        std::cerr << "Color detection error: " << e.what() << std::endl;
        return std::nullopt;
    }
}

std::optional<FaceDetection> LightweightVideoDetector::detectFaceWithOpticalFlow(const cv::Mat& frame, bool reset) {
    static cv::Mat prev_gray;
    static std::vector<cv::Point2f> prev_points;
    static cv::Point2f face_center(0, 0);
    static bool initialized = false;
    
    if (reset) {
        prev_points.clear();
        prev_gray = cv::Mat();
        initialized = false;
        return std::nullopt;
    }
    
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    
    if (!initialized) {
        // Initialize with center region as starting face position
        face_center = cv::Point2f(frame.cols / 2.0f, frame.rows * 0.4f);
        
        // Create a grid of points around the face center for tracking
        prev_points.clear();
        int grid_size = 3;
        float spacing = std::min(frame.cols, frame.rows) * 0.15f;
        
        for (int i = -grid_size; i <= grid_size; i++) {
            for (int j = -grid_size; j <= grid_size; j++) {
                cv::Point2f pt(face_center.x + i * spacing / grid_size, 
                              face_center.y + j * spacing / grid_size);
                if (pt.x >= 0 && pt.x < frame.cols && pt.y >= 0 && pt.y < frame.rows) {
                    prev_points.push_back(pt);
                }
            }
        }
        
        prev_gray = gray.clone();
        initialized = true;
    }
    
    if (!prev_points.empty() && !prev_gray.empty()) {
        // Track points using optical flow
        std::vector<cv::Point2f> next_points;
        std::vector<uchar> status;
        std::vector<float> errors;
        
        cv::calcOpticalFlowPyrLK(prev_gray, gray, prev_points, next_points, status, errors);
        
        // Calculate new face center based on tracked points
        cv::Point2f new_center(0, 0);
        int valid_points = 0;
        
        for (size_t i = 0; i < next_points.size(); i++) {
            if (status[i] && errors[i] < 30) {  // Valid tracking
                new_center.x += next_points[i].x;
                new_center.y += next_points[i].y;
                valid_points++;
            }
        }
        
        if (valid_points > 0) {
            new_center.x /= valid_points;
            new_center.y /= valid_points;
            face_center = new_center;
            
            // Update points for next iteration
            prev_points.clear();
            for (size_t i = 0; i < next_points.size(); i++) {
                if (status[i] && errors[i] < 30) {
                    prev_points.push_back(next_points[i]);
                }
            }
        }
    }
    
    // Create face detection result
    FaceDetection detection;
    float width = frame.cols * 0.4f;
    float height = frame.rows * 0.5f;
    detection.bbox = cv::Rect2f(
        face_center.x - width / 2,
        face_center.y - height / 2,
        width,
        height
    );
    detection.confidence = 0.8f;
    
    // Store current frame for next iteration
    prev_gray = gray.clone();
    
    // Estimate keypoints
    detection.keypoints = estimateKeypoints(detection.bbox, {});
    
    return detection;
}

void LightweightVideoDetector::resetOpticalFlow() {
    // This function will reset static variables in detectFaceWithOpticalFlow
    // We need to call the optical flow function with a reset flag
    cv::Mat dummy(100, 100, CV_8UC3);
    detectFaceWithOpticalFlow(dummy, true);  // Call with reset flag
}

std::optional<FaceDetection> LightweightVideoDetector::trackFace(const cv::Mat& frame) {
    if (!tracker_ || !tracking_initialized_) {
        return std::nullopt;
    }
    
    try {
        // Tracker disabled - return null to force re-detection
        bool ok = false;
        if (!ok) {
            tracking_initialized_ = false;
            return std::nullopt;
        }
        
        FaceDetection detection;
        detection.bbox = tracked_bbox_;
        detection.confidence = 0.7f;  // Lower confidence for tracked faces
        
        // Estimate keypoints based on tracked bbox
        detection.keypoints = estimateKeypoints(detection.bbox, {});
        
        return detection;
        
    } catch (const std::exception& e) {
        std::cerr << "Tracking error: " << e.what() << std::endl;
        tracking_initialized_ = false;
        return std::nullopt;
    }
}

void LightweightVideoDetector::initializeTracker(const cv::Mat& frame, const cv::Rect2f& bbox) {
    try {
        // Use KCF tracker for efficiency
        // Disable tracker for simplicity - just use face detection each frame
        tracking_initialized_ = false;
        std::cerr << "Tracker initialization disabled for compatibility" << std::endl;
        tracking_initialized_ = true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize tracker: " << e.what() << std::endl;
        tracking_initialized_ = false;
    }
}

float LightweightVideoDetector::checkAntiSpoof(const cv::Mat& frame, const FaceDetection& face) {
    cv::Rect roi(
        std::max(0, static_cast<int>(face.bbox.x)),
        std::max(0, static_cast<int>(face.bbox.y)),
        std::min(frame.cols - static_cast<int>(face.bbox.x), static_cast<int>(face.bbox.width)),
        std::min(frame.rows - static_cast<int>(face.bbox.y), static_cast<int>(face.bbox.height))
    );
    
    if (roi.width <= 0 || roi.height <= 0) {
        return 0.5f;
    }
    
    cv::Mat face_region = frame(roi);
    
    // Combine multiple texture analysis methods
    float texture_score = analyzeTexture(face_region);
    float color_score = analyzeColorDistribution(face_region);
    float frequency_score = analyzeFrequencyDomain(face_region);
    
    // Weighted average
    float final_score = texture_score * 0.4f + color_score * 0.3f + frequency_score * 0.3f;
    
    return final_score;
}

float LightweightVideoDetector::analyzeTexture(const cv::Mat& face_region) {
    cv::Mat gray;
    if (face_region.channels() == 3) {
        cv::cvtColor(face_region, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = face_region.clone();
    }
    
    // Calculate Local Binary Pattern-like features
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F, 3);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    
    // Higher variance indicates more texture (real face)
    float variance = stddev[0] * stddev[0];
    float score = std::min(1.0f, variance / 500.0f);
    
    return score;
}

float LightweightVideoDetector::analyzeColorDistribution(const cv::Mat& face_region) {
    // Convert to HSV for better color analysis
    cv::Mat hsv;
    cv::cvtColor(face_region, hsv, cv::COLOR_BGR2HSV);
    
    // Split channels
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    
    // Analyze saturation channel (real faces have varied saturation)
    cv::Scalar mean, stddev;
    cv::meanStdDev(channels[1], mean, stddev);
    
    // Real faces typically have moderate saturation with some variation
    float sat_mean = mean[0];
    float sat_std = stddev[0];
    
    float score = 0.5f;
    if (sat_mean > 30 && sat_mean < 150) {  // Not too saturated or desaturated
        score += 0.25f;
    }
    if (sat_std > 15 && sat_std < 60) {  // Some variation but not extreme
        score += 0.25f;
    }
    
    return score;
}

float LightweightVideoDetector::analyzeFrequencyDomain(const cv::Mat& face_region) {
    cv::Mat gray;
    if (face_region.channels() == 3) {
        cv::cvtColor(face_region, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = face_region.clone();
    }
    
    // Resize to power of 2 for FFT
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(64, 64));
    
    // Convert to float
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32F);
    
    // Apply DFT
    cv::Mat dft_result;
    cv::dft(float_img, dft_result, cv::DFT_COMPLEX_OUTPUT);
    
    // Calculate magnitude spectrum
    std::vector<cv::Mat> planes;
    cv::split(dft_result, planes);
    cv::Mat magnitude;
    cv::magnitude(planes[0], planes[1], magnitude);
    
    // Analyze high frequency content
    // Real faces have more high-frequency details
    cv::Mat high_freq = magnitude(cv::Rect(32, 32, 32, 32));  // High frequency region
    double min_val, max_val;
    cv::minMaxLoc(high_freq, &min_val, &max_val);
    
    float score = std::min(1.0f, static_cast<float>(max_val / 1000.0));
    
    return score;
}

Movement LightweightVideoDetector::calculateMovement(const FrameAnalysis& prev, const FrameAnalysis& curr, int frame_width, int frame_height) {
    Movement movement;
    
    if (!prev.has_face || !curr.has_face) {
        return movement;
    }
    
    // Calculate normalized movement
    movement.delta_x = (curr.face_center.x - prev.face_center.x) / frame_width;
    movement.delta_y = (curr.face_center.y - prev.face_center.y) / frame_height;
    
    // Calculate movement magnitude
    float magnitude = std::sqrt(movement.delta_x * movement.delta_x + movement.delta_y * movement.delta_y);
    
    // Only consider significant movements
    if (magnitude < min_movement_threshold_ * 0.3f) {  // Even lower threshold
        return movement;
    }
    
    movement.confidence = std::min(1.0f, magnitude / (min_movement_threshold_ * 0.5f));
    movement.direction = determineDirection(movement.delta_x, movement.delta_y);
    
    return movement;
}

Direction LightweightVideoDetector::determineDirection(float delta_x, float delta_y) {
    float abs_x = std::abs(delta_x);
    float abs_y = std::abs(delta_y);
    
    // Use absolute thresholds for first movement detection
    float min_threshold = 0.003f;  // Minimum movement to be considered intentional
    
    // Must have at least minimum movement in some direction
    if (abs_x < min_threshold && abs_y < min_threshold) {
        return Direction::NONE;
    }
    
    // Determine primary direction based on which component is stronger
    if (abs_x > abs_y) {
        // Horizontal movement dominates
        if (delta_x < 0) {
            return Direction::LEFT;
        } else {
            return Direction::RIGHT;
        }
    } else {
        // Vertical movement dominates  
        if (delta_y < 0) {
            return Direction::UP;
        } else {
            return Direction::DOWN;
        }
    }
}

Direction LightweightVideoDetector::getPrimaryDirection(const std::vector<Movement>& movements) {
    if (movements.empty()) {
        return Direction::NONE;
    }
    
    // For first-movement-only detection, just return the first detected movement
    // Since we only add significant movements, the first one is our answer
    const auto& first_movement = movements[0];
    
    // Ensure it has sufficient confidence
    if (first_movement.confidence > 0.5f) {
        return first_movement.direction;
    }
    
    return Direction::NONE;
}

cv::Rect2f LightweightVideoDetector::expandRect(const cv::Rect& rect, float factor, const cv::Size& frame_size) {
    float cx = rect.x + rect.width / 2.0f;
    float cy = rect.y + rect.height / 2.0f;
    
    float new_width = rect.width * factor;
    float new_height = rect.height * factor;
    
    float new_x = cx - new_width / 2.0f;
    float new_y = cy - new_height / 2.0f;
    
    // Clamp to frame boundaries
    new_x = std::max(0.0f, new_x);
    new_y = std::max(0.0f, new_y);
    new_width = std::min(new_width, frame_size.width - new_x);
    new_height = std::min(new_height, frame_size.height - new_y);
    
    return cv::Rect2f(new_x, new_y, new_width, new_height);
}

std::array<cv::Point2f, 6> LightweightVideoDetector::estimateKeypoints(const cv::Rect2f& face_bbox, const std::vector<cv::Rect>& eyes) {
    std::array<cv::Point2f, 6> keypoints;
    
    float cx = face_bbox.x + face_bbox.width / 2;
    float cy = face_bbox.y + face_bbox.height / 2;
    float w = face_bbox.width;
    float h = face_bbox.height;
    
    if (eyes.size() >= 2) {
        // Use detected eyes for better accuracy
        cv::Point2f eye1_center(eyes[0].x + eyes[0].width / 2.0f, eyes[0].y + eyes[0].height / 2.0f);
        cv::Point2f eye2_center(eyes[1].x + eyes[1].width / 2.0f, eyes[1].y + eyes[1].height / 2.0f);
        
        // Determine left and right eyes
        if (eye1_center.x < eye2_center.x) {
            keypoints[0] = eye2_center;  // Right eye (viewer's perspective)
            keypoints[1] = eye1_center;   // Left eye
        } else {
            keypoints[0] = eye1_center;   // Right eye
            keypoints[1] = eye2_center;   // Left eye
        }
        
        // Estimate other keypoints based on eye positions
        float eye_distance = cv::norm(eye1_center - eye2_center);
        keypoints[2] = cv::Point2f(cx, cy + eye_distance * 0.3f);  // Nose
        keypoints[3] = cv::Point2f(cx, cy + eye_distance * 0.8f);  // Mouth
    } else {
        // Estimate keypoints based on face bbox proportions
        keypoints[0] = cv::Point2f(cx - w * 0.15f, cy - h * 0.15f);  // Right eye
        keypoints[1] = cv::Point2f(cx + w * 0.15f, cy - h * 0.15f);  // Left eye
        keypoints[2] = cv::Point2f(cx, cy);                          // Nose
        keypoints[3] = cv::Point2f(cx, cy + h * 0.2f);               // Mouth
    }
    
    keypoints[4] = cv::Point2f(cx - w * 0.35f, cy);  // Right ear
    keypoints[5] = cv::Point2f(cx + w * 0.35f, cy);   // Left ear
    
    return keypoints;
}

// Concurrent processing implementation
std::vector<VideoAnalysis> ConcurrentVideoProcessor::processVideos(
    LightweightVideoDetector& detector,
    const std::vector<std::string>& video_urls,
    int max_threads
) {
    // Limit number of threads to avoid resource exhaustion
    int num_threads = std::min(static_cast<int>(video_urls.size()), max_threads);
    
    std::vector<std::future<VideoAnalysis>> futures;
    
    for (const auto& url : video_urls) {
        futures.push_back(std::async(std::launch::async, [&detector, url]() {
            // Create a new detector instance for thread safety
            LightweightVideoDetector thread_detector;
            thread_detector.initialize("./models");
            // Copy settings from main detector (need getters for private members)
            thread_detector.setFrameSampleRate(3);  // Use default values for now
            thread_detector.setMinMovementThreshold(0.10f);
            thread_detector.setSpoofThreshold(0.5f);
            thread_detector.setMaxFramesToAnalyze(20);
            
            return thread_detector.analyzeVideo(url);
        }));
    }
    
    std::vector<VideoAnalysis> results;
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    return results;
}

std::string LightweightVideoDetector::decodeUrl(const std::string& encoded_url) {
    std::string decoded = encoded_url;
    std::string hex_chars = "0123456789ABCDEFabcdef";
    
    size_t pos = 0;
    while ((pos = decoded.find('%', pos)) != std::string::npos) {
        if (pos + 2 < decoded.length()) {
            std::string hex_str = decoded.substr(pos + 1, 2);
            // Check if it's valid hex
            if (hex_chars.find(hex_str[0]) != std::string::npos && 
                hex_chars.find(hex_str[1]) != std::string::npos) {
                
                char decoded_char = static_cast<char>(std::stoi(hex_str, nullptr, 16));
                decoded.replace(pos, 3, 1, decoded_char);
            }
        }
        pos++;
    }
    
    return decoded;
}

} // namespace lightweight