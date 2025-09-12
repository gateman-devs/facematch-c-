#include "lightweight_video_detector.hpp"
#include <iostream>
#include <fstream>
#include <curl/curl.h>
#include <filesystem>
#include <thread>
#include <future>
#include <numeric>
#include <algorithm>
#include <cmath>

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
        std::cout << "Initializing Lightweight Video Detector..." << std::endl;
        
        // Load BlazeFace model
        std::string blazeface_path = models_dir + "/blazeface.tflite";
        if (!loadBlazeFaceModel(blazeface_path)) {
            // Try to download if not exists
            if (!std::filesystem::exists(blazeface_path)) {
                std::cout << "BlazeFace model not found. Downloading..." << std::endl;
                std::string download_cmd = "wget -O " + blazeface_path + 
                    " https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
                system(download_cmd.c_str());
                
                if (!loadBlazeFaceModel(blazeface_path)) {
                    std::cerr << "Failed to load BlazeFace model" << std::endl;
                    return false;
                }
            } else {
                std::cerr << "Failed to load BlazeFace model from: " << blazeface_path << std::endl;
                return false;
            }
        }
        
        // Load anti-spoof model
        std::string antispoof_path = models_dir + "/antispoof_mini.tflite";
        if (!loadAntiSpoofModel(antispoof_path)) {
            // For now, we'll continue without anti-spoofing
            std::cout << "Warning: Anti-spoof model not found. Continuing without anti-spoofing." << std::endl;
        }
        
        initialized_ = true;
        std::cout << "Lightweight Video Detector initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing detector: " << e.what() << std::endl;
        return false;
    }
}

bool LightweightVideoDetector::loadBlazeFaceModel(const std::string& model_path) {
    try {
        // Load model
        blazeface_model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        if (!blazeface_model_) {
            return false;
        }
        
        // Build interpreter
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*blazeface_model_, resolver);
        builder(&blazeface_interpreter_);
        
        if (!blazeface_interpreter_) {
            return false;
        }
        
        // Allocate tensors
        blazeface_interpreter_->AllocateTensors();
        
        // Get input details
        int input_tensor = blazeface_interpreter_->inputs()[0];
        TfLiteIntArray* input_dims = blazeface_interpreter_->tensor(input_tensor)->dims;
        
        blazeface_info_.input_height = 128;  // BlazeFace uses 128x128
        blazeface_info_.input_width = 128;
        blazeface_info_.input_channels = 3;
        blazeface_info_.input_normalized = true;
        
        std::cout << "BlazeFace model loaded successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading BlazeFace model: " << e.what() << std::endl;
        return false;
    }
}

bool LightweightVideoDetector::loadAntiSpoofModel(const std::string& model_path) {
    try {
        if (!std::filesystem::exists(model_path)) {
            return false;
        }
        
        antispoof_model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        if (!antispoof_model_) {
            return false;
        }
        
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*antispoof_model_, resolver);
        builder(&antispoof_interpreter_);
        
        if (!antispoof_interpreter_) {
            return false;
        }
        
        antispoof_interpreter_->AllocateTensors();
        
        antispoof_info_.input_height = 80;  // Typical anti-spoof model size
        antispoof_info_.input_width = 80;
        antispoof_info_.input_channels = 3;
        antispoof_info_.input_normalized = true;
        
        std::cout << "Anti-spoof model loaded successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading anti-spoof model: " << e.what() << std::endl;
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
    
    try {
        // Extract frames from video
        std::vector<cv::Mat> frames = extractFrames(video_path_or_url);
        
        if (frames.empty()) {
            analysis.error_message = "Failed to extract frames from video";
            return analysis;
        }
        
        std::vector<FrameAnalysis> frame_analyses;
        std::vector<float> spoof_scores;
        
        // Analyze frames with sampling
        int frames_to_analyze = std::min(static_cast<int>(frames.size()), max_frames_to_analyze_);
        int step = std::max(1, static_cast<int>(frames.size()) / frames_to_analyze);
        
        for (size_t i = 0; i < frames.size(); i += step) {
            FrameAnalysis frame_analysis;
            frame_analysis.timestamp = static_cast<float>(i) / 30.0f;  // Assume 30fps
            
            // Detect face
            auto detection = detectFace(frames[i]);
            
            if (detection.has_value()) {
                frame_analysis.has_face = true;
                frame_analysis.face = detection.value();
                frame_analysis.face_center = detection->getCenter();
                analysis.frames_with_face++;
                
                // Check anti-spoofing only on key frames to save resources
                if (antispoof_interpreter_ && (i == 0 || i == frames.size() / 2 || i == frames.size() - 1)) {
                    float spoof_score = checkAntiSpoof(frames[i], detection.value());
                    frame_analysis.spoof_score = spoof_score;
                    spoof_scores.push_back(spoof_score);
                }
            }
            
            frame_analyses.push_back(frame_analysis);
            analysis.total_frames_analyzed++;
        }
        
        // Calculate movements between consecutive frames with faces
        for (size_t i = 1; i < frame_analyses.size(); i++) {
            if (frame_analyses[i-1].has_face && frame_analyses[i].has_face) {
                Movement movement = calculateMovement(frame_analyses[i-1], frame_analyses[i]);
                if (movement.confidence > 0.1f) {  // Only track confident movements
                    analysis.movements.push_back(movement);
                }
            }
        }
        
        // Determine primary direction
        analysis.primary_direction = getPrimaryDirection(analysis.movements);
        
        // Calculate average spoof score
        if (!spoof_scores.empty()) {
            analysis.average_spoof_score = std::accumulate(spoof_scores.begin(), spoof_scores.end(), 0.0f) / spoof_scores.size();
            analysis.is_live = analysis.average_spoof_score > spoof_threshold_;
        } else {
            // If no anti-spoof model, rely on movement patterns
            analysis.is_live = analysis.frames_with_face > 0 && !analysis.movements.empty();
            analysis.average_spoof_score = analysis.is_live ? 0.7f : 0.3f;
        }
        
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
    
    // Extract frames with sampling
    cv::Mat frame;
    int frame_count = 0;
    
    while (cap.read(frame)) {
        if (frame_count % frame_sample_rate_ == 0) {
            // Resize frame to reduce memory usage
            cv::Mat resized;
            if (frame.cols > 640 || frame.rows > 480) {
                float scale = std::min(640.0f / frame.cols, 480.0f / frame.rows);
                cv::resize(frame, resized, cv::Size(), scale, scale);
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
    
    std::cout << "Extracted " << frames.size() << " frames from video" << std::endl;
    return frames;
}

cv::Mat LightweightVideoDetector::downloadVideo(const std::string& url, std::string& temp_path) {
    try {
        // Generate unique temp filename
        temp_path = "/tmp/video_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".mp4";
        
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
        
        CURLcode res = curl_easy_perform(curl);
        file.close();
        curl_easy_cleanup(curl);
        
        if (res != CURLE_OK) {
            std::filesystem::remove(temp_path);
            temp_path.clear();
            return cv::Mat();
        }
        
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
    if (!blazeface_interpreter_) {
        // Fallback to simple center detection if model not loaded
        FaceDetection detection;
        float width = frame.cols * 0.3f;
        float height = frame.rows * 0.4f;
        detection.bbox = cv::Rect2f(
            (frame.cols - width) / 2,
            (frame.rows - height) / 2,
            width,
            height
        );
        detection.confidence = 0.5f;
        
        // Estimate keypoints
        float cx = detection.bbox.x + detection.bbox.width / 2;
        float cy = detection.bbox.y + detection.bbox.height / 2;
        detection.keypoints[0] = cv::Point2f(cx - width * 0.15f, cy - height * 0.1f);  // Right eye
        detection.keypoints[1] = cv::Point2f(cx + width * 0.15f, cy - height * 0.1f);  // Left eye
        detection.keypoints[2] = cv::Point2f(cx, cy);  // Nose
        detection.keypoints[3] = cv::Point2f(cx, cy + height * 0.15f);  // Mouth
        detection.keypoints[4] = cv::Point2f(cx - width * 0.3f, cy);  // Right ear
        detection.keypoints[5] = cv::Point2f(cx + width * 0.3f, cy);  // Left ear
        
        return detection;
    }
    
    try {
        // Preprocess frame for BlazeFace
        cv::Mat input = preprocessForBlazeFace(frame);
        
        // Copy input to interpreter
        float* input_data = blazeface_interpreter_->typed_input_tensor<float>(0);
        std::memcpy(input_data, input.data, input.total() * input.elemSize());
        
        // Run inference
        if (blazeface_interpreter_->Invoke() != kTfLiteOk) {
            return std::nullopt;
        }
        
        // Get outputs (simplified - actual BlazeFace has more complex output)
        float* scores = blazeface_interpreter_->typed_output_tensor<float>(0);
        float* boxes = blazeface_interpreter_->typed_output_tensor<float>(1);
        
        // Find best detection
        FaceDetection best_detection;
        float best_score = 0.3f;  // Minimum confidence threshold
        
        // Simplified parsing - actual implementation would need proper anchor decoding
        if (scores[0] > best_score) {
            best_detection.confidence = scores[0];
            
            // Convert normalized coordinates to pixel coordinates
            best_detection.bbox.x = boxes[0] * frame.cols;
            best_detection.bbox.y = boxes[1] * frame.rows;
            best_detection.bbox.width = (boxes[2] - boxes[0]) * frame.cols;
            best_detection.bbox.height = (boxes[3] - boxes[1]) * frame.rows;
            
            // Estimate keypoints (simplified)
            float cx = best_detection.bbox.x + best_detection.bbox.width / 2;
            float cy = best_detection.bbox.y + best_detection.bbox.height / 2;
            float w = best_detection.bbox.width;
            float h = best_detection.bbox.height;
            
            best_detection.keypoints[0] = cv::Point2f(cx - w * 0.15f, cy - h * 0.1f);
            best_detection.keypoints[1] = cv::Point2f(cx + w * 0.15f, cy - h * 0.1f);
            best_detection.keypoints[2] = cv::Point2f(cx, cy);
            best_detection.keypoints[3] = cv::Point2f(cx, cy + h * 0.15f);
            best_detection.keypoints[4] = cv::Point2f(cx - w * 0.3f, cy);
            best_detection.keypoints[5] = cv::Point2f(cx + w * 0.3f, cy);
            
            return best_detection;
        }
        
        return std::nullopt;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in face detection: " << e.what() << std::endl;
        return std::nullopt;
    }
}

cv::Mat LightweightVideoDetector::preprocessForBlazeFace(const cv::Mat& frame) {
    cv::Mat processed;
    
    // Resize to model input size
    cv::resize(frame, processed, cv::Size(blazeface_info_.input_width, blazeface_info_.input_height));
    
    // Convert to RGB if needed
    if (processed.channels() == 4) {
        cv::cvtColor(processed, processed, cv::COLOR_BGRA2RGB);
    } else if (processed.channels() == 1) {
        cv::cvtColor(processed, processed, cv::COLOR_GRAY2RGB);
    } else if (processed.channels() == 3) {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    }
    
    // Convert to float and normalize
    processed.convertTo(processed, CV_32FC3, 1.0f / 255.0f);
    
    return processed;
}

float LightweightVideoDetector::checkAntiSpoof(const cv::Mat& frame, const FaceDetection& face) {
    if (!antispoof_interpreter_) {
        // Simple heuristic-based check as fallback
        cv::Rect roi(
            std::max(0, static_cast<int>(face.bbox.x)),
            std::max(0, static_cast<int>(face.bbox.y)),
            std::min(frame.cols - static_cast<int>(face.bbox.x), static_cast<int>(face.bbox.width)),
            std::min(frame.rows - static_cast<int>(face.bbox.y), static_cast<int>(face.bbox.height))
        );
        
        cv::Mat face_region = frame(roi);
        
        // Check for texture patterns indicating real face
        cv::Mat gray;
        cv::cvtColor(face_region, gray, cv::COLOR_BGR2GRAY);
        
        cv::Mat laplacian;
        cv::Laplacian(gray, laplacian, CV_64F);
        
        cv::Scalar mean, stddev;
        cv::meanStdDev(laplacian, mean, stddev);
        
        // Higher variance typically indicates real face
        float variance = stddev[0] * stddev[0];
        float score = std::min(1.0f, variance / 500.0f);
        
        return score;
    }
    
    try {
        cv::Mat input = preprocessForAntiSpoof(frame, face.bbox);
        
        float* input_data = antispoof_interpreter_->typed_input_tensor<float>(0);
        std::memcpy(input_data, input.data, input.total() * input.elemSize());
        
        if (antispoof_interpreter_->Invoke() != kTfLiteOk) {
            return 0.5f;  // Uncertain
        }
        
        float* output = antispoof_interpreter_->typed_output_tensor<float>(0);
        return output[0];  // Assuming single output score
        
    } catch (const std::exception& e) {
        std::cerr << "Error in anti-spoof check: " << e.what() << std::endl;
        return 0.5f;
    }
}

cv::Mat LightweightVideoDetector::preprocessForAntiSpoof(const cv::Mat& frame, const cv::Rect2f& face_rect) {
    // Extract and pad face region
    int padding = 20;
    cv::Rect expanded_rect(
        std::max(0, static_cast<int>(face_rect.x - padding)),
        std::max(0, static_cast<int>(face_rect.y - padding)),
        std::min(frame.cols - static_cast<int>(face_rect.x - padding), static_cast<int>(face_rect.width + 2 * padding)),
        std::min(frame.rows - static_cast<int>(face_rect.y - padding), static_cast<int>(face_rect.height + 2 * padding))
    );
    
    cv::Mat face_region = frame(expanded_rect);
    
    // Resize to model input
    cv::Mat processed;
    cv::resize(face_region, processed, cv::Size(antispoof_info_.input_width, antispoof_info_.input_height));
    
    // Convert to RGB and normalize
    if (processed.channels() == 3) {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    }
    
    processed.convertTo(processed, CV_32FC3, 1.0f / 255.0f);
    
    return processed;
}

Movement LightweightVideoDetector::calculateMovement(const FrameAnalysis& prev, const FrameAnalysis& curr) {
    Movement movement;
    
    if (!prev.has_face || !curr.has_face) {
        return movement;
    }
    
    // Calculate normalized movement
    float frame_width = 640.0f;  // Assuming normalized frame size
    float frame_height = 480.0f;
    
    movement.delta_x = (curr.face_center.x - prev.face_center.x) / frame_width;
    movement.delta_y = (curr.face_center.y - prev.face_center.y) / frame_height;
    
    // Calculate movement magnitude
    float magnitude = std::sqrt(movement.delta_x * movement.delta_x + movement.delta_y * movement.delta_y);
    
    // Only consider significant movements
    if (magnitude < min_movement_threshold_) {
        return movement;
    }
    
    movement.confidence = std::min(1.0f, magnitude / min_movement_threshold_);
    movement.direction = determineDirection(movement.delta_x, movement.delta_y);
    
    return movement;
}

Direction LightweightVideoDetector::determineDirection(float delta_x, float delta_y) {
    float abs_x = std::abs(delta_x);
    float abs_y = std::abs(delta_y);
    
    // Determine primary axis of movement
    if (abs_x > abs_y) {
        // Horizontal movement
        if (delta_x < -min_movement_threshold_) {
            return Direction::LEFT;
        } else if (delta_x > min_movement_threshold_) {
            return Direction::RIGHT;
        }
    } else {
        // Vertical movement
        if (delta_y < -min_movement_threshold_) {
            return Direction::UP;
        } else if (delta_y > min_movement_threshold_) {
            return Direction::DOWN;
        }
    }
    
    return Direction::NONE;
}

Direction LightweightVideoDetector::getPrimaryDirection(const std::vector<Movement>& movements) {
    if (movements.empty()) {
        return Direction::NONE;
    }
    
    // Count movements in each direction
    std::array<int, 5> direction_counts = {0, 0, 0, 0, 0};
    std::array<float, 5> direction_confidences = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    for (const auto& movement : movements) {
        int idx = static_cast<int>(movement.direction);
        direction_counts[idx]++;
        direction_confidences[idx] += movement.confidence;
    }
    
    // Find direction with highest weighted score
    Direction best_direction = Direction::NONE;
    float best_score = 0.0f;
    
    for (int i = 1; i <= 4; i++) {  // Skip NONE (0)
        if (direction_counts[i] > 0) {
            float score = direction_counts[i] * (direction_confidences[i] / direction_counts[i]);
            if (score > best_score) {
                best_score = score;
                best_direction = static_cast<Direction>(i);
            }
        }
    }
    
    // Require minimum confidence
    if (best_score < 2.0f) {  // At least 2 confident movements
        return Direction::NONE;
    }
    
    return best_direction;
}

cv::Mat LightweightVideoDetector::resizeKeepAspectRatio(const cv::Mat& input, int target_width, int target_height) {
    float scale = std::min(
        static_cast<float>(target_width) / input.cols,
        static_cast<float>(target_height) / input.rows
    );
    
    int new_width = static_cast<int>(input.cols * scale);
    int new_height = static_cast<int>(input.rows * scale);
    
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(new_width, new_height));
    
    // Pad to target size
    cv::Mat padded = cv::Mat::zeros(target_height, target_width, resized.type());
    int x_offset = (target_width - new_width) / 2;
    int y_offset = (target_height - new_height) / 2;
    
    resized.copyTo(padded(cv::Rect(x_offset, y_offset, new_width, new_height)));
    
    return padded;
}

// Concurrent processing implementation
std::vector<VideoAnalysis> ConcurrentVideoProcessor::processVideos(
    LightweightVideoDetector& detector,
    const std::vector<std::string>& video_urls,
    int max_threads
) {
    std::vector<std::future<VideoAnalysis>> futures;
    
    for (const auto& url : video_urls) {
        futures.push_back(std::async(std::launch::async, [&detector, url]() {
            return detector.analyzeVideo(url);
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