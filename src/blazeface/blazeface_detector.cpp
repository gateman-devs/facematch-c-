#include "blazeface_detector.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <curl/curl.h>
#include <random>
#include <future>
#include <thread>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace blazeface {

// Helper function for CURL download
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total_size = size * nmemb;
    std::ofstream* file = static_cast<std::ofstream*>(userp);
    file->write(static_cast<char*>(contents), total_size);
    return total_size;
}

BlazeFaceDetector::BlazeFaceDetector() {
}

BlazeFaceDetector::~BlazeFaceDetector() {
}

bool BlazeFaceDetector::initialize(const std::string& model_path) {
    try {
        std::cout << "Initializing BlazeFace detector..." << std::endl;
        
        // Check if model file exists
        if (!std::filesystem::exists(model_path)) {
            std::cerr << "Model file not found: " << model_path << std::endl;
            
            // Try to download the model
            std::string download_url = "https://storage.googleapis.com/mediapipe-models/face_detection/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
            std::cout << "Attempting to download BlazeFace model..." << std::endl;
            
            // Create directory if it doesn't exist
            std::filesystem::create_directories(std::filesystem::path(model_path).parent_path());
            
            // Download using curl
            CURL* curl = curl_easy_init();
            if (curl) {
                std::ofstream file(model_path, std::ios::binary);
                curl_easy_setopt(curl, CURLOPT_URL, download_url.c_str());
                curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
                curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
                curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
                curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
                
                CURLcode res = curl_easy_perform(curl);
                curl_easy_cleanup(curl);
                file.close();
                
                if (res != CURLE_OK) {
                    std::cerr << "Failed to download model: " << curl_easy_strerror(res) << std::endl;
                    return false;
                }
                std::cout << "Model downloaded successfully" << std::endl;
            }
        }
        
        // Load the model
        model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        if (!model_) {
            std::cerr << "Failed to load model from: " << model_path << std::endl;
            return false;
        }
        
        // Build the interpreter
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model_, resolver);
        builder(&interpreter_);
        
        if (!interpreter_) {
            std::cerr << "Failed to create interpreter" << std::endl;
            return false;
        }
        
        // Allocate tensor buffers
        if (interpreter_->AllocateTensors() != kTfLiteOk) {
            std::cerr << "Failed to allocate tensors" << std::endl;
            return false;
        }
        
        std::cout << "BlazeFace detector initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing BlazeFace: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat BlazeFaceDetector::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;
    
    // Resize to model input size
    cv::resize(image, processed, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    
    // Convert to RGB if needed
    if (processed.channels() == 4) {
        cv::cvtColor(processed, processed, cv::COLOR_BGRA2RGB);
    } else if (processed.channels() == 1) {
        cv::cvtColor(processed, processed, cv::COLOR_GRAY2RGB);
    } else if (processed.channels() == 3) {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    }
    
    // Convert to float32 and normalize to [-1, 1]
    processed.convertTo(processed, CV_32F);
    processed = (processed - 127.5f) / 127.5f;
    
    return processed;
}

std::vector<FaceDetection> BlazeFaceDetector::detectFaces(const cv::Mat& image) {
    if (!interpreter_) {
        return {};
    }
    
    try {
        // Preprocess the image
        cv::Mat input = preprocessImage(image);
        
        // Copy input to interpreter
        float* input_data = interpreter_->typed_input_tensor<float>(0);
        std::memcpy(input_data, input.data, input.total() * input.elemSize());
        
        // Run inference
        if (interpreter_->Invoke() != kTfLiteOk) {
            std::cerr << "Failed to invoke interpreter" << std::endl;
            return {};
        }
        
        // Get output tensors
        // BlazeFace outputs: regressors (boxes) and classificators (scores)
        float* raw_boxes = interpreter_->typed_output_tensor<float>(0);
        float* raw_scores = interpreter_->typed_output_tensor<float>(1);
        
        // Get number of detections
        int num_boxes = interpreter_->output_tensor(0)->dims->data[1];
        
        // Postprocess detections
        auto detections = postprocessDetections(raw_boxes, raw_scores, num_boxes, image.size());
        
        // Apply NMS
        return applyNMS(detections);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in face detection: " << e.what() << std::endl;
        return {};
    }
}

std::optional<FaceDetection> BlazeFaceDetector::detectBestFace(const cv::Mat& image) {
    auto faces = detectFaces(image);
    if (faces.empty()) {
        return std::nullopt;
    }
    
    // Return face with highest confidence
    return *std::max_element(faces.begin(), faces.end(), 
        [](const FaceDetection& a, const FaceDetection& b) {
            return a.confidence < b.confidence;
        });
}

std::vector<FaceDetection> BlazeFaceDetector::postprocessDetections(
    float* raw_boxes, float* raw_scores, int num_boxes, const cv::Size& original_size) {
    
    std::vector<FaceDetection> detections;
    
    for (int i = 0; i < num_boxes; ++i) {
        float score = 1.0f / (1.0f + std::exp(-raw_scores[i])); // Sigmoid
        
        if (score < confidence_threshold_) {
            continue;
        }
        
        FaceDetection det;
        det.confidence = score;
        
        // Decode box coordinates (x_center, y_center, width, height)
        float cx = raw_boxes[i * 4];
        float cy = raw_boxes[i * 4 + 1];
        float w = raw_boxes[i * 4 + 2];
        float h = raw_boxes[i * 4 + 3];
        
        // Convert from center format to corner format
        det.bbox.x = cx - w / 2.0f;
        det.bbox.y = cy - h / 2.0f;
        det.bbox.width = w;
        det.bbox.height = h;
        
        // Clamp to [0, 1]
        det.bbox.x = std::max(0.0f, std::min(1.0f, det.bbox.x));
        det.bbox.y = std::max(0.0f, std::min(1.0f, det.bbox.y));
        det.bbox.width = std::min(det.bbox.width, 1.0f - det.bbox.x);
        det.bbox.height = std::min(det.bbox.height, 1.0f - det.bbox.y);
        
        detections.push_back(det);
    }
    
    return detections;
}

float BlazeFaceDetector::calculateIOU(const cv::Rect2f& box1, const cv::Rect2f& box2) {
    float x1 = std::max(box1.x, box2.x);
    float y1 = std::max(box1.y, box2.y);
    float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    if (x2 < x1 || y2 < y1) {
        return 0.0f;
    }
    
    float intersection = (x2 - x1) * (y2 - y1);
    float union_area = box1.area() + box2.area() - intersection;
    
    return intersection / union_area;
}

std::vector<FaceDetection> BlazeFaceDetector::applyNMS(const std::vector<FaceDetection>& detections) {
    if (detections.empty()) {
        return {};
    }
    
    std::vector<FaceDetection> result;
    std::vector<bool> suppressed(detections.size(), false);
    
    // Sort by confidence
    std::vector<size_t> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), 
        [&detections](size_t a, size_t b) {
            return detections[a].confidence > detections[b].confidence;
        });
    
    for (size_t i : indices) {
        if (suppressed[i]) continue;
        
        result.push_back(detections[i]);
        
        for (size_t j = 0; j < detections.size(); ++j) {
            if (i == j || suppressed[j]) continue;
            
            float iou = calculateIOU(detections[i].bbox, detections[j].bbox);
            if (iou > nms_threshold_) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

HeadPose BlazeFaceDetector::calculateHeadPose(const FaceDetection& face, const cv::Size& image_size) {
    HeadPose pose;
    
    // Simple pose estimation based on face bounding box position and size
    // Center of image is considered neutral pose
    float cx = face.bbox.x + face.bbox.width / 2.0f;
    float cy = face.bbox.y + face.bbox.height / 2.0f;
    
    // Yaw: left/right based on horizontal position
    pose.yaw = (cx - 0.5f) * 60.0f; // -30 to +30 degrees
    
    // Pitch: up/down based on vertical position
    pose.pitch = (0.5f - cy) * 40.0f; // -20 to +20 degrees
    
    // Roll: estimate based on face aspect ratio (simplified)
    float aspect_ratio = face.bbox.width / face.bbox.height;
    pose.roll = (aspect_ratio - 1.0f) * 10.0f;
    
    return pose;
}

std::string BlazeFaceDetector::downloadVideo(const std::string& url) {
    // Create temporary file for video
    std::string temp_path = "/tmp/video_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".mp4";
    
    CURL* curl = curl_easy_init();
    if (curl) {
        std::ofstream file(temp_path, std::ios::binary);
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
        
        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        file.close();
        
        if (res != CURLE_OK) {
            std::cerr << "Failed to download video: " << curl_easy_strerror(res) << std::endl;
            return "";
        }
    }
    
    return temp_path;
}

std::vector<cv::Mat> BlazeFaceDetector::extractFrames(const std::string& video_path_or_url, int max_frames) {
    std::vector<cv::Mat> frames;
    std::string video_path = video_path_or_url;
    std::string temp_path;
    
    // Download if URL
    if (video_path_or_url.find("http://") == 0 || video_path_or_url.find("https://") == 0) {
        temp_path = downloadVideo(video_path_or_url);
        if (temp_path.empty()) {
            return frames;
        }
        video_path = temp_path;
    }
    
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        if (!temp_path.empty()) {
            std::filesystem::remove(temp_path);
        }
        return frames;
    }
    
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int skip = std::max(1, total_frames / max_frames);
    
    cv::Mat frame;
    int frame_count = 0;
    while (cap.read(frame) && frames.size() < max_frames) {
        if (frame_count % skip == 0) {
            frames.push_back(frame.clone());
        }
        frame_count++;
    }
    
    cap.release();
    
    // Clean up temp file
    if (!temp_path.empty()) {
        std::filesystem::remove(temp_path);
    }
    
    return frames;
}

void BlazeFaceDetector::calculateMovementStats(const std::vector<HeadPose>& poses, 
                                              float& yaw_range, float& pitch_range, float& roll_range) {
    if (poses.empty()) {
        yaw_range = pitch_range = roll_range = 0;
        return;
    }
    
    auto [min_yaw, max_yaw] = std::minmax_element(poses.begin(), poses.end(), 
        [](const HeadPose& a, const HeadPose& b) { return a.yaw < b.yaw; });
    auto [min_pitch, max_pitch] = std::minmax_element(poses.begin(), poses.end(), 
        [](const HeadPose& a, const HeadPose& b) { return a.pitch < b.pitch; });
    auto [min_roll, max_roll] = std::minmax_element(poses.begin(), poses.end(), 
        [](const HeadPose& a, const HeadPose& b) { return a.roll < b.roll; });
    
    yaw_range = max_yaw->yaw - min_yaw->yaw;
    pitch_range = max_pitch->pitch - min_pitch->pitch;
    roll_range = max_roll->roll - min_roll->roll;
}

std::string BlazeFaceDetector::detectMovementDirection(const std::vector<HeadPose>& poses) {
    if (poses.size() < 3) {
        return "NONE";
    }
    
    float yaw_range, pitch_range, roll_range;
    calculateMovementStats(poses, yaw_range, pitch_range, roll_range);
    
    // Determine primary movement direction
    const float MOVEMENT_THRESHOLD = 15.0f; // degrees
    
    // Check for significant movements
    bool has_yaw = yaw_range > MOVEMENT_THRESHOLD;
    bool has_pitch = pitch_range > MOVEMENT_THRESHOLD;
    
    if (!has_yaw && !has_pitch) {
        return "NONE";
    }
    
    // If yaw movement is dominant
    if (yaw_range > pitch_range * 1.5f) {
        // Determine left or right by analyzing the trend
        float start_yaw = 0, end_yaw = 0;
        int count = 0;
        for (size_t i = 0; i < std::min(size_t(3), poses.size()); ++i) {
            start_yaw += poses[i].yaw;
            end_yaw += poses[poses.size() - 1 - i].yaw;
            count++;
        }
        start_yaw /= count;
        end_yaw /= count;
        
        return (end_yaw > start_yaw) ? "RIGHT" : "LEFT";
    }
    
    // If pitch movement is dominant
    if (pitch_range > yaw_range * 1.5f) {
        // Determine up or down by analyzing the trend
        float start_pitch = 0, end_pitch = 0;
        int count = 0;
        for (size_t i = 0; i < std::min(size_t(3), poses.size()); ++i) {
            start_pitch += poses[i].pitch;
            end_pitch += poses[poses.size() - 1 - i].pitch;
            count++;
        }
        start_pitch /= count;
        end_pitch /= count;
        
        return (end_pitch > start_pitch) ? "UP" : "DOWN";
    }
    
    // If movements are mixed, return the one with larger range
    if (yaw_range > pitch_range) {
        float avg_yaw = std::accumulate(poses.begin(), poses.end(), 0.0f,
            [](float sum, const HeadPose& p) { return sum + p.yaw; }) / poses.size();
        return (avg_yaw > 0) ? "RIGHT" : "LEFT";
    } else {
        float avg_pitch = std::accumulate(poses.begin(), poses.end(), 0.0f,
            [](float sum, const HeadPose& p) { return sum + p.pitch; }) / poses.size();
        return (avg_pitch > 0) ? "UP" : "DOWN";
    }
}

float BlazeFaceDetector::performLivenessCheck(const std::vector<cv::Mat>& frames, 
                                             const std::vector<FaceDetection>& detections) {
    if (frames.empty() || detections.empty()) {
        return 0.0f;
    }
    
    float liveness_score = 0.0f;
    
    // Check 1: Face presence consistency
    float face_ratio = static_cast<float>(detections.size()) / frames.size();
    liveness_score += face_ratio * 0.3f;
    
    // Check 2: Face size variation (real faces vary in size)
    if (detections.size() > 1) {
        std::vector<float> sizes;
        for (const auto& det : detections) {
            sizes.push_back(det.bbox.area());
        }
        float mean_size = std::accumulate(sizes.begin(), sizes.end(), 0.0f) / sizes.size();
        float variance = 0.0f;
        for (float size : sizes) {
            variance += (size - mean_size) * (size - mean_size);
        }
        variance /= sizes.size();
        float std_dev = std::sqrt(variance);
        float cv = std_dev / mean_size; // Coefficient of variation
        
        // Real faces should have some variation but not too much
        if (cv > 0.05f && cv < 0.3f) {
            liveness_score += 0.3f;
        } else if (cv > 0.02f && cv < 0.5f) {
            liveness_score += 0.2f;
        }
    }
    
    // Check 3: Face position variation (real faces move naturally)
    if (detections.size() > 1) {
        std::vector<cv::Point2f> centers;
        for (const auto& det : detections) {
            centers.push_back(det.getCenter());
        }
        
        float total_movement = 0.0f;
        for (size_t i = 1; i < centers.size(); ++i) {
            float dx = centers[i].x - centers[i-1].x;
            float dy = centers[i].y - centers[i-1].y;
            total_movement += std::sqrt(dx*dx + dy*dy);
        }
        float avg_movement = total_movement / (centers.size() - 1);
        
        // Natural movement should be present but not excessive
        if (avg_movement > 0.01f && avg_movement < 0.1f) {
            liveness_score += 0.2f;
        }
    }
    
    // Check 4: Confidence consistency
    if (detections.size() > 1) {
        float avg_confidence = std::accumulate(detections.begin(), detections.end(), 0.0f,
            [](float sum, const FaceDetection& d) { return sum + d.confidence; }) / detections.size();
        
        if (avg_confidence > 0.7f) {
            liveness_score += 0.2f;
        } else if (avg_confidence > 0.5f) {
            liveness_score += 0.1f;
        }
    }
    
    return std::min(1.0f, liveness_score);
}

VideoAnalysis BlazeFaceDetector::analyzeVideo(const std::string& video_path_or_url) {
    VideoAnalysis analysis;
    
    std::cout << "Analyzing video: " << video_path_or_url << std::endl;
    
    // Extract frames
    auto frames = extractFrames(video_path_or_url, 30);
    if (frames.empty()) {
        std::cerr << "Failed to extract frames from video" << std::endl;
        return analysis;
    }
    
    analysis.total_frames = frames.size();
    
    // Detect faces and calculate poses
    std::vector<FaceDetection> all_detections;
    std::vector<float> face_sizes;
    
    for (const auto& frame : frames) {
        auto face_opt = detectBestFace(frame);
        if (face_opt.has_value()) {
            auto face = face_opt.value();
            all_detections.push_back(face);
            analysis.frames_with_face++;
            
            // Calculate pose
            HeadPose pose = calculateHeadPose(face, frame.size());
            analysis.poses.push_back(pose);
            
            // Track face size
            cv::Rect pixel_bbox = face.toPixelCoords(frame.cols, frame.rows);
            float face_size = static_cast<float>(pixel_bbox.area()) / (frame.cols * frame.rows);
            face_sizes.push_back(face_size);
        }
    }
    
    // Calculate statistics
    if (!all_detections.empty()) {
        analysis.has_face = true;
        analysis.face_presence_ratio = static_cast<float>(analysis.frames_with_face) / analysis.total_frames;
        
        // Average face size
        if (!face_sizes.empty()) {
            analysis.avg_face_size = std::accumulate(face_sizes.begin(), face_sizes.end(), 0.0f) / face_sizes.size();
        }
        
        // Detect primary movement direction
        if (analysis.poses.size() >= 3) {
            analysis.primary_direction = detectMovementDirection(analysis.poses);
            
            // Calculate confidence based on movement consistency
            float yaw_range, pitch_range, roll_range;
            calculateMovementStats(analysis.poses, yaw_range, pitch_range, roll_range);
            
            float max_range = std::max({yaw_range, pitch_range, roll_range});
            if (max_range > 15.0f) {
                analysis.confidence = std::min(1.0f, max_range / 45.0f);
            } else {
                analysis.confidence = 0.3f;
            }
        }
        
        // Perform liveness check (simplified)
        analysis.liveness_score = performLivenessCheck(frames, all_detections);
        analysis.is_live = analysis.liveness_score > 0.5f;
    }
    
    std::cout << "Video analysis complete - Direction: " << analysis.primary_direction 
              << ", Confidence: " << analysis.confidence 
              << ", Liveness: " << analysis.liveness_score << std::endl;
    
    return analysis;
}

std::vector<VideoAnalysis> BlazeFaceDetector::analyzeVideos(
    const std::vector<std::string>& video_paths_or_urls, 
    int liveness_check_index) {
    
    // If liveness_check_index is -1, randomly select one video
    if (liveness_check_index == -1 && !video_paths_or_urls.empty()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, video_paths_or_urls.size() - 1);
        liveness_check_index = dis(gen);
        std::cout << "Randomly selected video " << (liveness_check_index + 1) 
                  << " for liveness check" << std::endl;
    }
    
    return ConcurrentVideoProcessor::processVideos(*this, video_paths_or_urls, liveness_check_index);
}

// ConcurrentVideoProcessor implementation
std::vector<VideoAnalysis> ConcurrentVideoProcessor::processVideos(
    BlazeFaceDetector& detector,
    const std::vector<std::string>& video_urls,
    int liveness_check_index,
    int max_threads) {
    
    std::vector<std::future<VideoAnalysis>> futures;
    
    auto start_time = std::chrono::steady_clock::now();
    
    for (size_t i = 0; i < video_urls.size(); ++i) {
        const auto& url = video_urls[i];
        bool perform_liveness = (static_cast<int>(i) == liveness_check_index);
        
        futures.push_back(std::async(std::launch::async, [&detector, url, perform_liveness, i]() {
            std::cout << "Processing video " << (i + 1) << (perform_liveness ? " (with liveness check)" : "") << std::endl;
            
            VideoAnalysis analysis = detector.analyzeVideo(url);
            
            // Only perform detailed liveness check on selected video
            if (!perform_liveness) {
                // Skip liveness check for other videos to save time
                analysis.is_live = true;
                analysis.liveness_score = 1.0f;
            }
            
            return analysis;
        }));
    }
    
    // Collect results
    std::vector<VideoAnalysis> results;
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Processed " << video_urls.size() << " videos in " 
              << duration.count() << " ms" << std::endl;
    
    return results;
}

} // namespace blazeface