#include "face_recognizer.hpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <iostream>
#include <cmath>

FaceRecognizer::FaceRecognizer() : models_loaded(false) {
    // Constructor - models will be loaded in initialize()
}

bool FaceRecognizer::initialize(const std::string& face_recognition_model_path,
                               const std::string& shape_predictor_path) {
    try {
        std::cout << "Loading face detection models..." << std::endl;
        
        // Load face detector
        face_detector = dlib::get_frontal_face_detector();
        
        // Load shape predictor
        dlib::deserialize(shape_predictor_path) >> pose_model;
        std::cout << "Shape predictor loaded: " << shape_predictor_path << std::endl;
        
        // Load face recognition model
        dlib::deserialize(face_recognition_model_path) >> face_encoder;
        std::cout << "Face recognition model loaded: " << face_recognition_model_path << std::endl;
        
        models_loaded = true;
        std::cout << "All face recognition models loaded successfully!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading face recognition models: " << e.what() << std::endl;
        models_loaded = false;
        return false;
    }
}

FaceInfo FaceRecognizer::processFace(const cv::Mat& image) {
    FaceInfo face_info;
    
    if (!models_loaded || image.empty()) {
        return face_info;
    }
    
    try {
        // Convert OpenCV Mat to dlib format
        dlib::cv_image<dlib::bgr_pixel> dlib_image(image);
        
        // Detect faces
        std::vector<dlib::rectangle> faces = face_detector(dlib_image);
        
        if (faces.empty()) {
            std::cout << "No faces detected in image" << std::endl;
            return face_info;
        }
        
        // Use the largest face
        dlib::rectangle largest_face = faces[0];
        for (const auto& face : faces) {
            if (face.area() > largest_face.area()) {
                largest_face = face;
            }
        }
        
        // Validate face size
        if (!isValidFace(largest_face, image)) {
            std::cout << "Face too small or invalid" << std::endl;
            return face_info;
        }
        
        face_info.face_rect = largest_face;
        
        // Get facial landmarks
        dlib::full_object_detection landmarks = pose_model(dlib_image, largest_face);
        for (unsigned long i = 0; i < landmarks.num_parts(); ++i) {
            face_info.landmarks.push_back(landmarks.part(i));
        }
        
        // Extract face chip for encoding
        dlib::matrix<dlib::rgb_pixel> face_chip;
        dlib::extract_image_chip(dlib_image, dlib::get_face_chip_details(landmarks, 150, 0.25), face_chip);
        
        // Get face encoding
        face_info.encoding = face_encoder(face_chip);
        
        // Calculate quality score
        face_info.quality_score = calculateFaceQuality(face_chip);
        
        face_info.valid = true;
        
        std::cout << "Face processed successfully. Quality: " << face_info.quality_score << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing face: " << e.what() << std::endl;
        face_info.valid = false;
    }
    
    return face_info;
}

float FaceRecognizer::compareFaces(const dlib::matrix<float,0,1>& encoding1,
                                  const dlib::matrix<float,0,1>& encoding2) {
    if (encoding1.size() == 0 || encoding2.size() == 0) {
        return 1.0f; // Maximum distance for invalid encodings
    }
    
    // Calculate Euclidean distance
    float distance = dlib::length(encoding1 - encoding2);
    return distance;
}

FaceRecognizer::ComparisonResult FaceRecognizer::compareFacesDetailed(const FaceInfo& face1, const FaceInfo& face2) {
    ComparisonResult result;
    result.is_match = false;
    result.confidence = 0.0f;
    result.distance = 1.0f;
    
    if (!face1.valid || !face2.valid) {
        return result;
    }
    
    // Calculate distance
    result.distance = compareFaces(face1.encoding, face2.encoding);
    
    // Determine match based on threshold
    result.is_match = result.distance < MATCH_THRESHOLD;
    
    // Calculate confidence (inverse of distance, normalized)
    result.confidence = std::max(0.0f, (1.0f - result.distance / MATCH_THRESHOLD));
    
    return result;
}

float FaceRecognizer::getFaceQuality(const cv::Mat& image, const dlib::rectangle& face_rect) {
    if (image.empty() || !isValidFace(face_rect, image)) {
        return 0.0f;
    }
    
    // Extract face region
    cv::Rect cv_rect(face_rect.left(), face_rect.top(), 
                     face_rect.width(), face_rect.height());
    
    // Ensure rect is within image bounds
    cv_rect &= cv::Rect(0, 0, image.cols, image.rows);
    
    if (cv_rect.width <= 0 || cv_rect.height <= 0) {
        return 0.0f;
    }
    
    cv::Mat face_region = image(cv_rect);
    
    // Convert to grayscale for analysis
    cv::Mat gray;
    if (face_region.channels() == 3) {
        cv::cvtColor(face_region, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = face_region.clone();
    }
    
    // Calculate sharpness using Laplacian variance
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    double sharpness = stddev.val[0] * stddev.val[0];
    
    // Calculate contrast
    cv::meanStdDev(gray, mean, stddev);
    double contrast = stddev.val[0];
    
    // Calculate brightness (should be in reasonable range)
    double brightness = mean.val[0];
    double brightness_score = 1.0 - std::abs(brightness - 128.0) / 128.0;
    
    // Combine metrics (normalized and weighted)
    float sharpness_score = static_cast<float>(std::min(sharpness / 1000.0, 1.0));
    float contrast_score = static_cast<float>(std::min(contrast / 64.0, 1.0));
    float brightness_norm = static_cast<float>(std::max(0.0, brightness_score));
    
    // Weighted combination
    float quality = 0.5f * sharpness_score + 0.3f * contrast_score + 0.2f * brightness_norm;
    
    return std::min(1.0f, std::max(0.0f, quality));
}

bool FaceRecognizer::isValidFace(const dlib::rectangle& face_rect, const cv::Mat& image) {
    // Check minimum size
    if (face_rect.width() < MIN_FACE_SIZE || face_rect.height() < MIN_FACE_SIZE) {
        return false;
    }
    
    // Check if face is within image bounds
    if (face_rect.left() < 0 || face_rect.top() < 0 ||
        face_rect.right() >= image.cols || face_rect.bottom() >= image.rows) {
        return false;
    }
    
    // Check aspect ratio (faces should be roughly square)
    float aspect_ratio = static_cast<float>(face_rect.width()) / face_rect.height();
    if (aspect_ratio < 0.5f || aspect_ratio > 2.0f) {
        return false;
    }
    
    return true;
}

float FaceRecognizer::calculateFaceQuality(const dlib::matrix<dlib::rgb_pixel>& face_chip) {
    // Convert dlib matrix to OpenCV Mat for quality analysis
    cv::Mat cv_face_chip(face_chip.nr(), face_chip.nc(), CV_8UC3);
    
    for (long r = 0; r < face_chip.nr(); ++r) {
        for (long c = 0; c < face_chip.nc(); ++c) {
            dlib::rgb_pixel pixel = face_chip(r, c);
            cv_face_chip.at<cv::Vec3b>(r, c) = cv::Vec3b(pixel.blue, pixel.green, pixel.red);
        }
    }
    
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(cv_face_chip, gray, cv::COLOR_BGR2GRAY);
    
    // Calculate sharpness using Laplacian variance
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    double sharpness = stddev.val[0] * stddev.val[0];
    
    // Normalize sharpness score
    float quality = static_cast<float>(std::min(sharpness / 1000.0, 1.0));
    
    return std::max(0.0f, std::min(1.0f, quality));
}

std::vector<dlib::point> FaceRecognizer::getLandmarks(const cv::Mat& image, const dlib::rectangle& face_rect) {
    std::vector<dlib::point> landmarks;
    
    try {
        dlib::cv_image<dlib::bgr_pixel> dlib_image(image);
        dlib::full_object_detection shape = pose_model(dlib_image, face_rect);
        
        for (unsigned long i = 0; i < shape.num_parts(); ++i) {
            landmarks.push_back(shape.part(i));
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error getting landmarks: " << e.what() << std::endl;
    }
    
    return landmarks;
}
