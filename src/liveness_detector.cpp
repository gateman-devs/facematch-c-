#include "liveness_detector.hpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <cstring>

LivenessDetector::LivenessDetector() : initialized(false) {
    face_detector = dlib::get_frontal_face_detector();
}

bool LivenessDetector::initialize(const std::string& shape_predictor_path) {
    try {
        dlib::deserialize(shape_predictor_path) >> shape_predictor;
        initialized = true;
        std::cout << "Liveness detector initialized successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing liveness detector: " << e.what() << std::endl;
        initialized = false;
        return false;
    }
}

LivenessAnalysis LivenessDetector::checkLiveness(const cv::Mat& image) {
    LivenessAnalysis analysis;
    
    if (!initialized || image.empty()) {
        return analysis;
    }
    
    try {
        // Extract face region for analysis
        cv::Mat face_region = extractFaceRegion(image);
        if (face_region.empty()) {
            std::cout << "No face detected for liveness analysis" << std::endl;
            return analysis;
        }
        
        // Perform comprehensive liveness checks
        analysis.texture_score = analyzeTexture(face_region);
        analysis.landmark_consistency = checkLandmarkConsistency(image);
        analysis.image_quality = analyzeImageQuality(image);
        analysis.sharpness_score = analyzeSharpness(image);
        analysis.lighting_score = analyzeLightingQuality(image);
        analysis.spoof_detection_score = detectSpoofing(image);
        
        // Production-level filtering: Hard rejections for clear failures
        std::vector<std::string> failures;
        
        if (analysis.sharpness_score < MIN_SHARPNESS_THRESHOLD) {
            failures.push_back("insufficient_sharpness");
        }
        
        if (analysis.texture_score < MIN_TEXTURE_THRESHOLD) {
            failures.push_back("low_texture_quality");
        }
        
        if (analysis.lighting_score < MIN_LIGHTING_QUALITY) {
            failures.push_back("poor_lighting");
        }
        
        if (analysis.spoof_detection_score < 0.5f) {
            failures.push_back("spoofing_detected");
        }
        
        // Calculate overall score with new weights
        analysis.overall_score = (TEXTURE_WEIGHT * analysis.texture_score +
                                 LANDMARK_WEIGHT * analysis.landmark_consistency +
                                 QUALITY_WEIGHT * analysis.image_quality +
                                 SHARPNESS_WEIGHT * analysis.sharpness_score +
                                 LIGHTING_WEIGHT * analysis.lighting_score +
                                 SPOOF_WEIGHT * analysis.spoof_detection_score);
        
        // Determine liveness: fail on hard rejections or low overall score
        analysis.is_live = failures.empty() && (analysis.overall_score > LIVENESS_THRESHOLD);
        analysis.confidence = analysis.overall_score;
        
        if (!failures.empty()) {
            analysis.has_failure = true;
            // Copy primary failure reason to the char array
            strncpy(analysis.failure_reason, failures[0].c_str(), sizeof(analysis.failure_reason) - 1);
            analysis.failure_reason[sizeof(analysis.failure_reason) - 1] = '\0';  // Ensure null termination
        }
        
        std::cout << "Liveness analysis - Score: " << analysis.overall_score 
                  << ", Live: " << (analysis.is_live ? "Yes" : "No") << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during liveness analysis: " << e.what() << std::endl;
    }
    
    return analysis;
}

float LivenessDetector::analyzeTexture(const cv::Mat& face_region) {
    if (face_region.empty()) {
        return 0.0f;
    }
    
    cv::Mat gray;
    if (face_region.channels() == 3) {
        cv::cvtColor(face_region, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = face_region.clone();
    }
    
    // Multiple texture analysis methods
    float lbp_score = calculateLBPUniformity(gray);
    float frequency_score = analyzeFrequencyDomain(gray);
    float edge_score = checkEdgeConsistency(gray);
    float moire_score = calculateMoirePatterns(gray);
    
    // Combine scores
    float texture_score = (0.3f * lbp_score + 0.3f * frequency_score + 
                          0.2f * edge_score + 0.2f * moire_score);
    
    return std::max(0.0f, std::min(1.0f, texture_score));
}

float LivenessDetector::checkLandmarkConsistency(const cv::Mat& image) {
    if (!initialized || image.empty()) {
        return 0.0f;
    }
    
    std::vector<dlib::point> landmarks = detectLandmarks(image);
    if (landmarks.empty()) {
        return 0.0f;
    }
    
    float stability_score = calculateLandmarkStability(landmarks);
    float eye_score = analyzeEyeRegion(image, landmarks);
    
    return (0.6f * stability_score + 0.4f * eye_score);
}

float LivenessDetector::analyzeImageQuality(const cv::Mat& image) {
    if (image.empty()) {
        return 0.0f;
    }
    
    // Convert to grayscale for analysis
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Sharpness analysis
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    float sharpness = static_cast<float>(stddev.val[0] * stddev.val[0] / 1000.0);
    
    // Noise analysis
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::Mat noise = gray - blurred;
    cv::meanStdDev(noise, mean, stddev);
    float noise_level = static_cast<float>(stddev.val[0]);
    float noise_score = 1.0f - std::min(noise_level / 20.0f, 1.0f);
    
    // Color distribution analysis
    float color_score = analyzeColorDistribution(image);
    
    // Reflection detection
    float reflection_score = detectReflections(image);
    
    return (0.3f * std::min(sharpness, 1.0f) + 0.3f * noise_score + 
            0.2f * color_score + 0.2f * reflection_score);
}

cv::Mat LivenessDetector::extractFaceRegion(const cv::Mat& image) {
    try {
        dlib::cv_image<dlib::bgr_pixel> dlib_image(image);
        std::vector<dlib::rectangle> faces = face_detector(dlib_image);
        
        if (faces.empty()) {
            return cv::Mat();
        }
        
        // Use the largest face
        dlib::rectangle largest_face = faces[0];
        for (const auto& face : faces) {
            if (face.area() > largest_face.area()) {
                largest_face = face;
            }
        }
        
        // Convert to OpenCV rect and extract
        cv::Rect face_rect(largest_face.left(), largest_face.top(),
                          largest_face.width(), largest_face.height());
        
        // Ensure rect is within image bounds
        face_rect &= cv::Rect(0, 0, image.cols, image.rows);
        
        if (face_rect.width <= 0 || face_rect.height <= 0) {
            return cv::Mat();
        }
        
        return image(face_rect);
        
    } catch (const std::exception& e) {
        std::cerr << "Error extracting face region: " << e.what() << std::endl;
        return cv::Mat();
    }
}

float LivenessDetector::calculateLBPUniformity(const cv::Mat& region) {
    if (region.empty()) {
        return 0.0f;
    }
    
    cv::Mat lbp = cv::Mat::zeros(region.size(), CV_8UC1);
    
    // Calculate LBP
    for (int i = 1; i < region.rows - 1; i++) {
        for (int j = 1; j < region.cols - 1; j++) {
            uchar center = region.at<uchar>(i, j);
            uchar code = 0;
            
            // 8-neighbor LBP
            code |= (region.at<uchar>(i-1, j-1) >= center) << 7;
            code |= (region.at<uchar>(i-1, j) >= center) << 6;
            code |= (region.at<uchar>(i-1, j+1) >= center) << 5;
            code |= (region.at<uchar>(i, j+1) >= center) << 4;
            code |= (region.at<uchar>(i+1, j+1) >= center) << 3;
            code |= (region.at<uchar>(i+1, j) >= center) << 2;
            code |= (region.at<uchar>(i+1, j-1) >= center) << 1;
            code |= (region.at<uchar>(i, j-1) >= center) << 0;
            
            lbp.at<uchar>(i, j) = code;
        }
    }
    
    // Calculate histogram
    std::vector<int> hist(256, 0);
    for (int i = 1; i < lbp.rows - 1; i++) {
        for (int j = 1; j < lbp.cols - 1; j++) {
            hist[lbp.at<uchar>(i, j)]++;
        }
    }
    
    // Calculate uniformity (higher uniformity suggests printed/screen image)
    int total_pixels = (lbp.rows - 2) * (lbp.cols - 2);
    if (total_pixels == 0) return 0.0f;
    
    // Find dominant patterns
    std::sort(hist.rbegin(), hist.rend());
    float uniformity = static_cast<float>(hist[0] + hist[1] + hist[2]) / total_pixels;
    
    // Return inverted score (lower uniformity = more live)
    return 1.0f - std::min(uniformity, 1.0f);
}

float LivenessDetector::analyzeFrequencyDomain(const cv::Mat& region) {
    if (region.empty()) {
        return 0.0f;
    }
    
    cv::Mat float_region;
    region.convertTo(float_region, CV_32F);
    
    // Apply DFT
    cv::Mat dft_result;
    cv::dft(float_region, dft_result, cv::DFT_COMPLEX_OUTPUT);
    
    // Calculate magnitude spectrum
    std::vector<cv::Mat> planes;
    cv::split(dft_result, planes);
    cv::Mat magnitude;
    cv::magnitude(planes[0], planes[1], magnitude);
    
    // Analyze frequency distribution
    cv::Scalar mean_mag = cv::mean(magnitude);
    cv::Scalar std_mag;
    cv::meanStdDev(magnitude, mean_mag, std_mag);
    
    // Higher frequency variation suggests real texture
    float frequency_variation = static_cast<float>(std_mag.val[0] / (mean_mag.val[0] + 1e-6));
    
    return std::min(frequency_variation / 2.0f, 1.0f);
}

float LivenessDetector::checkEdgeConsistency(const cv::Mat& region) {
    if (region.empty()) {
        return 0.0f;
    }
    
    // Calculate edges using Canny
    cv::Mat edges;
    cv::Canny(region, edges, 50, 150);
    
    // Count edge pixels
    int edge_count = cv::countNonZero(edges);
    int total_pixels = region.rows * region.cols;
    
    if (total_pixels == 0) return 0.0f;
    
    float edge_density = static_cast<float>(edge_count) / total_pixels;
    
    // Optimal edge density for real faces (empirically determined)
    float optimal_density = 0.1f;
    float edge_score = 1.0f - std::abs(edge_density - optimal_density) / optimal_density;
    
    return std::max(0.0f, std::min(1.0f, edge_score));
}

std::vector<dlib::point> LivenessDetector::detectLandmarks(const cv::Mat& image) {
    std::vector<dlib::point> landmarks;
    
    try {
        dlib::cv_image<dlib::bgr_pixel> dlib_image(image);
        std::vector<dlib::rectangle> faces = face_detector(dlib_image);
        
        if (!faces.empty()) {
            dlib::full_object_detection shape = shape_predictor(dlib_image, faces[0]);
            for (unsigned long i = 0; i < shape.num_parts(); ++i) {
                landmarks.push_back(shape.part(i));
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error detecting landmarks: " << e.what() << std::endl;
    }
    
    return landmarks;
}

float LivenessDetector::calculateLandmarkStability(const std::vector<dlib::point>& landmarks) {
    if (landmarks.size() < 68) {
        return 0.0f;
    }
    
    // Calculate facial symmetry
    float symmetry_score = 0.0f;
    int symmetry_pairs = 0;
    
    // Face outline symmetry points
    std::vector<std::pair<int, int>> symmetry_pairs_idx = {
        {0, 16}, {1, 15}, {2, 14}, {3, 13}, {4, 12}, {5, 11}, {6, 10}, {7, 9}
    };
    
    dlib::point nose_tip = landmarks[30]; // Nose tip as reference
    
    for (const auto& pair : symmetry_pairs_idx) {
        dlib::point left_point = landmarks[pair.first];
        dlib::point right_point = landmarks[pair.second];
        
        // Calculate distance from nose tip to each point
        float left_dist = std::sqrt(std::pow(left_point.x() - nose_tip.x(), 2) + 
                                   std::pow(left_point.y() - nose_tip.y(), 2));
        float right_dist = std::sqrt(std::pow(right_point.x() - nose_tip.x(), 2) + 
                                    std::pow(right_point.y() - nose_tip.y(), 2));
        
        // Calculate symmetry score
        float diff = std::abs(left_dist - right_dist);
        float avg_dist = (left_dist + right_dist) / 2.0f;
        
        if (avg_dist > 0) {
            symmetry_score += 1.0f - (diff / avg_dist);
            symmetry_pairs++;
        }
    }
    
    return symmetry_pairs > 0 ? symmetry_score / symmetry_pairs : 0.0f;
}

float LivenessDetector::analyzeEyeRegion(const cv::Mat& image, const std::vector<dlib::point>& landmarks) {
    if (landmarks.size() < 68 || image.empty()) {
        return 0.0f;
    }
    
    // Extract eye regions
    std::vector<cv::Point> left_eye_points, right_eye_points;
    
    // Left eye landmarks (36-41)
    for (int i = 36; i <= 41; i++) {
        left_eye_points.push_back(cv::Point(landmarks[i].x(), landmarks[i].y()));
    }
    
    // Right eye landmarks (42-47)
    for (int i = 42; i <= 47; i++) {
        right_eye_points.push_back(cv::Point(landmarks[i].x(), landmarks[i].y()));
    }
    
    // Calculate eye region bounding rectangles
    cv::Rect left_eye_rect = cv::boundingRect(left_eye_points);
    cv::Rect right_eye_rect = cv::boundingRect(right_eye_points);
    
    // Expand rectangles slightly
    left_eye_rect.x = std::max(0, left_eye_rect.x - 5);
    left_eye_rect.y = std::max(0, left_eye_rect.y - 5);
    left_eye_rect.width = std::min(image.cols - left_eye_rect.x, left_eye_rect.width + 10);
    left_eye_rect.height = std::min(image.rows - left_eye_rect.y, left_eye_rect.height + 10);
    
    right_eye_rect.x = std::max(0, right_eye_rect.x - 5);
    right_eye_rect.y = std::max(0, right_eye_rect.y - 5);
    right_eye_rect.width = std::min(image.cols - right_eye_rect.x, right_eye_rect.width + 10);
    right_eye_rect.height = std::min(image.rows - right_eye_rect.y, right_eye_rect.height + 10);
    
    if (left_eye_rect.width <= 0 || left_eye_rect.height <= 0 ||
        right_eye_rect.width <= 0 || right_eye_rect.height <= 0) {
        return 0.0f;
    }
    
    cv::Mat left_eye = image(left_eye_rect);
    cv::Mat right_eye = image(right_eye_rect);
    
    // Analyze eye regions for natural characteristics
    cv::Mat left_gray, right_gray;
    cv::cvtColor(left_eye, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_eye, right_gray, cv::COLOR_BGR2GRAY);
    
    // Calculate variance (natural eyes have more texture variation)
    cv::Scalar left_mean, left_std, right_mean, right_std;
    cv::meanStdDev(left_gray, left_mean, left_std);
    cv::meanStdDev(right_gray, right_mean, right_std);
    
    float left_variance = static_cast<float>(left_std.val[0]);
    float right_variance = static_cast<float>(right_std.val[0]);
    
    // Normalize variance scores
    float left_score = std::min(left_variance / 30.0f, 1.0f);
    float right_score = std::min(right_variance / 30.0f, 1.0f);
    
    return (left_score + right_score) / 2.0f;
}

float LivenessDetector::calculateMoirePatterns(const cv::Mat& image) {
    if (image.empty()) {
        return 1.0f; // No moire patterns detected
    }
    
    cv::Mat float_img;
    image.convertTo(float_img, CV_32F);
    
    // Apply FFT to detect periodic patterns
    cv::Mat dft_result;
    cv::dft(float_img, dft_result, cv::DFT_COMPLEX_OUTPUT);
    
    // Calculate power spectrum
    std::vector<cv::Mat> planes;
    cv::split(dft_result, planes);
    cv::Mat magnitude;
    cv::magnitude(planes[0], planes[1], magnitude);
    
    // Look for strong periodic components (moire patterns)
    cv::Mat log_magnitude;
    cv::log(magnitude + 1, log_magnitude);
    
    // Calculate mean and std of spectrum
    cv::Scalar mean, stddev;
    cv::meanStdDev(log_magnitude, mean, stddev);
    
    // High variance in frequency domain suggests moire patterns
    float moire_indicator = static_cast<float>(stddev.val[0]);
    
    // Return inverse score (lower moire = higher score)
    return 1.0f - std::min(moire_indicator / 5.0f, 1.0f);
}

float LivenessDetector::analyzeColorDistribution(const cv::Mat& image) {
    if (image.empty() || image.channels() != 3) {
        return 0.0f;
    }
    
    // Calculate color histograms
    std::vector<cv::Mat> bgr_planes;
    cv::split(image, bgr_planes);
    
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    
    cv::Mat b_hist, g_hist, r_hist;
    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange);
    
    // Calculate histogram entropy (natural images have higher entropy)
    auto calculateEntropy = [](const cv::Mat& hist) {
        cv::Mat normalized_hist;
        cv::normalize(hist, normalized_hist, 0, 1, cv::NORM_L1);
        
        double entropy = 0.0;
        for (int i = 0; i < hist.rows; i++) {
            float p = normalized_hist.at<float>(i);
            if (p > 0) {
                entropy -= p * std::log2(p);
            }
        }
        return entropy;
    };
    
    double b_entropy = calculateEntropy(b_hist);
    double g_entropy = calculateEntropy(g_hist);
    double r_entropy = calculateEntropy(r_hist);
    
    float avg_entropy = static_cast<float>((b_entropy + g_entropy + r_entropy) / 3.0);
    
    // Normalize entropy (8 bits = max entropy of 8)
    return std::min(avg_entropy / 8.0f, 1.0f);
}

float LivenessDetector::detectReflections(const cv::Mat& image) {
    if (image.empty()) {
        return 1.0f;
    }
    
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Detect bright spots that might be reflections
    cv::Mat bright_spots;
    cv::threshold(gray, bright_spots, 200, 255, cv::THRESH_BINARY);
    
    // Find connected components
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bright_spots, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    int total_reflection_area = 0;
    for (const auto& contour : contours) {
        int area = static_cast<int>(cv::contourArea(contour));
        if (area > 10) { // Filter small noise
            total_reflection_area += area;
        }
    }
    
    int total_area = gray.rows * gray.cols;
    float reflection_ratio = static_cast<float>(total_reflection_area) / total_area;
    
    // Penalize excessive reflections
    return 1.0f - std::min(reflection_ratio * 10.0f, 1.0f);
}

float LivenessDetector::analyzeSharpness(const cv::Mat& image) {
    if (image.empty()) {
        return 0.0f;
    }
    
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Calculate Laplacian variance as sharpness measure
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    double sharpness = stddev.val[0] * stddev.val[0];
    
    // Normalize to 0-1 range based on production image analysis
    // Typical values: Good images ~130, Spoofed ~4.3, Poor quality ~100
    float normalized_sharpness = static_cast<float>(std::min(sharpness / 1000.0, 1.0));
    
    return std::max(0.0f, std::min(1.0f, normalized_sharpness));
}

float LivenessDetector::analyzeLightingQuality(const cv::Mat& image) {
    if (image.empty()) {
        return 0.0f;
    }
    
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Calculate brightness statistics
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    double brightness = mean.val[0];
    double contrast = stddev.val[0];
    
    // Brightness score (optimal range around 128)
    double brightness_score = 1.0 - std::abs(brightness - 128.0) / 128.0;
    
    // Contrast score (good lighting has reasonable contrast)
    double contrast_score = std::min(contrast / 60.0, 1.0);
    
    // Check for proper illumination distribution
    // Calculate histogram to detect over/under-exposure
    std::vector<int> hist(256, 0);
    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            hist[gray.at<uchar>(i, j)]++;
        }
    }
    
    int total_pixels = gray.rows * gray.cols;
    
    // Check for clipping (too many pixels at extremes)
    float dark_pixels = static_cast<float>(hist[0] + hist[1] + hist[2]) / total_pixels;
    float bright_pixels = static_cast<float>(hist[253] + hist[254] + hist[255]) / total_pixels;
    float clipping_penalty = std::max(dark_pixels, bright_pixels);
    
    // Combined lighting quality score
    float lighting_score = static_cast<float>(0.4 * brightness_score + 0.4 * contrast_score + 0.2 * (1.0 - clipping_penalty));
    
    return std::max(0.0f, std::min(1.0f, lighting_score));
}

float LivenessDetector::detectSpoofing(const cv::Mat& image) {
    if (image.empty()) {
        return 0.0f;
    }
    
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // 1. Enhanced LBP uniformity check (spoofed images show high uniformity)
    float lbp_score = calculateLBPUniformity(gray);
    
    // 2. Screen reflection detection
    cv::Mat bright_spots;
    cv::threshold(gray, bright_spots, 200, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bright_spots, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    int total_reflection_area = 0;
    for (const auto& contour : contours) {
        int area = static_cast<int>(cv::contourArea(contour));
        if (area > 10) { // Filter small noise
            total_reflection_area += area;
        }
    }
    
    int total_area = gray.rows * gray.cols;
    float reflection_ratio = static_cast<float>(total_reflection_area) / total_area;
    
    // 3. Moire pattern detection (enhanced)
    float moire_score = calculateMoirePatterns(gray);
    
    // 4. Frequency domain analysis
    float frequency_score = analyzeFrequencyDomain(gray);
    
    // 5. Edge consistency check
    float edge_score = checkEdgeConsistency(gray);
    
    // Combine anti-spoofing indicators
    float spoof_score = 0.3f * lbp_score +           // LBP uniformity (lower = more suspicious)
                       0.2f * (reflection_ratio > MAX_REFLECTION_RATIO ? 0.0f : 1.0f) +  // Reflection penalty
                       0.2f * moire_score +          // Moire patterns
                       0.15f * frequency_score +     // Frequency domain analysis  
                       0.15f * edge_score;           // Edge consistency
    
    return std::max(0.0f, std::min(1.0f, spoof_score));
}


