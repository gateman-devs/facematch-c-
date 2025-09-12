#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/video_liveness_detector.hpp"

int main() {
    std::cout << "=== Face Detection Test ===" << std::endl;

    // Initialize the video liveness detector
    VideoLivenessDetector detector;
    if (!detector.initialize()) {
        std::cerr << "Failed to initialize video liveness detector" << std::endl;
        return 1;
    }

    // Test with a sample image (create a dummy frame for testing)
    cv::Mat test_frame = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(test_frame, cv::Rect(200, 150, 120, 120), cv::Scalar(255, 255, 255), -1);
    cv::circle(test_frame, cv::Point(260, 180), 5, cv::Scalar(0, 0, 0), -1); // Left eye
    cv::circle(test_frame, cv::Point(320, 180), 5, cv::Scalar(0, 0, 0), -1); // Right eye
    cv::circle(test_frame, cv::Point(290, 220), 3, cv::Scalar(0, 0, 0), -1); // Nose
    cv::ellipse(test_frame, cv::Point(290, 240), cv::Size(15, 8), 0, 0, 180, cv::Scalar(0, 0, 0), -1); // Mouth

    std::cout << "Testing face landmark detection..." << std::endl;

    // Test landmark detection directly
    auto landmarks = detector.getFaceLandmarks(test_frame);

    if (landmarks.empty()) {
        std::cout << "No landmarks detected (expected for synthetic image)" << std::endl;
    } else {
        std::cout << "Detected " << landmarks.size() << " landmarks:" << std::endl;
        for (size_t i = 0; i < landmarks.size(); ++i) {
            std::cout << "  Landmark " << i << ": (" << landmarks[i].x << ", " << landmarks[i].y << ")" << std::endl;
        }
    }

    // Test head pose calculation
    HeadPoseMovement movement = detector.calculateHeadPose(test_frame, 0.0f);
    std::cout << "Head pose: yaw=" << movement.yaw_angle
              << "°, pitch=" << movement.pitch_angle
              << "°, roll=" << movement.roll_angle << "°" << std::endl;

    std::cout << "Face detection test completed successfully!" << std::endl;
    return 0;
}
