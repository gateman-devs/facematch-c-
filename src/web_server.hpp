#ifndef WEB_SERVER_HPP
#define WEB_SERVER_HPP

#include "image_processor.hpp"
#include "face_recognizer.hpp"
#include "liveness_detector.hpp"
#include "video_liveness_detector.hpp"
#include <crow.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <chrono>
#include <future>

using json = nlohmann::json;

class WebServer {
public:
    WebServer();
    ~WebServer() = default;
    
    // Initialize with model paths
    bool initialize(const std::string& models_path);
    
    // Start server
    void start(int port = 8080);
    
    // Stop server
    void stop();

private:
    // Core components
    std::unique_ptr<ImageProcessor> image_processor;
    std::unique_ptr<FaceRecognizer> face_recognizer;
    std::unique_ptr<LivenessDetector> liveness_detector;
    std::unique_ptr<VideoLivenessDetector> video_liveness_detector;
    
    // Crow app
    crow::SimpleApp app;
    
    // Server state
    bool initialized;
    std::string models_path;
    
    // Endpoint handlers
    crow::response handleFaceComparison(const crow::request& req);
    crow::response handleLivenessCheck(const crow::request& req);
    crow::response handleSingleVideoLivenessCheck(const crow::request& req);
    crow::response handleHealthCheck(const crow::request& req);
    
    // Helper methods
    json parseRequestBody(const std::string& body);
    std::pair<cv::Mat, cv::Mat> loadImagesConcurrently(const std::string& image1_input, 
                                                       const std::string& image2_input);
    json createErrorResponse(const std::string& error_message, int status_code = 400);
    json createSuccessResponse(const json& data);
    crow::response createResponse(int status_code, const json& data);
    
    // Validation methods
    bool validateComparisonRequest(const json& request_data);
    bool validateLivenessRequest(const json& request_data);
    bool validateSingleVideoLivenessRequest(const json& request_data);
    
    // Timing utility
    class Timer {
    private:
        std::chrono::high_resolution_clock::time_point start_time;
    public:
        Timer() : start_time(std::chrono::high_resolution_clock::now()) {}
        
        int64_t elapsed_ms() const {
            auto end_time = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        }
    };
    
    // CORS middleware
    void setupCORS();
    
    // Error handling
    void setupErrorHandlers();
};

#endif // WEB_SERVER_HPP
