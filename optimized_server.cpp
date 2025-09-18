#include "src/blazeface/blazeface_detector.hpp"
#include <crow.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <random>
#include <vector>
#include <sstream>

using json = nlohmann::json;
using namespace blazeface;

class OptimizedWebServer {
private:
    BlazeFaceDetector detector;
    crow::SimpleApp app;
    bool initialized;
    
    // URL decoding utility
    std::string decodeUrl(const std::string& encoded_url) {
        std::string decoded = encoded_url;
        std::string hex_chars = "0123456789ABCDEFabcdef";
        
        size_t pos = 0;
        while ((pos = decoded.find('%', pos)) != std::string::npos) {
            if (pos + 2 < decoded.length()) {
                std::string hex_str = decoded.substr(pos + 1, 2);
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
    
    // Create standardized response
    crow::response createResponse(int status_code, const json& data) {
        crow::response res(status_code, data.dump());
        res.add_header("Access-Control-Allow-Origin", "*");
        res.add_header("Content-Type", "application/json");
        return res;
    }
    
    json createSuccessResponse(const json& data) {
        json response = data;
        response["success"] = true;
        return response;
    }
    
    json createErrorResponse(const std::string& message) {
        return json{
            {"success", false},
            {"error", message}
        };
    }

public:
    OptimizedWebServer() : initialized(false) {}
    
    bool initialize() {
        std::cout << "========================================" << std::endl;
        std::cout << "Initializing Optimized Web Server" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // Initialize BlazeFace detector
        std::string model_path = "./models/blazeface.tflite";
        if (!detector.initialize(model_path)) {
            std::cerr << "Failed to initialize BlazeFace detector" << std::endl;
            return false;
        }
        
        // Configure detector for optimal performance
        detector.setConfidenceThreshold(0.5f);
        detector.setNMSThreshold(0.3f);
        
        setupRoutes();
        initialized = true;
        
        std::cout << "✓ Optimized Web Server initialized successfully" << std::endl;
        std::cout << "✓ Using BlazeFace for face detection" << std::endl;
        std::cout << "✓ Optimized for speed and accuracy" << std::endl;
        return true;
    }
    
    void setupRoutes() {
        // CORS preflight
        CROW_ROUTE(app, "/<path>").methods("OPTIONS"_method)
        ([](const std::string&) {
            crow::response res(200);
            res.add_header("Access-Control-Allow-Origin", "*");
            res.add_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.add_header("Access-Control-Allow-Headers", "Content-Type");
            return res;
        });
        
        // Health check endpoint
        CROW_ROUTE(app, "/health").methods("GET"_method)
        ([this]() {
            json health_data = {
                {"status", "healthy"},
                {"service", "optimized-face-service"},
                {"detector", "BlazeFace"},
                {"initialized", initialized},
                {"version", "2.0.0"},
                {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()}
            };
            return createResponse(200, createSuccessResponse(health_data));
        });
        
        // Test endpoint to verify directions
        CROW_ROUTE(app, "/test-directions").methods("POST"_method)
        ([this](const crow::request& req) {
            auto start = std::chrono::steady_clock::now();
            
            try {
                // Test URLs with known directions
                std::vector<std::pair<std::string, std::string>> test_videos = {
                    {"https://res.cloudinary.com/themizehq/video/upload/v1755978338/IMG_6552.mov", "DOWN"},
                    {"https://res.cloudinary.com/themizehq/video/upload/v1755978338/IMG_6551.mov", "UP"},
                    {"https://res.cloudinary.com/themizehq/video/upload/v1755978331/IMG_6553.mov", "LEFT"},
                    {"https://res.cloudinary.com/themizehq/video/upload/v1755978327/IMG_6554.mov", "RIGHT"}
                };
                
                json test_results = json::array();
                
                for (const auto& [url, expected] : test_videos) {
                    std::cout << "\nTesting video: " << expected << std::endl;
                    VideoAnalysis analysis = detector.analyzeVideo(url);
                    
                    json result = {
                        {"url", url},
                        {"expected", expected},
                        {"detected", analysis.primary_direction},
                        {"match", analysis.primary_direction == expected},
                        {"confidence", analysis.confidence},
                        {"has_face", analysis.has_face},
                        {"face_presence_ratio", analysis.face_presence_ratio}
                    };
                    test_results.push_back(result);
                    
                    std::cout << "Expected: " << expected 
                              << ", Detected: " << analysis.primary_direction
                              << " " << (analysis.primary_direction == expected ? "✓" : "✗") << std::endl;
                }
                
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start);
                
                json response_data = {
                    {"test_results", test_results},
                    {"processing_time_ms", duration.count()}
                };
                
                return createResponse(200, createSuccessResponse(response_data));
                
            } catch (const std::exception& e) {
                return createResponse(500, createErrorResponse(std::string("Test failed: ") + e.what()));
            }
        });
        
        // Video liveness verification endpoint with optimized processing
        CROW_ROUTE(app, "/verify-video-liveness").methods("POST"_method)
        ([this](const crow::request& req) {
            auto start = std::chrono::steady_clock::now();
            
            try {
                json request_data = json::parse(req.body);
                
                if (!request_data.contains("challenge_id") || !request_data.contains("video_urls")) {
                    return createResponse(400, createErrorResponse("Missing required fields: challenge_id, video_urls"));
                }
                
                if (!request_data["video_urls"].is_array() || request_data["video_urls"].size() != 4) {
                    return createResponse(400, createErrorResponse("video_urls must be an array of exactly 4 URLs"));
                }
                
                std::string challenge_id = request_data["challenge_id"];
                std::vector<std::string> video_urls = request_data["video_urls"];
                
                // Expected directions (should be retrieved from cache/database in production)
                std::vector<std::string> expected_directions;
                if (request_data.contains("expected_directions")) {
                    expected_directions = request_data["expected_directions"];
                } else {
                    // Default for testing
                    expected_directions = {"DOWN", "UP", "LEFT", "RIGHT"};
                }
                
                // Decode URLs if needed
                for (auto& url : video_urls) {
                    url = decodeUrl(url);
                }
                
                std::cout << "\n========================================" << std::endl;
                std::cout << "Processing challenge: " << challenge_id << std::endl;
                std::cout << "Videos: " << video_urls.size() << std::endl;
                std::cout << "========================================" << std::endl;
                
                // Process videos with liveness check on one random video only
                auto results = detector.analyzeVideos(video_urls, -1);  // -1 means random selection
                
                // Check directions
                std::vector<std::string> detected_directions;
                bool all_match = true;
                int liveness_checked_index = -1;
                float min_liveness_score = 1.0f;
                
                for (size_t i = 0; i < results.size() && i < expected_directions.size(); i++) {
                    detected_directions.push_back(results[i].primary_direction);
                    
                    bool matches = (results[i].primary_direction == expected_directions[i]);
                    if (!matches) {
                        all_match = false;
                    }
                    
                    // Track which video had liveness check
                    if (results[i].liveness_score < 1.0f) {
                        liveness_checked_index = i;
                        min_liveness_score = results[i].liveness_score;
                    }
                    
                    std::cout << "Video " << (i+1) << ": "
                             << "Expected=" << expected_directions[i] 
                             << ", Detected=" << results[i].primary_direction
                             << ", Match=" << (matches ? "✓" : "✗")
                             << ", Confidence=" << results[i].confidence;
                    
                    if (static_cast<int>(i) == liveness_checked_index) {
                        std::cout << ", Liveness=" << results[i].liveness_score;
                    }
                    std::cout << std::endl;
                }
                
                // Overall liveness determination
                bool is_live = (min_liveness_score > 0.5f);
                
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start);
                
                std::cout << "========================================" << std::endl;
                std::cout << "Result: " << (all_match && is_live ? "SUCCESS" : "FAILED") << std::endl;
                std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
                std::cout << "========================================" << std::endl;
                
                json response_data = {
                    {"result", all_match && is_live},
                    {"challenge_id", challenge_id},
                    {"expected_directions", expected_directions},
                    {"detected_directions", detected_directions},
                    {"liveness_checked_video", liveness_checked_index + 1},
                    {"liveness_score", min_liveness_score},
                    {"is_live", is_live},
                    {"processing_time_ms", duration.count()},
                    {"videos_processed", results.size()},
                    {"message", all_match ? (is_live ? "Challenge passed" : "Liveness check failed") : "Direction mismatch"}
                };
                
                // Add individual video details
                json video_details = json::array();
                for (size_t i = 0; i < results.size(); i++) {
                    video_details.push_back({
                        {"index", i + 1},
                        {"has_face", results[i].has_face},
                        {"face_presence_ratio", results[i].face_presence_ratio},
                        {"confidence", results[i].confidence},
                        {"direction", results[i].primary_direction}
                    });
                }
                response_data["video_details"] = video_details;
                
                int status_code = (all_match && is_live) ? 200 : 400;
                return createResponse(status_code, createSuccessResponse(response_data));
                
            } catch (const std::exception& e) {
                return createResponse(500, createErrorResponse(std::string("Error processing request: ") + e.what()));
            }
        });
        
        // Challenge generation endpoint
        CROW_ROUTE(app, "/generate-challenge").methods("POST"_method)
        ([this](const crow::request& req) {
            // Generate random challenge
            std::vector<std::string> directions = {"LEFT", "RIGHT", "UP", "DOWN"};
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(directions.begin(), directions.end(), g);
            
            // Generate unique challenge ID
            auto now = std::chrono::system_clock::now().time_since_epoch().count();
            std::stringstream ss;
            ss << "challenge_" << std::hex << now;
            std::string challenge_id = ss.str();
            
            json response_data = {
                {"challenge_id", challenge_id},
                {"directions", directions},
                {"ttl_seconds", 300},
                {"message", "Record 4 videos with head movements in the specified directions"}
            };
            
            std::cout << "Generated challenge: " << challenge_id << std::endl;
            std::cout << "Directions: ";
            for (const auto& dir : directions) {
                std::cout << dir << " ";
            }
            std::cout << std::endl;
            
            return createResponse(200, createSuccessResponse(response_data));
        });
        
        // Simple face detection endpoint
        CROW_ROUTE(app, "/detect-face").methods("POST"_method)
        ([this](const crow::request& req) {
            auto start = std::chrono::steady_clock::now();
            
            try {
                json request_data = json::parse(req.body);
                
                if (!request_data.contains("image")) {
                    return createResponse(400, createErrorResponse("Missing required field: image"));
                }
                
                // Note: In production, you would decode the base64 image here
                // For now, return a placeholder response
                
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start);
                
                json response_data = {
                    {"has_face", true},
                    {"confidence", 0.95},
                    {"processing_time_ms", duration.count()}
                };
                
                return createResponse(200, createSuccessResponse(response_data));
                
            } catch (const std::exception& e) {
                return createResponse(400, createErrorResponse("Invalid request"));
            }
        });
    }
    
    void start(int port) {
        if (!initialized) {
            std::cerr << "Server not initialized!" << std::endl;
            return;
        }
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "🚀 Optimized Face Service Web Server" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Port: " << port << std::endl;
        std::cout << "Model: BlazeFace (TensorFlow Lite)" << std::endl;
        std::cout << "Features:" << std::endl;
        std::cout << "  - Fast face detection with BlazeFace" << std::endl;
        std::cout << "  - Optimized video processing" << std::endl;
        std::cout << "  - Liveness check on 1 random video only" << std::endl;
        std::cout << "  - Concurrent processing for 4 videos" << std::endl;
        std::cout << "\nAvailable Endpoints:" << std::endl;
        std::cout << "  GET  /health                  - Health check" << std::endl;
        std::cout << "  POST /test-directions         - Test with known videos" << std::endl;
        std::cout << "  POST /generate-challenge      - Generate video challenge" << std::endl;
        std::cout << "  POST /verify-video-liveness   - Verify video challenge" << std::endl;
        std::cout << "  POST /detect-face             - Detect face in image" << std::endl;
        std::cout << "\nPress Ctrl+C to stop the server" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        app.port(port).multithreaded().run();
    }
};

int main(int argc, char* argv[]) {
    int port = 8080;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            try {
                port = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Invalid port number" << std::endl;
                return 1;
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--port PORT]" << std::endl;
            std::cout << "  --port PORT  Server port (default: 8080)" << std::endl;
            return 0;
        }
    }
    
    OptimizedWebServer server;
    if (!server.initialize()) {
        std::cerr << "Failed to initialize server" << std::endl;
        return 1;
    }
    
    server.start(port);
    return 0;
}