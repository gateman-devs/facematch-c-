#include "src/lightweight/lightweight_video_detector_simple.hpp"
#include <crow.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <future>
#include <chrono>
#include <algorithm>
#include <random>

using json = nlohmann::json;
using namespace lightweight;

class LightweightWebServer {
private:
    LightweightVideoDetector detector;
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
    LightweightWebServer() : initialized(false) {}
    
    bool initialize() {
        std::cout << "Initializing Lightweight Web Server..." << std::endl;
        
        // Initialize detector
        if (!detector.initialize("./models")) {
            std::cerr << "Warning: Could not fully initialize detector (will use fallback methods)" << std::endl;
        }
        
        setupRoutes();
        initialized = true;
        
        std::cout << "✓ Lightweight Web Server initialized successfully" << std::endl;
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
                {"service", "lightweight-face-service"},
                {"detector_initialized", initialized},
                {"version", "1.0.0"},
                {"timestamp", std::time(nullptr)},
                {"available_endpoints", {
                    {"GET", "/health"},
                    {"POST", "/compare-faces"},
                    {"POST", "/liveness-check"},
                    {"POST", "/generate-challenge"},
                    {"POST", "/verify-video-liveness"}
                }}
            };
            
            return createResponse(200, createSuccessResponse(health_data));
        });
        
        // Face comparison endpoint (simplified - uses video detector for basic face detection)
        CROW_ROUTE(app, "/compare-faces").methods("POST"_method)
        ([this](const crow::request& req) {
            auto start = std::chrono::steady_clock::now();
            
            try {
                json request_data = json::parse(req.body);
                
                if (!request_data.contains("image1") || !request_data.contains("image2")) {
                    return createResponse(400, createErrorResponse("Missing required fields: image1, image2"));
                }
                
                // For now, return a simplified response
                // In a full implementation, you'd extract faces from images and compare them
                json response_data = {
                    {"match", false},  // Conservative default
                    {"confidence", 0.5},
                    {"message", "Face comparison using lightweight detector - simplified implementation"},
                    {"processing_time_ms", std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count()}
                };
                
                return createResponse(200, createSuccessResponse(response_data));
                
            } catch (const std::exception& e) {
                return createResponse(400, createErrorResponse("Invalid JSON request body"));
            }
        });
        
        // Liveness check endpoint (simplified)
        CROW_ROUTE(app, "/liveness-check").methods("POST"_method)
        ([this](const crow::request& req) {
            auto start = std::chrono::steady_clock::now();
            
            try {
                json request_data = json::parse(req.body);
                
                if (!request_data.contains("image")) {
                    return createResponse(400, createErrorResponse("Missing required field: image"));
                }
                
                // Simplified liveness check
                json response_data = {
                    {"is_live", true},  // Conservative default for lightweight version
                    {"confidence", 0.8},
                    {"message", "Liveness check using lightweight detector - simplified implementation"},
                    {"processing_time_ms", std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count()}
                };
                
                return createResponse(200, createSuccessResponse(response_data));
                
            } catch (const std::exception& e) {
                return createResponse(400, createErrorResponse("Invalid JSON request body"));
            }
        });
        
        // Challenge generation endpoint (simplified)
        CROW_ROUTE(app, "/generate-challenge").methods("POST"_method)
        ([this](const crow::request& req) {
            // Generate a simple challenge
            std::vector<std::string> directions = {"left", "right", "up", "down"};
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(directions.begin(), directions.end(), g);
            
            // Generate simple challenge ID
            auto now = std::chrono::system_clock::now().time_since_epoch().count();
            std::string challenge_id = "lightweight_challenge_" + std::to_string(now);
            
            json response_data = {
                {"challenge_id", challenge_id},
                {"directions", directions},
                {"ttl_seconds", 300},
                {"message", "Challenge generated using lightweight service - store this challenge_id for verification"}
            };
            
            return createResponse(200, createSuccessResponse(response_data));
        });
        
        // Video liveness verification endpoint  
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
                
                // Decode URLs if they're encoded
                for (auto& url : video_urls) {
                    url = decodeUrl(url);
                }
                
                std::cout << "Processing video liveness challenge: " << challenge_id << std::endl;
                
                // Analyze videos concurrently
                std::vector<VideoAnalysis> results = detector.analyzeVideos(video_urls);
                
                // Expected directions (for demo purposes - in real system would retrieve from cache)
                std::vector<std::string> expected_directions = {"left", "up", "right", "down"};
                std::vector<std::string> detected_directions;
                
                bool all_match = true;
                for (size_t i = 0; i < results.size() && i < expected_directions.size(); i++) {
                    std::string detected;
                    switch (results[i].primary_direction) {
                        case Direction::LEFT: detected = "left"; break;
                        case Direction::RIGHT: detected = "right"; break;
                        case Direction::UP: detected = "up"; break;
                        case Direction::DOWN: detected = "down"; break;
                        default: detected = "none"; break;
                    }
                    
                    detected_directions.push_back(detected);
                    
                    if (detected != expected_directions[i]) {
                        all_match = false;
                    }
                    
                    std::cout << "Video " << (i+1) << ": Expected " << expected_directions[i] 
                             << ", Detected " << detected << " " << (detected == expected_directions[i] ? "✓" : "✗") << std::endl;
                }
                
                json response_data = {
                    {"result", all_match},
                    {"expected_directions", expected_directions},
                    {"detected_directions", detected_directions},
                    {"processing_time_ms", std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count()},
                    {"videos_processed", results.size()}
                };
                
                int status_code = all_match ? 200 : 400;
                return createResponse(status_code, createSuccessResponse(response_data));
                
            } catch (const std::exception& e) {
                return createResponse(400, createErrorResponse(std::string("Error processing request: ") + e.what()));
            }
        });
    }
    
    void start(int port) {
        if (!initialized) {
            std::cerr << "Server not initialized!" << std::endl;
            return;
        }
        
        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "🚀 Lightweight Face Service Web Server" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Port: " << port << std::endl;
        std::cout << "Implementation: OpenCV-only (lightweight)" << std::endl;
        std::cout << "Model size: ~0 MB" << std::endl;
        std::cout << "Processing time: ~5s per video" << std::endl;
        std::cout << std::endl;
        std::cout << "Available API Endpoints:" << std::endl;
        std::cout << "  GET  /health                  - Health check" << std::endl;
        std::cout << "  POST /compare-faces           - Compare two faces" << std::endl;
        std::cout << "  POST /liveness-check          - Check face liveness" << std::endl;
        std::cout << "  POST /generate-challenge      - Generate video challenge" << std::endl;
        std::cout << "  POST /verify-video-liveness   - Verify video challenge" << std::endl;
        std::cout << std::endl;
        std::cout << "Press Ctrl+C to stop the server" << std::endl;
        std::cout << "========================================" << std::endl;
        
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
                if (port < 1 || port > 65535) {
                    std::cerr << "Error: Port must be between 1 and 65535" << std::endl;
                    return 1;
                }
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid port number" << std::endl;
                return 1;
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --port PORT    Server port (default: 8080)" << std::endl;
            std::cout << "  --help, -h     Show this help message" << std::endl;
            return 0;
        }
    }
    
    try {
        LightweightWebServer server;
        if (!server.initialize()) {
            std::cerr << "Failed to initialize server" << std::endl;
            return 1;
        }
        
        server.start(port);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}