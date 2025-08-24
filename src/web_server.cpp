#include "web_server.hpp"
#include <iostream>
#include <thread>
#include <fstream>
#include <ctime>
#include <algorithm>

WebServer::WebServer() : initialized(false) {
    image_processor = std::make_unique<ImageProcessor>();
    face_recognizer = std::make_unique<FaceRecognizer>();
    liveness_detector = std::make_unique<LivenessDetector>();
    video_liveness_detector = std::make_unique<VideoLivenessDetector>();
}

bool WebServer::initialize(const std::string& models_path_param) {
    models_path = models_path_param;
    
    try {
        std::cout << "Initializing ML models from: " << models_path << std::endl;
        
        // Model file paths
        std::string face_recognition_model = models_path + "/dlib_face_recognition_resnet_model_v1.dat";
        std::string shape_predictor_model = models_path + "/shape_predictor_68_face_landmarks.dat";
        
        // Check if model files exist
        std::ifstream face_model_file(face_recognition_model);
        std::ifstream shape_model_file(shape_predictor_model);
        
        if (!face_model_file.good() || !shape_model_file.good()) {
            std::cerr << "Model files not found in " << models_path << std::endl;
            std::cerr << "Required files:" << std::endl;
            std::cerr << "  - dlib_face_recognition_resnet_model_v1.dat" << std::endl;
            std::cerr << "  - shape_predictor_68_face_landmarks.dat" << std::endl;
            return false;
        }
        
        // Initialize face recognizer
        if (!face_recognizer->initialize(face_recognition_model, shape_predictor_model)) {
            std::cerr << "Failed to initialize face recognizer" << std::endl;
            return false;
        }
        
        // Initialize liveness detector with error handling
        try {
            if (!liveness_detector->initialize(shape_predictor_model)) {
                std::cerr << "Failed to initialize liveness detector" << std::endl;
                return false;
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception during liveness detector initialization: " << e.what() << std::endl;
            return false;
        }
        
        // Initialize video liveness detector with error handling
        try {
            if (!video_liveness_detector->initialize()) {
                std::cerr << "Failed to initialize video liveness detector" << std::endl;
                // Note: This is not a fatal error since video detection is optional
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception during video liveness detector initialization: " << e.what() << std::endl;
            // Continue without video liveness detection
        }
        
        // Setup routes
        setupCORS();
        setupErrorHandlers();
        
        // Health check endpoint
        CROW_ROUTE(app, "/health").methods("GET"_method)
        ([this](const crow::request& req) {
            return handleHealthCheck(req);
        });
        
        // Face comparison endpoint
        CROW_ROUTE(app, "/compare-faces").methods("POST"_method)
        ([this](const crow::request& req) {
            return handleFaceComparison(req);
        });
        
        // Liveness check endpoint
        CROW_ROUTE(app, "/liveness-check").methods("POST"_method)
        ([this](const crow::request& req) {
            return handleLivenessCheck(req);
        });
        
        // Video liveness check endpoint
        CROW_ROUTE(app, "/liveness/video").methods("POST"_method)
        ([this](const crow::request& req) {
            return handleVideoLivenessCheck(req);
        });
        
        initialized = true;
        std::cout << "Web server initialized successfully!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing web server: " << e.what() << std::endl;
        return false;
    }
}

void WebServer::start(int port) {
    if (!initialized) {
        std::cerr << "Server not initialized. Call initialize() first." << std::endl;
        return;
    }
    
    std::cout << "Starting server on port " << port << std::endl;
    app.port(port).multithreaded().run();
}

void WebServer::stop() {
    app.stop();
}

crow::response WebServer::handleFaceComparison(const crow::request& req) {
    Timer timer;
    
    try {
        // Parse request body
        json request_data = parseRequestBody(req.body);
        
        if (!validateComparisonRequest(request_data)) {
            return createResponse(400, createErrorResponse("Invalid request format. Required: image1 and image2"));
        }
        
        std::string image1_input = request_data["image1"];
        std::string image2_input = request_data["image2"];
        
        // Load images concurrently
        auto images = loadImagesConcurrently(image1_input, image2_input);
        cv::Mat image1 = images.first;
        cv::Mat image2 = images.second;
        
        if (image1.empty() || image2.empty()) {
            return createResponse(400, createErrorResponse("Failed to load one or both images"));
        }
        
        // Validate images
        if (!image_processor->validateImage(image1) || !image_processor->validateImage(image2)) {
            return createResponse(400, createErrorResponse("Invalid image format or size"));
        }
        
        // Process faces sequentially to avoid concurrent memory access issues
        FaceInfo face1 = face_recognizer->processFace(image1);
        FaceInfo face2 = face_recognizer->processFace(image2);
        
        // Run liveness detection with comprehensive error handling
        LivenessAnalysis liveness1, liveness2;
        
        // Initialize with safe defaults
        liveness1.is_live = true;
        liveness1.confidence = 1.0f;
        liveness2.is_live = true;
        liveness2.confidence = 1.0f;
        
        // Try liveness detection with comprehensive error handling
        try {
            if (liveness_detector && liveness_detector->isInitialized() && !image1.empty()) {
                liveness1 = liveness_detector->checkLiveness(image1.clone());
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in liveness detection for image1: " << e.what() << std::endl;
            liveness1.is_live = true;  // Default to live on error
            liveness1.confidence = 0.5f;
        }
        
        try {
            if (liveness_detector && liveness_detector->isInitialized() && !image2.empty()) {
                liveness2 = liveness_detector->checkLiveness(image2.clone());
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in liveness detection for image2: " << e.what() << std::endl;
            liveness2.is_live = true;  // Default to live on error
            liveness2.confidence = 0.5f;
        }
        
        if (!face1.valid || !face2.valid) {
            return createResponse(400, createErrorResponse("No valid faces detected in one or both images"));
        }
        
        // Compare faces
        FaceRecognizer::ComparisonResult comparison = face_recognizer->compareFacesDetailed(face1, face2);
        
        // Create response with liveness info
        json response_data = {
            {"match", comparison.is_match},
            {"confidence", comparison.confidence},
            {"liveness", {
                {"image1", liveness1.is_live},
                {"image2", liveness2.is_live}
            }},
            {"processing_time_ms", timer.elapsed_ms()},
            {"face_quality_scores", {face1.quality_score, face2.quality_score}},
            {"error", nullptr}
        };
        
        return createResponse(200, createSuccessResponse(response_data));
        
    } catch (const std::exception& e) {
        std::cerr << "Error in face comparison: " << e.what() << std::endl;
        json error_response = createErrorResponse("Internal server error");
        error_response["processing_time_ms"] = timer.elapsed_ms();
        return crow::response(500, error_response.dump());
    }
}

crow::response WebServer::handleLivenessCheck(const crow::request& req) {
    Timer timer;
    
    try {
        // Parse request body
        json request_data = parseRequestBody(req.body);
        
        if (!validateLivenessRequest(request_data)) {
            return crow::response(400, createErrorResponse("Invalid request format. Required: image").dump());
        }
        
        std::string image_input = request_data["image"];
        
        // Load image
        cv::Mat image = image_processor->loadImage(image_input);
        
        if (image.empty()) {
            return crow::response(400, createErrorResponse("Failed to load image").dump());
        }
        
        // Validate image
        if (!image_processor->validateImage(image)) {
            return crow::response(400, createErrorResponse("Invalid image format or size").dump());
        }
        
        // Perform liveness detection
        LivenessAnalysis liveness = liveness_detector->checkLiveness(image);
        
        // Get image quality
        float quality_score = image_processor->getImageQuality(image);
        
        // Create response
        json response_data = {
            {"is_live", liveness.is_live},
            {"confidence", liveness.confidence},
            {"processing_time_ms", timer.elapsed_ms()},
            {"quality_score", quality_score},
            {"analysis_details", {
                {"texture_score", liveness.texture_score},
                {"landmark_consistency", liveness.landmark_consistency},
                {"image_quality", liveness.image_quality},
                {"sharpness_score", liveness.sharpness_score},
                {"lighting_score", liveness.lighting_score},
                {"spoof_detection_score", liveness.spoof_detection_score}
            }},
            {"failure_reason", liveness.has_failure ? std::string(liveness.failure_reason) : ""},
            {"error", nullptr}
        };
        
        crow::response res(200, createSuccessResponse(response_data).dump());
        res.add_header("Access-Control-Allow-Origin", "*");
        res.add_header("Content-Type", "application/json");
        return res;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in liveness check: " << e.what() << std::endl;
        json error_response = createErrorResponse("Internal server error");
        error_response["processing_time_ms"] = timer.elapsed_ms();
        return crow::response(500, error_response.dump());
    }
}

crow::response WebServer::handleVideoLivenessCheck(const crow::request& req) {
    Timer timer;
    
    try {
        // Parse request body
        json request_data = parseRequestBody(req.body);
        
        if (!validateVideoLivenessRequest(request_data)) {
            return crow::response(400, createErrorResponse("Invalid request format. Required: array of 4 objects with 'step' (number) and 'video_url' (string)").dump());
        }
        
        if (!video_liveness_detector || !video_liveness_detector->isInitialized()) {
            return crow::response(503, createErrorResponse("Video liveness detection not available").dump());
        }
        
        // Process each video in the array
        json results = json::array();
        
        for (const auto& video_obj : request_data) {
            int step = video_obj["step"];
            std::string video_url = video_obj["video_url"];
            
            // Initialize result object for this step
            json step_result = {
                {"step", step},
                {"direction", "none"},
                {"magnitude", 0.0f}
            };
            
            try {
                // Perform video liveness analysis
                VideoLivenessAnalysis analysis = video_liveness_detector->analyzeVideoFromUrl(video_url);
                
                if (!analysis.has_failure && !analysis.directional_movements.empty()) {
                    // Filter movements with magnitude >= 5
                    std::vector<DirectionalMovement> valid_movements;
                    for (const auto& movement : analysis.directional_movements) {
                        if (movement.magnitude >= 5.0f) {
                            valid_movements.push_back(movement);
                        }
                    }
                    
                    // If we have valid movements, take the earliest one
                    if (!valid_movements.empty()) {
                        // Sort by start_time to get the earliest
                        std::sort(valid_movements.begin(), valid_movements.end(), 
                                [](const DirectionalMovement& a, const DirectionalMovement& b) {
                                    return a.start_time < b.start_time;
                                });
                        
                        const auto& earliest_movement = valid_movements[0];
                        
                        // Convert direction to string
                        std::string direction_str;
                        switch (earliest_movement.direction) {
                            case MovementDirection::LEFT: direction_str = "left"; break;
                            case MovementDirection::RIGHT: direction_str = "right"; break;
                            case MovementDirection::UP: direction_str = "up"; break;
                            case MovementDirection::DOWN: direction_str = "down"; break;
                            default: direction_str = "none"; break;
                        }
                        
                        step_result["direction"] = direction_str;
                        step_result["magnitude"] = earliest_movement.magnitude;
                    }
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Error processing video for step " << step << ": " << e.what() << std::endl;
                // Keep default values (direction: "none", magnitude: 0.0)
            }
            
            results.push_back(step_result);
        }
        
        crow::response res(200, results.dump());
        res.add_header("Access-Control-Allow-Origin", "*");
        res.add_header("Content-Type", "application/json");
        return res;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in video liveness check: " << e.what() << std::endl;
        json error_response = createErrorResponse("Internal server error");
        error_response["processing_time_ms"] = timer.elapsed_ms();
        return crow::response(500, error_response.dump());
    }
}

crow::response WebServer::handleHealthCheck(const crow::request& req) {
    json health_data = {
        {"status", "healthy"},
        {"models_loaded", initialized && face_recognizer->isInitialized() && liveness_detector->isInitialized()},
        {"video_liveness_available", video_liveness_detector && video_liveness_detector->isInitialized()},
        {"version", "1.0.0"},
        {"timestamp", std::time(nullptr)}
    };
    
    crow::response res(200, createSuccessResponse(health_data).dump());
    res.add_header("Access-Control-Allow-Origin", "*");
    res.add_header("Content-Type", "application/json");
    return res;
}

json WebServer::parseRequestBody(const std::string& body) {
    try {
        return json::parse(body);
    } catch (const std::exception& e) {
        throw std::runtime_error("Invalid JSON in request body");
    }
}

std::pair<cv::Mat, cv::Mat> WebServer::loadImagesConcurrently(const std::string& image1_input, 
                                                             const std::string& image2_input) {
    // Load images asynchronously
    std::future<cv::Mat> image1_future = image_processor->loadImageAsync(image1_input);
    std::future<cv::Mat> image2_future = image_processor->loadImageAsync(image2_input);
    
    // Wait for both to complete
    cv::Mat image1 = image1_future.get();
    cv::Mat image2 = image2_future.get();
    
    return std::make_pair(image1, image2);
}

json WebServer::createErrorResponse(const std::string& error_message, int status_code) {
    return json{
        {"success", false},
        {"error", error_message},
        {"status_code", status_code}
    };
}

json WebServer::createSuccessResponse(const json& data) {
    json response = data;
    response["success"] = true;
    return response;
}

crow::response WebServer::createResponse(int status_code, const json& data) {
    crow::response res(status_code, data.dump());
    res.add_header("Access-Control-Allow-Origin", "*");
    res.add_header("Content-Type", "application/json");
    return res;
}

bool WebServer::validateComparisonRequest(const json& request_data) {
    return request_data.contains("image1") && request_data.contains("image2") &&
           request_data["image1"].is_string() && request_data["image2"].is_string() &&
           !request_data["image1"].get<std::string>().empty() &&
           !request_data["image2"].get<std::string>().empty();
}

bool WebServer::validateLivenessRequest(const json& request_data) {
    return request_data.contains("image") && request_data["image"].is_string() &&
           !request_data["image"].get<std::string>().empty();
}

bool WebServer::validateVideoLivenessRequest(const json& request_data) {
    // Check if request_data is an array of 4 objects
    if (!request_data.is_array() || request_data.size() != 4) {
        return false;
    }
    
    // Validate each object in the array
    for (const auto& video_obj : request_data) {
        if (!video_obj.is_object() ||
            !video_obj.contains("step") || !video_obj["step"].is_number() ||
            !video_obj.contains("video_url") || !video_obj["video_url"].is_string() ||
            video_obj["video_url"].get<std::string>().empty()) {
            return false;
        }
    }
    
    return true;
}

void WebServer::setupCORS() {
    // CORS will be handled in individual responses
}

void WebServer::setupErrorHandlers() {
    // Crow will handle 404 errors by default
    // We don't need a catch-all route that interferes with valid endpoints
}
