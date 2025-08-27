#include "web_server.hpp"
#include <iostream>
#include <string>
#include <filesystem>
#include <signal.h>

// Global server instance for signal handling
std::unique_ptr<WebServer> global_server;

void signalHandler(int signal) {
    std::cout << "\nReceived signal " << signal << ". Shutting down gracefully..." << std::endl;
    if (global_server) {
        global_server->stop();
    }
    exit(0);
}

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n"
              << "Options:\n"
              << "  --port PORT        Server port (default: 8080)\n"
              << "  --models PATH      Path to models directory (default: ./models)\n"
              << "  --help             Show this help message\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    // Default configuration
    int port = 8080;
    std::string models_path = "./models";
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--port" && i + 1 < argc) {
            try {
                port = std::stoi(argv[++i]);
                if (port < 1 || port > 65535) {
                    std::cerr << "Error: Port must be between 1 and 65535" << std::endl;
                    return 1;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid port number" << std::endl;
                return 1;
            }
        } else if (arg == "--models" && i + 1 < argc) {
            models_path = argv[++i];
        } else {
            std::cerr << "Error: Unknown argument " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Validate models directory
    if (!std::filesystem::exists(models_path)) {
        std::cerr << "Error: Models directory does not exist: " << models_path << std::endl;
        std::cerr << "Please ensure the models directory exists and contains the required model files." << std::endl;
        std::cerr << "Run the ./download_models.sh script to download models automatically." << std::endl;
        return 1;
    }
    
    // Setup signal handlers for graceful shutdown
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    try {
        std::cout << "=== ML Face Recognition Service ===" << std::endl;
        std::cout << "Port: " << port << std::endl;
        std::cout << "Models path: " << models_path << std::endl;
        std::cout << "====================================" << std::endl;
        
        // Create and initialize server
        global_server = std::make_unique<WebServer>();
        
        if (!global_server->initialize(models_path)) {
            std::cerr << "Failed to initialize server" << std::endl;
            return 1;
        }
        
        std::cout << "\nServer ready! Available endpoints:" << std::endl;
        std::cout << "  GET  /health           - Health check" << std::endl;
        std::cout << "  POST /compare-faces    - Compare two faces" << std::endl;
        std::cout << "  POST /liveness-check   - Check face liveness" << std::endl;
        std::cout << "  POST /video/liveness   - Video liveness detection" << std::endl;
        std::cout << "\nPress Ctrl+C to stop the server." << std::endl;
        
        // Start server (blocking call)
        global_server->start(port);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
