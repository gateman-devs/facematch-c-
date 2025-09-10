#include "src/lightweight/lightweight_video_detector_simple.hpp"
#include <iostream>
#include <iomanip>

using namespace lightweight;

int main() {
    std::cout << "\n==== Video Movement Analysis Diagnostic ====\n" << std::endl;

    // Test videos with expected movements
    std::vector<std::pair<std::string, std::string>> test_videos = {
        {"https://res.cloudinary.com/themizehq/video/upload/v1755978338/IMG_6552.mov", "DOWN"},
        {"https://res.cloudinary.com/themizehq/video/upload/v1755978338/IMG_6551.mov", "UP"},
        {"https://res.cloudinary.com/themizehq/video/upload/v1755978331/IMG_6553.mov", "LEFT"},
        {"https://res.cloudinary.com/themizehq/video/upload/v1755978327/IMG_6554.mov", "RIGHT"}
    };

    LightweightVideoDetector detector;
    detector.setFrameSampleRate(1);  // Process every frame for detailed analysis
    detector.setMinMovementThreshold(0.01f);  // Very low threshold to catch all movements
    detector.setMaxFramesToAnalyze(30);  // More frames for analysis

    if (!detector.initialize("./models")) {
        std::cout << "Warning: Could not fully initialize (will use fallback methods)\n";
    }

    for (size_t i = 0; i < test_videos.size(); i++) {
        const auto& [url, expected] = test_videos[i];
        
        std::cout << "\n=== Analyzing " << expected << " movement video ===" << std::endl;
        std::cout << "URL: " << url << std::endl;
        
        auto analysis = detector.analyzeVideo(url);
        
        std::cout << "Results:" << std::endl;
        std::cout << "  - Total frames: " << analysis.total_frames_analyzed << std::endl;
        std::cout << "  - Frames with face: " << analysis.frames_with_face << std::endl;
        std::cout << "  - Movements detected: " << analysis.movements.size() << std::endl;
        std::cout << "  - Primary direction: " << static_cast<int>(analysis.primary_direction) << " (0=NONE, 1=LEFT, 2=RIGHT, 3=UP, 4=DOWN)" << std::endl;
        std::cout << "  - Processing time: " << analysis.total_processing_time_ms << "ms" << std::endl;
        
        if (!analysis.movements.empty()) {
            std::cout << "  - Movement details:" << std::endl;
            for (size_t j = 0; j < std::min(analysis.movements.size(), size_t(5)); j++) {
                const auto& mov = analysis.movements[j];
                std::cout << "    Frame " << j+1 << ": dx=" << std::fixed << std::setprecision(4) 
                         << mov.delta_x << ", dy=" << mov.delta_y 
                         << ", confidence=" << mov.confidence 
                         << ", direction=" << static_cast<int>(mov.direction) << std::endl;
            }
        }
        
        std::cout << std::endl;
    }
    
    return 0;
}