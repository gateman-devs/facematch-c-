#ifndef IMAGE_PROCESSOR_HPP
#define IMAGE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <future>
#include <vector>
#include <memory>

class ImageProcessor {
public:
    ImageProcessor();
    ~ImageProcessor() = default;

    // Load image from Base64 string or URL
    cv::Mat loadImage(const std::string& input);
    
    // Async image loading for URLs
    std::future<cv::Mat> loadImageAsync(const std::string& input);
    
    // Load multiple images concurrently
    std::vector<std::future<cv::Mat>> loadImagesAsync(const std::vector<std::string>& inputs);
    
    // Validate image format and size
    bool validateImage(const cv::Mat& image);
    
    // Get image quality score
    float getImageQuality(const cv::Mat& image);

private:
    // Check if input is Base64 encoded
    bool isBase64(const std::string& input);
    
    // Decode Base64 string to cv::Mat
    cv::Mat decodeBase64(const std::string& base64_string);
    
    // Download image from URL
    cv::Mat downloadImage(const std::string& url);
    
    // Validate image format
    bool isValidImageFormat(const std::string& format);
    
    // Maximum image size in bytes (10MB)
    static constexpr size_t MAX_IMAGE_SIZE = 10 * 1024 * 1024;
    
    // Supported image formats
    std::vector<std::string> supported_formats = {"jpeg", "jpg", "png", "webp"};
};

#endif // IMAGE_PROCESSOR_HPP
