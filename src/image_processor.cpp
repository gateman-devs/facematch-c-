#include "image_processor.hpp"
#include <curl/curl.h>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <algorithm>
#include <regex>
#include <cstring>

ImageProcessor::ImageProcessor() {
    // Initialize libcurl
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

cv::Mat ImageProcessor::loadImage(const std::string& input) {
    if (isBase64(input)) {
        return decodeBase64(input);
    } else {
        return downloadImage(input);
    }
}

std::future<cv::Mat> ImageProcessor::loadImageAsync(const std::string& input) {
    return std::async(std::launch::async, [this, input]() {
        return loadImage(input);
    });
}

std::vector<std::future<cv::Mat>> ImageProcessor::loadImagesAsync(const std::vector<std::string>& inputs) {
    std::vector<std::future<cv::Mat>> futures;
    for (const auto& input : inputs) {
        futures.push_back(loadImageAsync(input));
    }
    return futures;
}

bool ImageProcessor::validateImage(const cv::Mat& image) {
    if (image.empty()) {
        return false;
    }
    
    // Check minimum dimensions
    if (image.rows < 100 || image.cols < 100) {
        return false;
    }
    
    // Check maximum dimensions (prevent memory issues)
    if (image.rows > 4096 || image.cols > 4096) {
        return false;
    }
    
    return true;
}

float ImageProcessor::getImageQuality(const cv::Mat& image) {
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
    
    double variance = stddev.val[0] * stddev.val[0];
    
    // Normalize to 0-1 range (empirically determined threshold)
    float quality = static_cast<float>(std::min(variance / 1000.0, 1.0));
    
    return quality;
}

bool ImageProcessor::isBase64(const std::string& input) {
    // Check for data URL prefix
    return input.find("data:image/") == 0 && input.find("base64,") != std::string::npos;
}

cv::Mat ImageProcessor::decodeBase64(const std::string& base64_string) {
    try {
        // Extract base64 data after "base64,"
        size_t comma_pos = base64_string.find("base64,");
        if (comma_pos == std::string::npos) {
            return cv::Mat();
        }
        
        std::string base64_data = base64_string.substr(comma_pos + 7);
        
        // Decode base64
        std::vector<unsigned char> decoded_data;
        decoded_data.resize((base64_data.length() * 3) / 4);
        
        // Simple base64 decoding (in production, use a proper library)
        const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::vector<int> lookup(256, -1);
        for (int i = 0; i < 64; i++) {
            lookup[chars[i]] = i;
        }
        
        int val = 0, valb = -8;
        size_t out_len = 0;
        for (unsigned char c : base64_data) {
            if (lookup[c] == -1) break;
            val = (val << 6) + lookup[c];
            valb += 6;
            if (valb >= 0) {
                decoded_data[out_len++] = char((val >> valb) & 0xFF);
                valb -= 8;
            }
        }
        
        decoded_data.resize(out_len);
        
        // Check size limit
        if (decoded_data.size() > MAX_IMAGE_SIZE) {
            std::cerr << "Image size exceeds limit: " << decoded_data.size() << " bytes" << std::endl;
            return cv::Mat();
        }
        
        // Decode image from memory
        cv::Mat image = cv::imdecode(decoded_data, cv::IMREAD_COLOR);
        return image;
        
    } catch (const std::exception& e) {
        std::cerr << "Error decoding base64 image: " << e.what() << std::endl;
        return cv::Mat();
    }
}

// Callback function to write data from curl
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::vector<unsigned char>* userp) {
    size_t total_size = size * nmemb;
    size_t old_size = userp->size();
    userp->resize(old_size + total_size);
    std::memcpy(&((*userp)[old_size]), contents, total_size);
    return total_size;
}

cv::Mat ImageProcessor::downloadImage(const std::string& url) {
    try {
        CURL* curl = curl_easy_init();
        if (!curl) {
            std::cerr << "Failed to initialize curl" << std::endl;
            return cv::Mat();
        }
        
        std::vector<unsigned char> image_data;
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &image_data);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
        curl_easy_setopt(curl, CURLOPT_MAXFILESIZE, MAX_IMAGE_SIZE);
        
        CURLcode res = curl_easy_perform(curl);
        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        
        curl_easy_cleanup(curl);
        
        if (res != CURLE_OK || response_code != 200) {
            std::cerr << "Failed to download image from URL: " << url << std::endl;
            return cv::Mat();
        }
        
        if (image_data.empty()) {
            std::cerr << "Downloaded image is empty" << std::endl;
            return cv::Mat();
        }
        
        // Decode image from memory
        cv::Mat image = cv::imdecode(image_data, cv::IMREAD_COLOR);
        return image;
        
    } catch (const std::exception& e) {
        std::cerr << "Error downloading image: " << e.what() << std::endl;
        return cv::Mat();
    }
}

bool ImageProcessor::isValidImageFormat(const std::string& format) {
    std::string lower_format = format;
    std::transform(lower_format.begin(), lower_format.end(), lower_format.begin(), ::tolower);
    
    return std::find(supported_formats.begin(), supported_formats.end(), lower_format) != supported_formats.end();
}
