#ifndef FACE_RECOGNIZER_HPP
#define FACE_RECOGNIZER_HPP

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <vector>
#include <memory>

// Face embedding network type
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128,dlib::avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,
                            dlib::input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

struct FaceInfo {
    dlib::rectangle face_rect;
    std::vector<dlib::point> landmarks;
    dlib::matrix<float,0,1> encoding;
    float quality_score;
    bool valid;
    
    FaceInfo() : quality_score(0.0f), valid(false) {}
};

class FaceRecognizer {
public:
    FaceRecognizer();
    ~FaceRecognizer() = default;
    
    // Initialize with model files
    bool initialize(const std::string& face_recognition_model_path,
                   const std::string& shape_predictor_path);
    
    // Detect face and extract features
    FaceInfo processFace(const cv::Mat& image);
    
    // Compare two face encodings
    float compareFaces(const dlib::matrix<float,0,1>& encoding1,
                      const dlib::matrix<float,0,1>& encoding2);
    
    // Compare two faces with confidence and match decision
    struct ComparisonResult {
        bool is_match;
        float confidence;
        float distance;
    };
    
    ComparisonResult compareFacesDetailed(const FaceInfo& face1, const FaceInfo& face2);
    
    // Get face quality score
    float getFaceQuality(const cv::Mat& image, const dlib::rectangle& face_rect);
    
    // Check if models are loaded
    bool isInitialized() const { return models_loaded; }

private:
    // Models
    dlib::frontal_face_detector face_detector;
    dlib::shape_predictor pose_model;
    anet_type face_encoder;
    
    bool models_loaded;
    
    // Configuration
    static constexpr float MATCH_THRESHOLD = 0.6f;  // Euclidean distance threshold
    static constexpr int MIN_FACE_SIZE = 80;        // Minimum face size in pixels
    
    // Helper methods
    bool isValidFace(const dlib::rectangle& face_rect, const cv::Mat& image);
    float calculateFaceQuality(const dlib::matrix<dlib::rgb_pixel>& face_chip);
    std::vector<dlib::point> getLandmarks(const cv::Mat& image, const dlib::rectangle& face_rect);
};

#endif // FACE_RECOGNIZER_HPP
