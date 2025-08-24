cmake_minimum_required(VERSION 3.16)
project(MLFaceService VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui objdetect calib3d)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

# dlib
find_package(dlib REQUIRED)
if(dlib_FOUND)
    message(STATUS "dlib found")
else()
    message(FATAL_ERROR "dlib not found")
endif()

# libcurl
find_package(CURL REQUIRED)
if(CURL_FOUND)
    message(STATUS "libcurl found: ${CURL_VERSION_STRING}")
else()
    message(FATAL_ERROR "libcurl not found")
endif()

# Crow (HTTP framework)
find_package(Crow QUIET)
if(NOT Crow_FOUND)
    message(STATUS "Crow not found via find_package, checking for manual installation")
    find_path(CROW_INCLUDE_DIR crow.h PATHS /usr/local/include /usr/include)
    if(NOT CROW_INCLUDE_DIR)
        message(FATAL_ERROR "Crow not found. Please install Crow framework.")
    endif()
    set(CROW_LIBRARIES "")
endif()

# nlohmann/json
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann/json not found via find_package, checking for manual installation")
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp PATHS /usr/local/include /usr/include)
    if(NOT NLOHMANN_JSON_INCLUDE_DIR)
        message(FATAL_ERROR "nlohmann/json not found. Please install nlohmann/json library.")
    endif()
endif()

# MediaPipe
find_package(PkgConfig REQUIRED)
pkg_check_modules(MEDIAPIPE mediapipe)
if(NOT MEDIAPIPE_FOUND)
    # Try to find MediaPipe manually
    find_path(MEDIAPIPE_INCLUDE_DIR mediapipe/framework/api2/node.h 
        PATHS 
        /usr/local/include 
        /usr/include
        /opt/homebrew/include
        ${CMAKE_PREFIX_PATH}/include
    )
    find_library(MEDIAPIPE_LIB mediapipe 
        PATHS 
        /usr/local/lib 
        /usr/lib
        /opt/homebrew/lib
        ${CMAKE_PREFIX_PATH}/lib
    )
    
    if(MEDIAPIPE_INCLUDE_DIR AND MEDIAPIPE_LIB)
        message(STATUS "MediaPipe found manually: ${MEDIAPIPE_LIB}")
        set(MEDIAPIPE_LIBRARIES ${MEDIAPIPE_LIB})
        set(MEDIAPIPE_INCLUDE_DIRS ${MEDIAPIPE_INCLUDE_DIR})
    else()
        message(WARNING "MediaPipe not found. Video liveness detection will be disabled.")
        set(MEDIAPIPE_FOUND FALSE)
    endif()
else()
    message(STATUS "MediaPipe found via pkg-config")
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIRS})

if(CROW_INCLUDE_DIR)
    include_directories(${CROW_INCLUDE_DIR})
endif()

if(NLOHMANN_JSON_INCLUDE_DIR)
    include_directories(${NLOHMANN_JSON_INCLUDE_DIR})
endif()

if(MEDIAPIPE_INCLUDE_DIRS)
    include_directories(${MEDIAPIPE_INCLUDE_DIRS})
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/image_processor.cpp
    src/face_recognizer.cpp
    src/liveness_detector.cpp
    src/video_liveness_detector.cpp
    src/web_server.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBS}
    dlib::dlib
    ${CURL_LIBRARIES}
    Threads::Threads
)

# Link MediaPipe if found
if(MEDIAPIPE_FOUND AND MEDIAPIPE_LIBRARIES)
    target_link_libraries(${PROJECT_NAME} ${MEDIAPIPE_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME} PRIVATE MEDIAPIPE_AVAILABLE)
endif()

# Link Crow if found via find_package
if(Crow_FOUND)
    target_link_libraries(${PROJECT_NAME} Crow::Crow)
else()
    # Manual linking for Crow dependencies
    target_link_libraries(${PROJECT_NAME} pthread)
endif()

# nlohmann/json linking
if(nlohmann_json_FOUND)
    target_link_libraries(${PROJECT_NAME} nlohmann_json::nlohmann_json)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
endif()

# Install rules
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Configuration Summary ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "============================")
message(STATUS "")

# Create models directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/models)

# Custom target to download models
add_custom_target(download_models
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/models/download_models.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Downloading ML models"
)

# Custom target for running the application
add_custom_target(run
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME} --models ${CMAKE_CURRENT_BINARY_DIR}/models
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running ML Face Service"
)
