cmake_minimum_required(VERSION 3.16)
project(MLFaceService VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio video objdetect highgui ml)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

# Find libcurl
find_package(CURL REQUIRED)

# Find hiredis (Redis client) - optional for lightweight builds
find_package(PkgConfig REQUIRED)
pkg_check_modules(HIREDIS hiredis)

# Find dlib
find_package(dlib REQUIRED)

# Find nlohmann/json
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    # Try pkg-config as fallback
    pkg_check_modules(JSON REQUIRED nlohmann_json)
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/crow/include)

# Add crow as subdirectory if it exists
if(EXISTS "${CMAKE_SOURCE_DIR}/crow/CMakeLists.txt")
    add_subdirectory(crow)
    set(CROW_FOUND TRUE)
else()
    message(WARNING "Crow not found - full web server will not be available")
    set(CROW_FOUND FALSE)
endif()

# Source files for full web server
set(WEB_SERVER_SOURCES
    src/main.cpp
    src/web_server.cpp
    src/image_processor.cpp
    src/face_recognizer.cpp
    src/liveness_detector.cpp
    src/video_liveness_detector.cpp
    src/challenge_generator.cpp
    src/redis_cache.cpp
    src/challenge_verifier.cpp
)

# Source files for lightweight components
set(LIGHTWEIGHT_SOURCES
    src/lightweight/lightweight_video_detector_simple.cpp
)

# Build full web server if dependencies are available
if(CROW_FOUND AND nlohmann_json_FOUND AND HIREDIS_FOUND)
    add_executable(MLFaceService ${WEB_SERVER_SOURCES})

    target_link_libraries(MLFaceService
        ${OpenCV_LIBS}
        ${CURL_LIBRARIES}
        Crow::Crow
        Threads::Threads
        dlib::dlib
    )

    # Link hiredis libraries
    target_link_directories(MLFaceService PRIVATE ${HIREDIS_LIBRARY_DIRS})
    target_link_libraries(MLFaceService ${HIREDIS_LIBRARIES})
    target_include_directories(MLFaceService PRIVATE ${HIREDIS_INCLUDE_DIRS})

    if(nlohmann_json_FOUND)
        target_link_libraries(MLFaceService nlohmann_json::nlohmann_json)
    else()
        target_include_directories(MLFaceService PRIVATE ${JSON_INCLUDE_DIRS})
        target_link_libraries(MLFaceService ${JSON_LIBRARIES})
    endif()

    message(STATUS "✓ Full web server (MLFaceService) will be built")
else()
    if(NOT HIREDIS_FOUND)
        message(STATUS "✗ Full web server skipped - hiredis not found")
    else()
        message(STATUS "✗ Full web server skipped - missing dependencies")
    endif()
endif()

# Build lightweight executables only if source files exist (for development)
if(EXISTS "${CMAKE_SOURCE_DIR}/test_lightweight_simple.cpp")
    add_executable(test_lightweight
        test_lightweight_simple.cpp
        ${LIGHTWEIGHT_SOURCES}
    )

    target_link_libraries(test_lightweight
        ${OpenCV_LIBS}
        ${CURL_LIBRARIES}
        Threads::Threads
    )

    message(STATUS "✓ test_lightweight will be built")
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/lightweight_server.cpp")
    add_executable(lightweight_server
        lightweight_server.cpp
        ${LIGHTWEIGHT_SOURCES}
    )

    target_link_libraries(lightweight_server
        ${OpenCV_LIBS}
        ${CURL_LIBRARIES}
        Threads::Threads
    )

    message(STATUS "✓ lightweight_server will be built")
endif()

# Build lightweight web server with HTTP endpoints if Crow is available and source exists
if(CROW_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/lightweight_web_server.cpp")
    add_executable(lightweight_web_server
        lightweight_web_server.cpp
        ${LIGHTWEIGHT_SOURCES}
    )

    target_link_libraries(lightweight_web_server
        ${OpenCV_LIBS}
        ${CURL_LIBRARIES}
        Crow::Crow
        Threads::Threads
    )

    if(nlohmann_json_FOUND)
        target_link_libraries(lightweight_web_server nlohmann_json::nlohmann_json)
    endif()

    message(STATUS "✓ Lightweight web server (lightweight_web_server) will be built")
elseif(NOT EXISTS "${CMAKE_SOURCE_DIR}/lightweight_web_server.cpp")
    message(STATUS "⚠ Lightweight web server skipped - source file not found")
else()
    message(STATUS "✗ Lightweight web server skipped - Crow not found")
endif()

# Set executable properties conditionally
if(TARGET test_lightweight)
    set_target_properties(test_lightweight PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

if(TARGET lightweight_server)
    set_target_properties(lightweight_server PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

if(TARGET MLFaceService)
    set_target_properties(MLFaceService PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

# Print build configuration
message(STATUS "")
message(STATUS "=== Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Executables to build:")
if(TARGET test_lightweight)
    message(STATUS "  - test_lightweight (lightweight video test)")
else()
    message(STATUS "  - test_lightweight (skipped - source not found)")
endif()
if(TARGET lightweight_server)
    message(STATUS "  - lightweight_server (basic server, no HTTP endpoints)")
else()
    message(STATUS "  - lightweight_server (skipped - source not found)")
endif()
if(TARGET lightweight_web_server)
    message(STATUS "  - lightweight_web_server (HTTP web server with all endpoints)")
else()
    message(STATUS "  - lightweight_web_server (skipped - source not found or Crow not found)")
endif()
if(TARGET MLFaceService)
    message(STATUS "  - MLFaceService (full HTTP web server with heavy ML models)")
else()
    message(STATUS "  - MLFaceService (skipped - missing dependencies)")
endif()
message(STATUS "=============================")
message(STATUS "")