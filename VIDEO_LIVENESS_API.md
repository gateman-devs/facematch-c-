# Video Liveness Detection API

This document describes the new video liveness detection functionality that has been added to the Gateman Face Recognition service.

## Overview

The video liveness detection feature analyzes video files to determine if they contain a live person performing head movements (yaw and pitch rotations). This helps prevent spoofing attacks using static images or pre-recorded videos.

## API Endpoint

### POST `/liveness/video`

Analyzes a video file for liveness by detecting head pose movements and calculating yaw/pitch ranges.

#### Request Body

```json
[
  {
    "step": 1,
    "video_url": "https://example.com/video1.mp4"
  },
  {
    "step": 2,
    "video_url": "https://example.com/video2.mp4"
  },
  {
    "step": 3,
    "video_url": "https://example.com/video3.mp4"
  },
  {
    "step": 4,
    "video_url": "https://example.com/video4.mp4"
  }
]
```

#### Response

```json
[
  {
    "step": 1,
    "direction": "left",
    "magnitude": 8.5
  },
  {
    "step": 2,
    "direction": "right", 
    "magnitude": 7.2
  },
  {
    "step": 3,
    "direction": "up",
    "magnitude": 6.8
  },
  {
    "step": 4,
    "direction": "down",
    "magnitude": 9.1
  }
]
```

#### Response Fields

Each object in the response array contains:

- `step` (number): The step number from the request
- `direction` (string): The detected movement direction ("left", "right", "up", "down", or "none")  
- `magnitude` (number): The magnitude of movement in degrees (0.0 if no valid movement detected)

#### Movement Detection Logic

For each video:
1. All detected movements are analyzed
2. Movements with magnitude less than 5.0 are filtered out
3. From the remaining valid movements, the earliest one (by start time) is selected
4. If no valid movements are found, direction is "none" and magnitude is 0.0

## Liveness Detection Criteria

The system determines liveness based on several factors:

### Movement Thresholds
- **Minimum Yaw Range**: 15° (left-right head rotation)
- **Minimum Pitch Range**: 10° (up-down head movement)
- **Minimum Duration**: 1.0 seconds
- **Minimum Frames**: 30 frames with detected faces

### Analysis Process
1. **Video Download**: Downloads video from provided URL
2. **Frame Processing**: Extracts frames and processes at ~10 FPS
3. **Face Detection**: Detects faces in each frame
4. **Head Pose Estimation**: Calculates yaw, pitch, and roll angles
5. **Movement Analysis**: Analyzes movement patterns over time
6. **Liveness Decision**: Determines if movements indicate a live person

## Implementation Details

### Technology Stack
- **Primary**: MediaPipe for enhanced face mesh detection (if available)
- **Fallback**: OpenCV-based head pose estimation using facial landmarks
- **Video Processing**: OpenCV VideoCapture
- **Network**: libcurl for video downloads

### Head Pose Calculation
The system uses computer vision techniques to estimate head pose:

1. **Face Detection**: Detects faces using Haar cascades or MediaPipe
2. **Landmark Detection**: Identifies key facial landmarks
3. **3D Model Fitting**: Fits a 3D face model to 2D landmarks
4. **Pose Estimation**: Calculates rotation angles using PnP algorithm

### Key Features
- **URL Support**: Downloads videos from any accessible URL
- **Multiple Formats**: Supports common video formats (MP4, MOV, AVI, etc.)
- **Robust Analysis**: Handles various lighting conditions and video qualities
- **Efficient Processing**: Processes subset of frames for faster analysis
- **Comprehensive Output**: Provides detailed movement analysis

## Error Handling

The API handles various error conditions:

### Common Failure Reasons
- `detector_not_initialized`: Video detector not properly initialized
- `failed_to_download_video`: Unable to download video from URL
- `failed_to_open_video_file`: Video file is corrupt or unsupported format
- `video_too_short`: Video duration less than minimum requirement
- `insufficient_face_detection`: Too few faces detected in video frames
- `error_processing_video_url`: Generic error processing video URL
- `error_during_analysis`: Error during video analysis

### HTTP Status Codes
- `200`: Success
- `400`: Bad request (invalid input)
- `503`: Service unavailable (video detection not available)
- `500`: Internal server error

## Testing

Use the provided test script to verify the implementation:

```bash
# Basic test with default video
python3 test_video_liveness.py

# Test with custom server and video
python3 test_video_liveness.py http://localhost:8080 https://example.com/test-video.mp4
```

### Test Video
The implementation has been tested with the provided Cloudinary video:
```
https://res.cloudinary.com/themizehq/video/upload/v1755671605/IMG_6487.mov
```

## Installation Requirements

### Dependencies
- OpenCV 4.x
- libcurl
- MediaPipe (optional, enhances accuracy)

### macOS Installation
```bash
# Install dependencies
./setup_environment.sh

# Build project
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Verifying Installation
Check if video liveness detection is available:

```bash
curl http://localhost:8080/health
```

Look for `"video_liveness_available": true` in the response.

## Performance Considerations

### Processing Time
- Typical processing: 2-5 seconds for 3-5 second videos
- Depends on video resolution, duration, and system performance
- Processing is done at reduced frame rate (~10 FPS) for efficiency

### Resource Usage
- Memory: ~50-100MB additional for video processing
- CPU: Moderate usage during video analysis
- Network: Downloads entire video before processing

### Optimization Tips
- Use shorter videos (3-5 seconds) for faster processing
- Lower resolution videos process faster
- Ensure stable network connection for video downloads

## Security Considerations

### Video Validation
- Only processes videos from HTTPS URLs (recommended)
- Validates video format and size
- Temporary files are cleaned up after processing

### Anti-Spoofing Features
- Analyzes natural head movement patterns
- Detects insufficient or artificial movements
- Combines multiple detection methods for robustness

## Future Enhancements

Potential improvements for future versions:
- Support for uploaded video files (multipart/form-data)
- Real-time video stream analysis
- Enhanced MediaPipe integration
- Machine learning-based movement pattern analysis
- Support for additional biometric indicators (eye blinks, facial expressions)

