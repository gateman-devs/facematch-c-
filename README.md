# Gateman Face Recognition Service

A C++ web service for face recognition and liveness detection using OpenCV and dlib.

## Features

- **Face Comparison**: Compare two faces with confidence scoring
- **Liveness Detection**: Detect fake/printed images 
- **RESTful API**: JSON-based HTTP endpoints
- **Docker Support**: Easy deployment with Docker

## Quick Start

### Docker (Recommended)

```bash
# Build and start
docker-compose up --build

# Service available at http://localhost:8080
```

### Local Development

```bash
# Install dependencies
./setup_environment.sh

# Start service (downloads models, builds, and runs)
./startup_local.sh
```

## API Endpoints

### Health Check
```http
GET /health
```

### Face Comparison
```http
POST /compare-faces
Content-Type: application/json

{
  "image1": "data:image/jpeg;base64,..." OR "https://example.com/face1.jpg",
  "image2": "data:image/jpeg;base64,..." OR "https://example.com/face2.jpg"
}
```

### Liveness Check
```http
POST /liveness-check
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,..." OR "https://example.com/face.jpg"
}
```

### Input Formats
- Base64 data URLs: `data:image/jpeg;base64,<base64-data>`
- HTTP URLs: `https://example.com/image.jpg`
- Supported formats: JPEG, PNG, WebP

## Dependencies

- OpenCV 4.x
- dlib 19.24+
- Crow HTTP framework
- libcurl
- nlohmann/json
- CMake 3.16+

## Build & Run

### Local Development
```bash
# Install dependencies
./setup_environment.sh

# Download models and build
./startup_local.sh

# Manual build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./MLFaceService --port 8080 --models ../models
```

### Docker
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f
```

## Testing

Test the service endpoints:
```bash
./test_service.sh
```

## Configuration

Command line options:
```bash
./MLFaceService --help

Options:
  --port PORT        Server port (default: 8080)
  --models PATH      Path to models directory (default: ./models)
  --help             Show help message
```

## Models

The service uses dlib models:
- `dlib_face_recognition_resnet_model_v1.dat` (22.5MB)
- `shape_predictor_68_face_landmarks.dat` (99.7MB)

Models are automatically downloaded by the setup scripts.
