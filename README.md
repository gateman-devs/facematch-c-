# Gateman Face Recognition Service

A C++ web service for face recognition and liveness detection using OpenCV and dlib.

## Features

- **Face Comparison**: Compare two faces with confidence scoring
- **Liveness Detection**: Detect fake/printed images 
- **Challenge-Based Video Liveness**: Generate directional challenges and verify user compliance
- **Concurrent Video Processing**: Process multiple videos simultaneously for fast verification
- **Redis Caching**: Secure challenge storage with TTL expiration
- **RESTful API**: JSON-based HTTP endpoints
- **Docker Support**: Easy deployment with Docker

## Quick Start

### Docker (Recommended)

```bash
# Build and start (includes Redis)
docker-compose up --build

# Service available at http://localhost:8080
# Redis available at localhost:6379

# Optional: Configure Redis password
cp env.example .env
# Edit .env to set REDIS_PASSWORD
```

### Local Development

```bash
# Install dependencies (includes Redis)
./setup_environment.sh

# Setup Redis (if not auto-installed)
./setup_redis.sh

# Start service (auto-detects Redis, downloads models, builds, and runs)
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

### Challenge Generation
```http
POST /generate-challenge
Content-Type: application/json

{
  "ttl_seconds": 300  // Optional: Time to live in seconds (default: 300, range: 60-1800)
}

Response:
{
  "success": true,
  "challenge_id": "challenge_1701234567890_1234",
  "directions": ["left", "up", "right", "down"],
  "ttl_seconds": 300
}
```

### Video Liveness Verification
```http
POST /verify-video-liveness
Content-Type: application/json

{
  "challenge_id": "challenge_1701234567890_1234",
  "video_urls": [
    "https://example.com/video1.mp4",  // Video for direction 0 (left)
    "https://example.com/video2.mp4",  // Video for direction 1 (up)
    "https://example.com/video3.mp4",  // Video for direction 2 (right)
    "https://example.com/video4.mp4"   // Video for direction 3 (down)
  ]
}

Response:
{
  "success": true,
  "result": true,  // true if challenge passed, false if failed
  "expected_directions": ["left", "up", "right", "down"],
  "detected_directions": ["left", "up", "right", "down"]
}
```

## Challenge System Workflow

The challenge-based liveness detection works as follows:

1. **Generate Challenge**: Call `/generate-challenge` to get a unique challenge with 4 random directions (up, down, left, right)
2. **Record Videos**: User records 4 short videos (1.5-5 seconds each), looking in the directions specified by the challenge
3. **Verify Challenge**: Call `/verify-video-liveness` with the challenge ID and 4 video URLs
4. **Concurrent Processing**: The system processes all 4 videos simultaneously using head pose estimation
5. **Direction Verification**: Each video is analyzed to detect the primary head movement direction
6. **Result**: Returns `true` if all detected directions match the challenge, `false` otherwise

### Important Notes:
- **Redis Required**: Challenge system requires Redis for caching challenges
- Challenges expire after the specified TTL (default: 5 minutes)
- Each challenge can only be used once (deleted after verification)
- Videos must show clear head movements in the specified directions
- The system uses MediaPipe FaceMesh (if available) or OpenCV for head pose estimation
- All video processing is done concurrently for maximum performance

### Example Usage:
```bash
# 1. Generate challenge
curl -X POST http://localhost:8080/generate-challenge

# 2. User records 4 videos based on returned directions

# 3. Verify videos
curl -X POST http://localhost:8080/verify-video-liveness \
  -H "Content-Type: application/json" \
  -d '{
    "challenge_id": "challenge_1701234567890_1234",
    "video_urls": ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]
  }'
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
- hiredis (Redis client)
- Redis server (for challenge caching)
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
# Build and run (includes Redis)
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Test the Docker deployment
./test_docker.sh

# Stop services
docker-compose down

# Stop and remove volumes (Redis data)
docker-compose down -v
```

#### Manual Docker Commands
```bash
# Build the production image (includes all ML models)
docker build -t gateman-face .

# Run with Redis (recommended for full functionality)
docker run -d --name redis redis:7-alpine
docker run -d -p 8080:8080 --link redis:redis -e REDIS_HOST=redis gateman-face

# Run standalone (without Redis - challenge system disabled)
docker run -d -p 8080:8080 gateman-face

# Test the deployment
./test_docker.sh
```

#### Docker Environment Variables
- `REDIS_HOST`: Redis hostname (default: none, disables challenge system)
- `REDIS_PORT`: Redis port (default: 6379)
- `REDIS_PASSWORD`: Redis password (optional)

## CI/CD

### GitHub Actions

The project includes a unified CI/CD pipeline (`.github/workflows/deploy.prod.yml`):

- **Pull Requests**: Runs build tests and API validation without pushing to registry
- **Main Branch Pushes**: Full deployment with model downloads, security scanning, and registry push
- **Manual Triggers**: Supports workflow_dispatch for custom deployments

**Pipeline Features:**
- Downloads all required ML model files automatically
- Builds the full production image with complete ML functionality
- Runs comprehensive API endpoint testing on PRs
- Pushes to Harbor registry with proper tagging on main branch
- Includes security attestation and SBOM generation

### Required Secrets for Production Deployment

Set these in your GitHub repository secrets:
- `HARBOR_REGISTRY`: Your Harbor registry URL
- `HARBOR_USERNAME`: Registry username
- `HARBOR_PASSWORD`: Registry password

## Testing

### Docker Testing
Test the Docker deployment locally:
```bash
./test_docker.sh
```

### API Testing
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
