# Docker Build Troubleshooting Guide

## Common Issues and Solutions

### 1. Package Not Found Error (404)

**Error:**
```
E: Failed to fetch http://ports.ubuntu.com/ubuntu-ports/pool/main/...  404  Not Found
E: Unable to fetch some archives, maybe run apt-get update or try with --fix-missing?
```

**Solutions:**

#### Option A: Use the Robust Dockerfile
```bash
docker build -f Dockerfile.robust -t gateman-face-optimized .
```

#### Option B: Use the Docker Build Script
```bash
./docker_build.sh build
```

#### Option C: Manual Fix in Dockerfile
Add `--fix-missing` flag to apt-get install:
```dockerfile
RUN apt-get update && apt-get install -y --fix-missing \
    your-packages-here
```

### 2. OpenCV Package Version Issues

**Problem:** Different Ubuntu versions have different OpenCV package names.

**Solution:** The robust Dockerfile tries multiple package names:
```dockerfile
RUN apt-get install -y libopencv-dev || \
    apt-get install -y libopencv-core4.5d || \
    apt-get install -y libopencv-core-dev
```

### 3. Architecture-Specific Issues (ARM/ARM64)

**Problem:** Building on Apple Silicon (M1/M2) or ARM servers.

**Solutions:**

#### Use Platform Flag
```bash
docker build --platform linux/amd64 -f Dockerfile.robust -t gateman-face-optimized .
```

#### Use BuildKit
```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.robust -t gateman-face-optimized .
```

### 4. Build Failures Due to Network Issues

**Problem:** Model downloads fail during build.

**Solution:** Download models locally first:
```bash
# Download models before building
mkdir -p models
curl -L -o models/deploy.prototxt \
    https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
curl -L -o models/res10_300x300_ssd_iter_140000.caffemodel \
    https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
curl -L -o models/haarcascade_frontalface_default.xml \
    https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

# Then build
docker build -f Dockerfile.robust -t gateman-face-optimized .
```

### 5. Out of Memory During Build

**Problem:** Docker build fails with memory errors.

**Solutions:**

#### Increase Docker Memory
```bash
# Docker Desktop: Preferences → Resources → Memory
# Set to at least 4GB
```

#### Use Build Arguments to Limit Parallelism
```bash
docker build --build-arg MAKE_JOBS=2 -f Dockerfile.robust -t gateman-face-optimized .
```

### 6. CMake Cannot Find OpenCV

**Problem:** CMake fails to locate OpenCV during build.

**Solution:** Set PKG_CONFIG_PATH in Dockerfile:
```dockerfile
ENV PKG_CONFIG_PATH=/usr/lib/pkgconfig:/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
```

## Quick Fixes

### Clean Build (Remove Cache)
```bash
# Remove all cache and rebuild
docker system prune -a
docker build --no-cache -f Dockerfile.robust -t gateman-face-optimized .
```

### Use Pre-built Image (If Available)
```bash
# Pull from registry instead of building
docker pull gateman-face-optimized:latest
```

### Build with Logging
```bash
# Verbose build output
docker build --progress=plain -f Dockerfile.robust -t gateman-face-optimized . 2>&1 | tee build.log
```

## Platform-Specific Instructions

### Ubuntu 22.04
```bash
docker build -f Dockerfile.robust -t gateman-face-optimized .
```

### Ubuntu 20.04
```bash
# Use different base image
sed -i 's/FROM ubuntu:22.04/FROM ubuntu:20.04/g' Dockerfile.robust
docker build -f Dockerfile.robust -t gateman-face-optimized .
```

### Alpine Linux (Minimal)
```bash
# Note: Requires additional configuration for OpenCV
docker build -f Dockerfile.alpine -t gateman-face-optimized .
```

## Docker Compose Alternative

If Docker build continues to fail, use docker-compose:

```bash
# This handles dependencies better
docker-compose up --build
```

## Verification Steps

After successful build:

### 1. Check Image
```bash
docker images | grep gateman-face-optimized
```

### 2. Test Container
```bash
# Run test
docker run --rm gateman-face-optimized ./test_directions
```

### 3. Health Check
```bash
# Start service
docker run -d -p 8080:8080 --name gateman-test gateman-face-optimized

# Check health
curl http://localhost:8080/health

# Stop test
docker stop gateman-test && docker rm gateman-test
```

## Emergency Fallback

If all Docker builds fail, use the local build:

```bash
# Build locally without Docker
./build_simple.sh

# Run locally
./build_simple/optimized_server --port 8080
```

## Useful Docker Commands

### Debug Build Issues
```bash
# Interactive debug
docker run -it ubuntu:22.04 bash
# Then manually run apt-get commands to test
```

### Check Build Logs
```bash
docker build -f Dockerfile.robust -t gateman-face-optimized . 2>&1 | grep -i error
```

### Multi-Stage Build Debug
```bash
# Build only first stage
docker build --target builder -f Dockerfile.robust -t gateman-builder .
```

### Resource Usage
```bash
docker stats
```

## Contact Support

If issues persist after trying these solutions:

1. Check system requirements:
   - Docker version: `docker --version` (should be 20.10+)
   - Available RAM: `free -h` (need at least 2GB)
   - Available disk: `df -h` (need at least 5GB)

2. Collect diagnostic info:
   ```bash
   docker version > diagnostic.txt
   docker info >> diagnostic.txt
   uname -a >> diagnostic.txt
   ```

3. Try the robust Dockerfile which has multiple fallbacks:
   ```bash
   docker build -f Dockerfile.robust -t gateman-face-optimized .
   ```

## Quick Start Script

Save this as `docker_quickstart.sh`:

```bash
#!/bin/bash
echo "Attempting Docker build with fallbacks..."

# Try robust build first
if docker build -f Dockerfile.robust -t gateman-face-optimized .; then
    echo "Build successful!"
    docker run -d -p 8080:8080 gateman-face-optimized
    echo "Service running on http://localhost:8080"
elif docker build -f Dockerfile.simple -t gateman-face-optimized .; then
    echo "Simple build successful!"
    docker run -d -p 8080:8080 gateman-face-optimized
    echo "Service running on http://localhost:8080"
else
    echo "Docker build failed. Falling back to local build..."
    ./build_simple.sh
    ./build_simple/optimized_server --port 8080
fi
```

Make it executable: `chmod +x docker_quickstart.sh`
Run it: `./docker_quickstart.sh`
