# ML Face Service - Multi-stage Docker Build
# Stage 1: Build environment with all dependencies
# Stage 2: Runtime environment with compiled application

# ================================
# Stage 1: Builder
# ================================
FROM ubuntu:22.04 as builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    git \
    pkg-config \
    curl \
    wget \
    bzip2 \
    unzip \
    # OpenCV dependencies
    libopencv-dev \
    libopencv-contrib-dev \
    # Networking libraries
    libcurl4-openssl-dev \
    libssl-dev \
    # Redis client
    libhiredis-dev \
    # Threading libraries
    libtbb-dev \
    # Additional dependencies
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install dlib from source for better optimization
WORKDIR /tmp/dlib
RUN git clone https://github.com/davisking/dlib.git . && \
    git checkout v19.24 && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DDLIB_USE_BLAS=ON \
        -DDLIB_USE_LAPACK=ON \
        -DDLIB_PNG_SUPPORT=ON \
        -DDLIB_JPEG_SUPPORT=ON \
        -DDLIB_GIF_SUPPORT=ON && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf /tmp/dlib

# Install nlohmann/json from source
WORKDIR /tmp/json
RUN git clone https://github.com/nlohmann/json.git . && \
    git checkout v3.11.2 && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DJSON_BuildTests=OFF \
        -DJSON_Install=ON && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf /tmp/json

# Install Crow HTTP framework from source
WORKDIR /tmp/crow
RUN git clone https://github.com/CrowCpp/Crow.git . && \
    git checkout v1.0+5 && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCROW_BUILD_EXAMPLES=OFF \
        -DCROW_BUILD_TESTS=OFF && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf /tmp/crow

# Update library cache
RUN ldconfig

# Set working directory for application build
WORKDIR /app/build

# Copy source code
COPY src/ /app/src/
COPY CMakeLists.txt /app/

# Build the application
RUN cd /app && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    # Verify binary was created
    ls -la MLFaceService && \
    # Test that the binary can run (will fail without models, but should show usage)
    timeout 5s ./MLFaceService --help || true

# ================================
# Stage 2: Runtime
# ================================
FROM ubuntu:22.04 as runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    # OpenCV runtime
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libopencv-imgcodecs4.5d \
    libopencv-objdetect4.5d \
    # Networking
    libcurl4 \
    libssl3 \
    # Redis client runtime
    libhiredis0.14 \
    # System libraries
    libtbb12 \
    # Utilities for model download
    curl \
    wget \
    bzip2 \
    ca-certificates \
    # Process utilities
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /bin/bash appuser

# Create application directory
WORKDIR /app

# Copy built application from builder stage
COPY --from=builder /app/build/MLFaceService /app/
COPY --from=builder /usr/local/lib/libdlib.so* /usr/local/lib/
COPY --from=builder /usr/local/include/dlib/ /usr/local/include/dlib/

# Copy scripts and configuration
COPY download_models.sh /app/
COPY models/.gitkeep /app/models/

# Make scripts executable
RUN chmod +x /app/MLFaceService && \
    chmod +x /app/download_models.sh

# Update library cache
RUN ldconfig

# Create models directory and set permissions
RUN mkdir -p /app/models && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Download models during build (cached layer)
RUN cd /app && \
    ./download_models.sh && \
    # Verify models were downloaded
    ls -la models/*.dat

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Set environment variables
ENV SERVER_PORT=8080
ENV MODEL_PATH=/app/models

# Default command
CMD ["./MLFaceService", "--port", "8080", "--models", "/app/models"]

# ================================
# Build metadata
# ================================
LABEL maintainer="ML Face Service"
LABEL version="1.0.0"
LABEL description="High-performance C++ face recognition and liveness detection service"
LABEL org.opencontainers.image.source="https://github.com/your-repo/ml-face-service"
