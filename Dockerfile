# ============================================
# Full Face Service - Production Dockerfile
# Complete implementation with ML models and Redis
# ============================================

# Build stage - Full dependencies
FROM ubuntu:22.04 AS builder

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_BUILD_TYPE=Release \
    MAKEFLAGS="-j$(nproc)"

# Set working directory
WORKDIR /app

# Install comprehensive build dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    wget \
    # OpenCV with all features
    libopencv-dev \
    libopencv-core-dev \
    libopencv-imgproc-dev \
    libopencv-imgcodecs-dev \
    libopencv-video-dev \
    libopencv-videoio-dev \
    libopencv-highgui-dev \
    libopencv-objdetect-dev \
    libopencv-ml-dev \
    # Threading and system libraries
    libtbb-dev \
    libboost-system-dev \
    libboost-thread-dev \
    # SSL and crypto
    libssl-dev \
    libcurl4-openssl-dev \
    # JSON library
    nlohmann-json3-dev \
    # Redis client
    libhiredis-dev \
    # BLAS and LAPACK for dlib
    libblas-dev \
    liblapack-dev \
    # PNG and JPEG for image processing
    libpng-dev \
    libjpeg-dev \
    # X11 for GUI (optional, for dlib)
    libx11-dev \
    # ASIO for networking
    libasio-dev \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Install dlib from source (required for face recognition)
RUN git clone --depth 1 --branch v19.24 https://github.com/davisking/dlib.git && \
    cd dlib && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && rm -rf dlib

# Copy source code
COPY CMakeLists.txt ./
COPY crow/ ./crow/
COPY src/ ./src/
COPY models/ ./models/

# Build the full service
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make MLFaceService -j$(nproc)

# ============================================
# Runtime stage - Minimal runtime dependencies
# ============================================

FROM ubuntu:22.04 AS runtime

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    # OpenCV runtime
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libopencv-imgcodecs4.5d \
    libopencv-video4.5d \
    libopencv-videoio4.5d \
    libopencv-highgui4.5d \
    libopencv-objdetect4.5d \
    libopencv-ml4.5d \
    # Threading
    libtbb12 \
    # Boost runtime
    libboost-system1.74.0 \
    libboost-thread1.74.0 \
    # SSL and crypto
    libssl3 \
    libcurl4 \
    # Redis client
    libhiredis0.14 \
    # BLAS and LAPACK
    libblas3 \
    liblapack3 \
    # Image libraries
    libpng16-16 \
    libjpeg8 \
    # X11 runtime (minimal)
    libx11-6 \
    # Utilities
    curl \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd -r -s /bin/false appuser && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy the built binary and models
COPY --from=builder /app/build/MLFaceService ./
COPY --from=builder /app/models/ ./models/

# Make binary executable and set ownership
RUN chmod +x ./MLFaceService && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["./MLFaceService", "--port", "8080", "--models", "./models"]