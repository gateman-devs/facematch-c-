# ============================================
# ML Face Service - Multi-stage Dockerfile
# ============================================

# Build stage - Install dependencies and build the application
FROM ubuntu:22.04 AS builder

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_BUILD_TYPE=Release

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    curl \
    wget \
    bzip2 \
    unzip \
    libssl-dev \
    libcurl4-openssl-dev \
    libhiredis-dev \
    redis-server \
    libtbb-dev \
    libopencv-dev \
    libopencv-contrib-dev \
    python3-opencv \
    libdlib-dev \
    libdlib19 \
    nlohmann-json3-dev \
    libasio-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Boost libraries (required by Crow/ASIO)
RUN apt-get update && apt-get install -y \
    libboost-system-dev \
    libboost-thread-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Crow HTTP framework from source
RUN git clone https://github.com/CrowCpp/Crow.git /tmp/crow && \
    cd /tmp/crow && \
    mkdir -p build && cd build && \
    cmake .. -DCROW_BUILD_EXAMPLES=OFF -DCROW_BUILD_TESTS=OFF && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/crow

# Set working directory
WORKDIR /app

# Copy source code
COPY CMakeLists.txt ./
COPY src/ ./src/

# Copy model download script
COPY download_models.sh ./

# Create models directory and download ML models
RUN mkdir -p models && \
    chmod +x download_models.sh && \
    ./download_models.sh

# Build the application
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# ============================================
# Runtime stage - Minimal image for running the service
# ============================================

FROM ubuntu:22.04 AS runtime

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libcurl4 \
    curl \
    libssl3 \
    libhiredis0.14 \
    libtbb2 \
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libopencv-imgcodecs4.5d \
    libopencv-highgui4.5d \
    libopencv-objdetect4.5d \
    libopencv-calib3d4.5d \
    libdlib19 \
    redis-server \
    libboost-system1.74.0 \
    libboost-thread1.74.0 \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd -r -s /bin/false appuser

# Set working directory
WORKDIR /app

# Copy the built binary from builder stage
COPY --from=builder /app/build/MLFaceService ./

# Copy models from builder stage
COPY --from=builder /app/models ./models/

# Make the binary executable
RUN chmod +x ./MLFaceService

# Create logs directory
RUN mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["./MLFaceService", "--port", "8080", "--models", "./models"]
