# ============================================
# Lightweight Face Service - Dockerfile
# OpenCV-only implementation (no heavy ML models)
# ============================================

# Build stage - Lightweight dependencies only
FROM ubuntu:22.04 AS builder

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_BUILD_TYPE=Release \
    MAKEFLAGS="-j$(nproc)"

# Set working directory
WORKDIR /app

# Install only essential build dependencies (no heavy ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    wget \
    libssl-dev \
    libcurl4-openssl-dev \
    # OpenCV core libraries only (no contrib/extra modules)
    libopencv-dev \
    libopencv-core-dev \
    libopencv-imgproc-dev \
    libopencv-imgcodecs-dev \
    libopencv-video-dev \
    libopencv-videoio-dev \
    # Threading support
    libtbb-dev \
    # Clean up apt cache
    && rm -rf /var/lib/apt/lists/*

# Copy source code and build configuration
COPY CMakeLists.txt ./
COPY src/lightweight/ ./src/lightweight/
COPY *.cpp ./

# Build the lightweight application  
RUN mkdir -p build_lightweight && cd build_lightweight && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# ============================================
# Runtime stage - Minimal runtime dependencies
# ============================================

FROM ubuntu:22.04 AS runtime

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    # OpenCV runtime libraries
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libopencv-imgcodecs4.5d \
    libopencv-video4.5d \
    libopencv-videoio4.5d \
    # CURL for video downloads
    libcurl4 \
    # Threading
    libtbb12 \
    # Utilities
    curl \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd -r -s /bin/false appuser

# Set working directory
WORKDIR /app

# Copy the built binaries from builder stage
COPY --from=builder /app/build_lightweight/test_lightweight ./

# Make binaries executable
RUN chmod +x ./test_lightweight

# Create logs directory
RUN mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port (for future web service integration)
EXPOSE 8080

# Health check using the test
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
    CMD timeout 25s ./test_lightweight > /dev/null 2>&1 || exit 1

# Default command - run the test to demonstrate functionality
CMD ["./test_lightweight"]