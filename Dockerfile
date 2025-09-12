# ============================================
# Lightweight Face Service - Dockerfile
# OpenCV-only web server implementation
# ============================================

# Build stage - Lightweight dependencies only
FROM ubuntu:22.04 AS builder

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_BUILD_TYPE=Release \
    MAKEFLAGS="-j$(nproc)"

# Set working directory
WORKDIR /app

# Install build dependencies (including web server dependencies)
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
    # Boost for Crow web framework
    libboost-system-dev \
    libboost-thread-dev \
    # ASIO networking library for Crow
    libasio-dev \
    # nlohmann/json dependency
    nlohmann-json3-dev \
    # Clean up apt cache
    && rm -rf /var/lib/apt/lists/*

# Copy source code and build configuration
COPY CMakeLists.txt ./
COPY crow/ ./crow/
COPY src/lightweight/ ./src/lightweight/
COPY *.cpp ./

# Build the lightweight web server
RUN mkdir -p build_lightweight && cd build_lightweight && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make lightweight_web_server -j$(nproc)

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
    # Boost runtime for Crow
    libboost-system1.74.0 \
    libboost-thread1.74.0 \
    # Utilities
    curl \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd -r -s /bin/false appuser

# Set working directory
WORKDIR /app

# Copy the built web server binary from builder stage
COPY --from=builder /app/build_lightweight/lightweight_web_server ./

# Make binary executable
RUN chmod +x ./lightweight_web_server

# Create logs directory
RUN mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for web server
EXPOSE 8080

# Health check using curl to test the /health endpoint
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command - run the web server
CMD ["./lightweight_web_server", "--port", "8080"]