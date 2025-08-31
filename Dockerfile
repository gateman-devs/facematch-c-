# ============================================
# ML Face Service - Optimized Multi-stage Dockerfile
# ============================================

# Build stage - Use pre-built base image with all dependencies
FROM ghcr.io/emekarr/gateman-face-base-image:latest AS builder

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_BUILD_TYPE=Release \
    MAKEFLAGS="-j$(nproc)"

# Set working directory
WORKDIR /app

# Copy source code and build scripts in separate layers for better caching
COPY CMakeLists.txt ./
COPY src/ ./src/
COPY download_models.sh ./

# Install missing build tools and development headers that are not in the base image
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    pkg-config \
    libssl-dev \
    libcurl4-openssl-dev \
    libhiredis-dev \
    nlohmann-json3-dev \
    libasio-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libsqlite3-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libtbb-dev \
    libopencv-dev \
    libopencv-contrib-dev \
    libdlib-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy models from source (already downloaded)
COPY models/ ./models/

# Build the application with single-threaded compilation to avoid memory exhaustion
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j1

# ============================================
# Runtime stage - Use base image with all dependencies
# ============================================

FROM ghcr.io/emekarr/gateman-face-base-image:latest AS runtime

# Install missing runtime dependencies (redis-server and FFmpeg libs are not in base image)
RUN apt-get update && apt-get install -y \
    redis-server \
    libavcodec58 \
    libavformat58 \
    libavutil56 \
    libswscale5 \
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
