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

# Install missing build tools (cmake, build-essential)
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install curl for model downloads (ensure it's available)
RUN apt-get update && apt-get install -y curl bzip2 && rm -rf /var/lib/apt/lists/*

# Download models during build
RUN chmod +x download_models.sh && \
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
    libblas3 \
    liblapack3 \
    libatlas3-base \
    libsqlite3-0 \
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
