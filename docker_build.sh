#!/bin/bash

# Docker build and run script for Gateman Face Service
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="gateman-face-optimized"
CONTAINER_NAME="gateman-face-service"
PORT="${PORT:-8080}"
DOCKERFILE="Dockerfile.robust"
BUILD_CONTEXT="."

# Functions
print_header() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Parse command line arguments
ACTION="build"
DETACHED=false
FORCE_REBUILD=false
USE_COMPOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        build)
            ACTION="build"
            shift
            ;;
        run)
            ACTION="run"
            shift
            ;;
        stop)
            ACTION="stop"
            shift
            ;;
        restart)
            ACTION="restart"
            shift
            ;;
        logs)
            ACTION="logs"
            shift
            ;;
        test)
            ACTION="test"
            shift
            ;;
        clean)
            ACTION="clean"
            shift
            ;;
        -d|--detached)
            DETACHED=true
            shift
            ;;
        -f|--force)
            FORCE_REBUILD=true
            shift
            ;;
        -c|--compose)
            USE_COMPOSE=true
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --dockerfile)
            DOCKERFILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [ACTION] [OPTIONS]"
            echo ""
            echo "Actions:"
            echo "  build       Build Docker image"
            echo "  run         Run Docker container"
            echo "  stop        Stop running container"
            echo "  restart     Restart container"
            echo "  logs        View container logs"
            echo "  test        Run tests in container"
            echo "  clean       Clean up images and containers"
            echo ""
            echo "Options:"
            echo "  -d, --detached     Run container in background"
            echo "  -f, --force        Force rebuild image"
            echo "  -c, --compose      Use docker-compose"
            echo "  -p, --port PORT    Set port (default: 8080)"
            echo "  --dockerfile FILE  Specify Dockerfile (default: Dockerfile.robust)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check Docker installation
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Main logic
case $ACTION in
    build)
        print_header "Building Docker Image"

        # Check which Dockerfile to use
        if [ ! -f "$DOCKERFILE" ]; then
            print_warning "Dockerfile '$DOCKERFILE' not found, trying alternatives..."
            if [ -f "Dockerfile.simple" ]; then
                DOCKERFILE="Dockerfile.simple"
                print_info "Using Dockerfile.simple"
            elif [ -f "Dockerfile" ]; then
                DOCKERFILE="Dockerfile"
                print_info "Using Dockerfile"
            else
                print_error "No Dockerfile found!"
                exit 1
            fi
        fi

        # Build options
        BUILD_ARGS=""
        if [ "$FORCE_REBUILD" = true ]; then
            BUILD_ARGS="--no-cache"
            print_info "Force rebuild enabled"
        fi

        # Download models first if not present
        if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
            print_info "Downloading models..."
            mkdir -p models

            # Download with retry logic
            for i in 1 2 3; do
                curl -L -o models/deploy.prototxt \
                    https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt && \
                    break || sleep 2
            done

            for i in 1 2 3; do
                curl -L -o models/res10_300x300_ssd_iter_140000.caffemodel \
                    https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel && \
                    break || sleep 2
            done

            for i in 1 2 3; do
                curl -L -o models/haarcascade_frontalface_default.xml \
                    https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml && \
                    break || sleep 2
            done

            print_success "Models downloaded"
        else
            print_info "Models already present"
        fi

        # Build image
        print_info "Building with Dockerfile: $DOCKERFILE"
        if docker build $BUILD_ARGS -f "$DOCKERFILE" -t "$IMAGE_NAME:latest" "$BUILD_CONTEXT"; then
            print_success "Docker image built successfully: $IMAGE_NAME:latest"

            # Show image info
            echo -e "\n${MAGENTA}Image Information:${NC}"
            docker images "$IMAGE_NAME:latest"
        else
            print_error "Docker build failed"
            exit 1
        fi
        ;;

    run)
        print_header "Running Docker Container"

        # Check if image exists
        if ! docker image inspect "$IMAGE_NAME:latest" &>/dev/null; then
            print_warning "Image not found, building first..."
            $0 build
        fi

        # Stop existing container if running
        if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
            print_info "Stopping existing container..."
            docker stop "$CONTAINER_NAME"
            docker rm "$CONTAINER_NAME"
        fi

        # Remove stopped container with same name
        if docker ps -aq -f name="$CONTAINER_NAME" | grep -q .; then
            docker rm "$CONTAINER_NAME"
        fi

        # Run options
        RUN_OPTIONS="-p ${PORT}:8080"
        RUN_OPTIONS="$RUN_OPTIONS --name $CONTAINER_NAME"
        RUN_OPTIONS="$RUN_OPTIONS --restart unless-stopped"
        RUN_OPTIONS="$RUN_OPTIONS -v $(pwd)/models:/app/models:ro"

        if [ "$DETACHED" = true ]; then
            RUN_OPTIONS="$RUN_OPTIONS -d"
            print_info "Running in detached mode"
        else
            RUN_OPTIONS="$RUN_OPTIONS -it"
        fi

        # Run container
        print_info "Starting container on port $PORT..."
        if docker run $RUN_OPTIONS "$IMAGE_NAME:latest"; then
            if [ "$DETACHED" = true ]; then
                print_success "Container started successfully"
                echo -e "\n${MAGENTA}Container Information:${NC}"
                docker ps -f name="$CONTAINER_NAME"
                echo -e "\n${YELLOW}Access the service at:${NC} http://localhost:${PORT}"
                echo -e "${YELLOW}View logs with:${NC} $0 logs"
                echo -e "${YELLOW}Stop with:${NC} $0 stop"
            fi
        else
            print_error "Failed to start container"
            exit 1
        fi
        ;;

    stop)
        print_header "Stopping Docker Container"

        if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
            print_info "Stopping container..."
            docker stop "$CONTAINER_NAME"
            docker rm "$CONTAINER_NAME"
            print_success "Container stopped and removed"
        else
            print_warning "Container is not running"
        fi
        ;;

    restart)
        print_header "Restarting Docker Container"
        $0 stop
        sleep 2
        $0 run -d
        ;;

    logs)
        print_header "Container Logs"

        if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
            docker logs -f "$CONTAINER_NAME"
        else
            print_error "Container is not running"
            print_info "Showing last logs if available..."
            docker logs "$CONTAINER_NAME" 2>/dev/null || print_error "No logs available"
        fi
        ;;

    test)
        print_header "Running Tests"

        # Check if container is running
        if ! docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
            print_info "Starting container for testing..."
            $0 run -d
            sleep 5
        fi

        print_info "Running health check..."
        if curl -f "http://localhost:${PORT}/health" &>/dev/null; then
            print_success "Health check passed"
        else
            print_error "Health check failed"
            exit 1
        fi

        print_info "Testing direction detection..."
        echo -e "${YELLOW}Sending test request...${NC}"

        RESPONSE=$(curl -s -X POST "http://localhost:${PORT}/test-directions" \
            -H "Content-Type: application/json" 2>/dev/null || echo "")

        if [ -n "$RESPONSE" ]; then
            echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
            print_success "Test completed"
        else
            print_error "Test failed - no response"
            exit 1
        fi
        ;;

    clean)
        print_header "Cleaning Up"

        # Stop and remove container
        if docker ps -aq -f name="$CONTAINER_NAME" | grep -q .; then
            print_info "Removing container..."
            docker stop "$CONTAINER_NAME" 2>/dev/null || true
            docker rm "$CONTAINER_NAME" 2>/dev/null || true
        fi

        # Remove image
        if docker image inspect "$IMAGE_NAME:latest" &>/dev/null; then
            print_info "Removing image..."
            docker rmi "$IMAGE_NAME:latest"
        fi

        # Remove dangling images
        DANGLING=$(docker images -f "dangling=true" -q)
        if [ -n "$DANGLING" ]; then
            print_info "Removing dangling images..."
            docker rmi $DANGLING
        fi

        print_success "Cleanup completed"
        ;;

    *)
        print_error "Unknown action: $ACTION"
        exit 1
        ;;
esac

echo ""
print_success "Operation completed successfully!"
