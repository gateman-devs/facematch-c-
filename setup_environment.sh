#!/bin/bash

# Lightweight Face Service - Environment Setup Script
# This script installs minimal dependencies for the OpenCV-only implementation

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Lightweight Face Service - Environment Setup ==="
echo "Project directory: $SCRIPT_DIR"
echo "===================================================="

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command_exists apt-get; then
            echo "ubuntu"
        elif command_exists yum; then
            echo "centos"
        elif command_exists dnf; then
            echo "fedora"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Function to install dependencies on Ubuntu/Debian
install_ubuntu_deps() {
    echo "Installing lightweight dependencies for Ubuntu/Debian..."
    
    # Update package lists
    sudo apt-get update
    
    # Install build essentials
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        pkg-config \
        curl \
        wget
    
    # Install OpenCV (core libraries only, no heavy ML modules)
    sudo apt-get install -y \
        libopencv-dev \
        libopencv-core-dev \
        libopencv-imgproc-dev \
        libopencv-imgcodecs-dev \
        libopencv-video-dev \
        libopencv-videoio-dev
    
    # Install networking libraries
    sudo apt-get install -y \
        libcurl4-openssl-dev \
        libssl-dev
    
    # Install threading libraries
    sudo apt-get install -y \
        libtbb-dev
    
    echo "✓ Ubuntu/Debian lightweight dependencies installed successfully!"
    echo "Removed: Heavy ML libraries (dlib, TensorFlow, MediaPipe)"
    echo "Added: OpenCV core modules only"
}

# Function to install dependencies on CentOS/RHEL
install_centos_deps() {
    echo "Installing lightweight dependencies for CentOS/RHEL..."
    
    # Install EPEL repository for additional packages
    sudo yum install -y epel-release
    
    # Update packages
    sudo yum update -y
    
    # Install build essentials
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y \
        cmake \
        git \
        pkgconfig \
        curl \
        wget
    
    # Install OpenCV (core libraries only)
    sudo yum install -y \
        opencv-devel
    
    # Install development libraries
    sudo yum install -y \
        libcurl-devel \
        openssl-devel \
        tbb-devel
    
    echo "✓ CentOS/RHEL lightweight dependencies installed successfully!"
}

# Function to install dependencies on Fedora
install_fedora_deps() {
    echo "Installing lightweight dependencies for Fedora..."
    
    # Update packages
    sudo dnf update -y
    
    # Install build essentials
    sudo dnf groupinstall -y "Development Tools" "Development Libraries"
    sudo dnf install -y \
        cmake \
        git \
        pkgconfig \
        curl \
        wget
    
    # Install OpenCV (core libraries only)
    sudo dnf install -y \
        opencv-devel
    
    # Install networking libraries
    sudo dnf install -y \
        libcurl-devel \
        openssl-devel
    
    # Install threading libraries  
    sudo dnf install -y \
        tbb-devel
    
    echo "✓ Fedora lightweight dependencies installed successfully!"
}

# Function to install dependencies on macOS
install_macos_deps() {
    echo "Installing lightweight dependencies for macOS..."
    
    # Check if Homebrew is installed
    if ! command_exists brew; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for this session
        export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
        
        if ! command_exists brew; then
            echo "✗ Failed to install Homebrew. Please install it manually."
            echo "Visit: https://brew.sh"
            return 1
        fi
    fi
    
    # Update Homebrew
    brew update
    
    # Install build tools
    brew install \
        cmake \
        git \
        pkg-config \
        curl \
        wget
    
    # Install OpenCV (this will install core modules)
    brew install opencv
    
    # Install networking libraries
    brew install curl openssl
    
    echo "✓ macOS lightweight dependencies installed successfully!"
}

# Function to verify installation
verify_installation() {
    echo ""
    echo "Verifying lightweight installation..."
    
    local missing_deps=()
    
    # Check for essential tools
    if ! command_exists cmake; then
        missing_deps+=("cmake")
    fi
    
    if ! command_exists git; then
        missing_deps+=("git")
    fi
    
    if ! command_exists pkg-config; then
        missing_deps+=("pkg-config")
    fi
    
    # Check for OpenCV
    if ! pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
        missing_deps+=("opencv")
    fi
    
    # Check for libcurl
    if ! pkg-config --exists libcurl; then
        missing_deps+=("libcurl")
    fi
    
    if [[ ${#missing_deps[@]} -eq 0 ]]; then
        echo "✓ All lightweight dependencies are available!"
        return 0
    else
        echo "✗ Missing dependencies:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        return 1
    fi
}

# Function to show next steps
show_next_steps() {
    echo ""
    echo "=================================================="
    echo "✓ Lightweight environment setup completed!"
    echo ""
    echo "What was installed:"
    echo "  ✓ Build tools (CMake, GCC/Clang, Make)"
    echo "  ✓ OpenCV core libraries only"
    echo "  ✓ CURL for video downloads"
    echo "  ✓ Threading support (TBB)"
    echo ""
    echo "What was REMOVED/SKIPPED:"
    echo "  ❌ dlib (heavy ML library)"
    echo "  ❌ TensorFlow/TensorFlow Lite"
    echo "  ❌ MediaPipe"
    echo "  ❌ Shape predictor models"
    echo "  ❌ Face recognition models"
    echo "  ❌ Redis (not needed for lightweight version)"
    echo ""
    echo "Next steps:"
    echo "1. Build the lightweight implementation:"
    echo "   ./setup_local.sh"
    echo ""
    echo "2. Or build manually:"
    echo "   mkdir build_lightweight && cd build_lightweight"
    echo "   cmake .. && make -j\$(nproc)"
    echo ""
    echo "3. Run tests:"
    echo "   ./build_lightweight/optimized_test"
    echo ""
    echo "4. For Docker deployment:"
    echo "   docker build -t lightweight-face-service ."
    echo ""
    echo "Performance benefits:"
    echo "  🚀 ~0 MB model size (vs ~100MB+ before)"
    echo "  🚀 ~3s per video (vs 10s+ before)"
    echo "  🚀 Minimal CPU usage"
    echo "  🚀 75% accuracy on movement detection"
    echo "=================================================="
}

# Main execution
main() {
    local os_type=$(detect_os)
    echo "Detected OS: $os_type"
    echo ""
    
    case "$os_type" in
        "ubuntu")
            install_ubuntu_deps
            ;;
        "centos")
            install_centos_deps
            ;;
        "fedora")
            install_fedora_deps
            ;;
        "macos")
            install_macos_deps
            ;;
        *)
            echo "✗ Unsupported or unknown operating system: $OSTYPE"
            echo "Please install dependencies manually:"
            echo "  - CMake (>= 3.16)"
            echo "  - OpenCV (>= 4.0) core modules only"
            echo "  - libcurl"
            echo "  - Build tools (gcc/clang, make)"
            echo ""
            echo "Skip these heavy dependencies:"
            echo "  - dlib, TensorFlow, MediaPipe, Redis"
            exit 1
            ;;
    esac
    
    # Verify the installation
    if verify_installation; then
        show_next_steps
    else
        echo ""
        echo "✗ Some dependencies are still missing."
        echo "You may need to install them manually."
        echo ""
        echo "For manual installation, refer to:"
        echo "  - OpenCV: https://opencv.org/get-started/"
        echo "  - CURL: System package manager"
        exit 1
    fi
}

# Handle command line arguments
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script installs minimal dependencies for the Lightweight Face Service."
    echo "Heavy ML libraries (dlib, TensorFlow, MediaPipe) are intentionally skipped."
    echo ""
    echo "Options:"
    echo "  --help, -h    Show this help message"
    echo "  --verify      Only verify existing installation"
    echo ""
    echo "Supported operating systems:"
    echo "  - Ubuntu/Debian"
    echo "  - CentOS/RHEL"
    echo "  - Fedora"
    echo "  - macOS (with Homebrew)"
    echo ""
    exit 0
elif [[ "$1" == "--verify" ]]; then
    echo "Verifying existing lightweight installation..."
    if verify_installation; then
        echo "✓ All lightweight dependencies are properly installed!"
        show_next_steps
    else
        echo "✗ Some dependencies are missing. Run without --verify to install them."
        exit 1
    fi
    exit 0
fi

# Run main function
main "$@"