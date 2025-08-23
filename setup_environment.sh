#!/bin/bash

# ML Face Service - Environment Setup Script
# This script installs all required dependencies for building and running the service

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== ML Face Service - Environment Setup ==="
echo "Project directory: $SCRIPT_DIR"
echo "=========================================="

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
    echo "Installing dependencies for Ubuntu/Debian..."
    
    # Update package lists
    sudo apt-get update
    
    # Install build essentials
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        pkg-config \
        curl \
        wget \
        bzip2 \
        unzip
    
    # Install OpenCV dependencies
    sudo apt-get install -y \
        libopencv-dev \
        libopencv-contrib-dev \
        python3-opencv
    
    # Install dlib dependencies
    sudo apt-get install -y \
        libdlib-dev \
        libdlib19
    
    # Install networking libraries
    sudo apt-get install -y \
        libcurl4-openssl-dev \
        libssl-dev
    
    # Install JSON library
    sudo apt-get install -y \
        nlohmann-json3-dev
    
    # Install threading libraries
    sudo apt-get install -y \
        libtbb-dev
    
    echo "✓ Ubuntu/Debian dependencies installed successfully!"
}

# Function to install dependencies on CentOS/RHEL
install_centos_deps() {
    echo "Installing dependencies for CentOS/RHEL..."
    
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
        wget \
        bzip2 \
        unzip
    
    # Install OpenCV (may need to build from source on older CentOS)
    sudo yum install -y \
        opencv-devel \
        opencv-contrib-devel
    
    # Install development libraries
    sudo yum install -y \
        libcurl-devel \
        openssl-devel
    
    echo "✓ CentOS/RHEL dependencies installed successfully!"
    echo "Note: You may need to build dlib and nlohmann/json from source."
}

# Function to install dependencies on Fedora
install_fedora_deps() {
    echo "Installing dependencies for Fedora..."
    
    # Update packages
    sudo dnf update -y
    
    # Install build essentials
    sudo dnf groupinstall -y "Development Tools" "Development Libraries"
    sudo dnf install -y \
        cmake \
        git \
        pkgconfig \
        curl \
        wget \
        bzip2 \
        unzip
    
    # Install OpenCV
    sudo dnf install -y \
        opencv-devel \
        opencv-contrib-devel
    
    # Install dlib
    sudo dnf install -y \
        dlib-devel
    
    # Install networking libraries
    sudo dnf install -y \
        libcurl-devel \
        openssl-devel
    
    # Install JSON library
    sudo dnf install -y \
        json-devel
    
    echo "✓ Fedora dependencies installed successfully!"
}

# Function to install dependencies on macOS
install_macos_deps() {
    echo "Installing dependencies for macOS..."
    
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
        wget \
        bzip2
    
    # Install OpenCV
    brew install opencv
    
    # Install dlib
    brew install dlib
    
    # Install networking libraries
    brew install curl openssl
    
    # Install JSON library
    brew install nlohmann-json
    
    # Install Crow HTTP framework
    brew install crow
    
    echo "✓ macOS dependencies installed successfully!"
}

# Function to build and install dependencies from source
build_from_source() {
    echo "Building dependencies from source..."
    
    local build_dir="$SCRIPT_DIR/build_deps"
    mkdir -p "$build_dir"
    cd "$build_dir"
    
    # Build nlohmann/json if not available
    if ! pkg-config --exists nlohmann_json; then
        echo "Building nlohmann/json from source..."
        git clone https://github.com/nlohmann/json.git
        cd json
        mkdir -p build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release -DJSON_BuildTests=OFF
        make -j$(nproc)
        sudo make install
        cd "$build_dir"
        echo "✓ nlohmann/json built and installed"
    fi
    
    # Build Crow if not available
    if [[ ! -f "/usr/local/include/crow.h" && ! -f "/usr/include/crow.h" ]]; then
        echo "Building Crow HTTP framework from source..."
        git clone https://github.com/CrowCpp/Crow.git
        cd Crow
        mkdir -p build && cd build
        cmake .. -DCROW_BUILD_EXAMPLES=OFF -DCROW_BUILD_TESTS=OFF
        make -j$(nproc)
        sudo make install
        cd "$build_dir"
        echo "✓ Crow built and installed"
    fi
    
    cd "$SCRIPT_DIR"
    rm -rf "$build_dir"
}

# Function to verify installation
verify_installation() {
    echo ""
    echo "Verifying installation..."
    
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
    
    # Check for dlib
    if ! pkg-config --exists dlib-1 && [[ ! -f "/usr/include/dlib/dlib_version.h" && ! -f "/usr/local/include/dlib/dlib_version.h" ]]; then
        missing_deps+=("dlib")
    fi
    
    # Check for libcurl
    if ! pkg-config --exists libcurl; then
        missing_deps+=("libcurl")
    fi
    
    if [[ ${#missing_deps[@]} -eq 0 ]]; then
        echo "✓ All essential dependencies are available!"
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
    echo "=========================================="
    echo "✓ Environment setup completed!"
    echo ""
    echo "Next steps:"
    echo "1. Download ML models:"
    echo "   ./models/download_models.sh"
    echo ""
    echo "2. Build the project:"
    echo "   mkdir build && cd build"
    echo "   cmake .."
    echo "   make -j\$(nproc)"
    echo ""
    echo "3. Or use the startup script:"
    echo "   ./startup_local.sh"
    echo ""
    echo "=========================================="
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
            echo "Note: Some dependencies may need to be built from source."
            build_from_source
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
            echo "  - OpenCV (>= 4.0)"
            echo "  - dlib"
            echo "  - libcurl"
            echo "  - nlohmann/json"
            echo "  - Crow HTTP framework"
            echo "  - Build tools (gcc/clang, make)"
            exit 1
            ;;
    esac
    
    # Try to build missing dependencies from source
    if [[ "$os_type" != "macos" ]]; then
        build_from_source
    fi
    
    # Verify the installation
    if verify_installation; then
        show_next_steps
    else
        echo ""
        echo "✗ Some dependencies are still missing."
        echo "You may need to install them manually or build from source."
        echo ""
        echo "For manual installation, refer to the documentation of each library:"
        echo "  - OpenCV: https://opencv.org/get-started/"
        echo "  - dlib: http://dlib.net/compile.html"
        echo "  - Crow: https://crowcpp.org/master/getting_started/setup/"
        echo "  - nlohmann/json: https://github.com/nlohmann/json"
        exit 1
    fi
}

# Handle command line arguments
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script installs all dependencies required for the ML Face Service."
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
    echo "Verifying existing installation..."
    if verify_installation; then
        echo "✓ All dependencies are properly installed!"
        show_next_steps
    else
        echo "✗ Some dependencies are missing. Run without --verify to install them."
        exit 1
    fi
    exit 0
fi

# Run main function
main "$@"
