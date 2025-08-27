#!/bin/bash

# Redis setup script for different platforms

echo "=== Setting up Redis for Challenge System ==="

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "Detected Linux system"
    
    # Ubuntu/Debian
    if command -v apt-get &> /dev/null; then
        echo "Installing Redis using apt..."
        sudo apt-get update
        sudo apt-get install -y redis-server libhiredis-dev
        
        # Start Redis service
        sudo systemctl start redis-server
        sudo systemctl enable redis-server
        
    # CentOS/RHEL/Fedora
    elif command -v yum &> /dev/null; then
        echo "Installing Redis using yum..."
        sudo yum install -y redis hiredis-devel
        
        # Start Redis service
        sudo systemctl start redis
        sudo systemctl enable redis
        
    # Arch Linux
    elif command -v pacman &> /dev/null; then
        echo "Installing Redis using pacman..."
        sudo pacman -S redis hiredis
        
        # Start Redis service
        sudo systemctl start redis
        sudo systemctl enable redis
    else
        echo "Unsupported Linux distribution. Please install Redis manually."
        exit 1
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Detected macOS system"
    
    if command -v brew &> /dev/null; then
        echo "Installing Redis using Homebrew..."
        brew install redis hiredis
        
        # Start Redis service
        brew services start redis
    else
        echo "Homebrew not found. Please install Homebrew first:"
        echo "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
else
    echo "Unsupported operating system: $OSTYPE"
    echo "Please install Redis and hiredis manually."
    exit 1
fi

# Test Redis connection
echo
echo "Testing Redis connection..."
if command -v redis-cli &> /dev/null; then
    if redis-cli ping | grep -q "PONG"; then
        echo "✅ Redis is running and accessible"
        
        # Set some basic configuration for security
        redis-cli CONFIG SET protected-mode yes
        redis-cli CONFIG SET port 6379
        
        echo "Redis setup completed successfully!"
        echo "Redis is now running on localhost:6379"
    else
        echo "❌ Redis is installed but not running"
        echo "Try starting Redis manually:"
        echo "  Linux: sudo systemctl start redis"
        echo "  macOS: brew services start redis"
    fi
else
    echo "❌ Redis CLI not found. Installation may have failed."
fi

echo
echo "To start the Redis server manually:"
echo "  redis-server"
echo
echo "To connect to Redis:"
echo "  redis-cli"
echo
echo "=== Redis Setup Complete ==="
