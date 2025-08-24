#!/usr/bin/env python3
"""
Test script for video liveness detection API
"""

import json
import requests
import sys
import time

def test_video_liveness_api(server_url, video_url):
    """Test the video liveness detection endpoint"""
    
    endpoint = f"{server_url}/liveness/video"
    
    # Prepare request payload
    payload = {
        "video_url": video_url
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"Testing video liveness API...")
    print(f"Server: {server_url}")
    print(f"Video URL: {video_url}")
    print(f"Endpoint: {endpoint}")
    print("-" * 50)
    
    try:
        # Make the API request
        start_time = time.time()
        response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
        end_time = time.time()
        
        print(f"HTTP Status: {response.status_code}")
        print(f"Response Time: {end_time - start_time:.2f} seconds")
        print("-" * 50)
        
        if response.status_code == 200:
            # Parse response
            result = response.json()
            
            if result.get("success", False):
                data = result
                print("✅ Video Liveness Analysis Results:")
                print(f"   Is Live: {data.get('is_live', False)}")
                print(f"   Confidence: {data.get('confidence', 0):.3f}")
                print(f"   Yaw Range: {data.get('yaw_range', 0):.2f}°")
                print(f"   Pitch Range: {data.get('pitch_range', 0):.2f}°")
                print(f"   Frame Count: {data.get('frame_count', 0)}")
                print(f"   Duration: {data.get('duration_seconds', 0):.2f} seconds")
                print(f"   Sufficient Movement: {data.get('has_sufficient_movement', False)}")
                print(f"   Processing Time: {data.get('processing_time_ms', 0)} ms")
                
                # Display movement counts
                movement_counts = data.get('movement_counts', {})
                if movement_counts:
                    print(f"\n📊 Directional Movement Counts:")
                    print(f"   Left movements: {movement_counts.get('left', 0)}")
                    print(f"   Right movements: {movement_counts.get('right', 0)}")
                    print(f"   Up movements: {movement_counts.get('up', 0)}")
                    print(f"   Down movements: {movement_counts.get('down', 0)}")
                    print(f"   Total directional movements: {movement_counts.get('total_directional', 0)}")
                
                # Display failure reason if any
                failure_reason = data.get('failure_reason', '')
                if failure_reason:
                    print(f"   Failure Reason: {failure_reason}")
                
                # Display directional movements
                directional_movements = data.get('directional_movements', [])
                if directional_movements:
                    print(f"\n🎯 Detected Directional Movements ({len(directional_movements)} total):")
                    for i, movement in enumerate(directional_movements):
                        direction = movement['direction'].upper()
                        magnitude = movement['magnitude']
                        duration = movement['duration']
                        start_time = movement['start_time']
                        similarity = movement.get('similarity_to_reference', 0) * 100
                        print(f"   {i+1}. {direction} movement: {magnitude:.1f}° over {duration:.2f}s (at {start_time:.2f}s, {similarity:.0f}% similarity)")
                else:
                    print(f"\n🎯 No significant directional movements detected")
                
                # Display some pose movements (reduced output)
                pose_movements = data.get('pose_movements', [])
                if pose_movements:
                    print(f"\n📊 Sample Head Pose Data (showing first 3):")
                    for i, movement in enumerate(pose_movements[:3]):
                        print(f"   {i+1}. Time: {movement['timestamp']:.2f}s, "
                              f"Yaw: {movement['yaw']:.1f}°, "
                              f"Pitch: {movement['pitch']:.1f}°")
                    
                    if len(pose_movements) > 3:
                        print(f"   ... and {len(pose_movements) - 3} more pose samples")
                
                return True
            else:
                print("❌ API returned error:")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP Error {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (60 seconds)")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - is the server running?")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_health_check(server_url):
    """Test the health check endpoint"""
    
    endpoint = f"{server_url}/health"
    
    try:
        response = requests.get(endpoint, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("🏥 Health Check Results:")
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Models Loaded: {result.get('models_loaded', False)}")
            print(f"   Video Liveness Available: {result.get('video_liveness_available', False)}")
            print(f"   Version: {result.get('version', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def main():
    # Default server URL
    server_url = "http://localhost:8080"
    
    # Test video URL (the provided Cloudinary video)
    video_url = "https://res.cloudinary.com/themizehq/video/upload/v1755671605/IMG_6487.mov"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    if len(sys.argv) > 2:
        video_url = sys.argv[2]
    
    print("🧪 Video Liveness Detection API Test")
    print("=" * 50)
    
    # Test health check first
    print("1. Testing health check...")
    if not test_health_check(server_url):
        print("\n❌ Health check failed. Make sure the server is running.")
        sys.exit(1)
    
    print("\n2. Testing video liveness detection...")
    success = test_video_liveness_api(server_url, video_url)
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("❌ Some tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
