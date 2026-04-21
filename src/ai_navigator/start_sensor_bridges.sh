#!/bin/bash
# AI Navigator Sensor Bridges Launcher
# This script bridges Gazebo sensor topics to ROS2
# Run this AFTER starting PX4 SITL simulation

echo "🌉 Starting AI Navigator Sensor Bridges..."
echo "================================================"

# Kill any existing bridge processes (except the RGB camera which PX4 starts)
killall -q parameter_bridge 2>/dev/null
sleep 1

# Start RGB camera bridge (from x500_depth model)
echo "✅ Starting RGB Camera bridge..."
ros2 run ros_gz_bridge parameter_bridge \
  /world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image@sensor_msgs/msg/Image[gz.msgs.Image \
  --ros-args -r /world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image:=/camera/image &
RGB_BRIDGE_PID=$!
sleep 1

# Start Depth camera bridge (if it exists)
echo "✅ Starting Depth Camera bridge..."
ros2 run ros_gz_bridge parameter_bridge \
  /world/default/model/x500_depth_0/link/camera_link/sensor/StereoOV7251/depth_image@sensor_msgs/msg/Image[gz.msgs.Image \
  --ros-args -r /world/default/model/x500_depth_0/link/camera_link/sensor/StereoOV7251/depth_image:=/depth_camera &
DEPTH_BRIDGE_PID=$!
sleep 1

# Start LIDAR bridge
echo "✅ Starting LIDAR bridge..."
ros2 run ros_gz_bridge parameter_bridge \
  /world/default/model/x500_0/link/link/sensor/lidar_2d_v2/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan \
  --ros-args -r /world/default/model/x500_0/link/link/sensor/lidar_2d_v2/scan:=/lidar &
LIDAR_BRIDGE_PID=$!
sleep 1

echo ""
echo "================================================"
echo "✅ All sensor bridges started!"
echo ""
echo "Active bridges:"
echo "  - RGB Camera:   /camera/image"
echo "  - Depth Camera: /depth_camera"
echo "  - LIDAR:        /lidar"
echo ""
echo "Verify with: ros2 topic list | grep -E 'camera|depth|lidar'"
echo ""
echo "Press Ctrl+C to stop all bridges"
echo "================================================"

# Wait for all background processes
wait
