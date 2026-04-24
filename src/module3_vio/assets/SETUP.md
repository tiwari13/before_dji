# Assets — deploy these files after cloning

## 1. OakD-Lite camera model (640x480 @ 10Hz for VIO)
Copy to PX4-Autopilot models directory:
```bash
cp assets/OakD-Lite_model.sdf \
   ~/PX4-Autopilot/Tools/simulation/gz/models/OakD-Lite/model.sdf
```
Reduces all 6 drone cameras from 1920x1080@30Hz → 640x480@10Hz so Gazebo
runs at real-time factor ~1.0 instead of ~0.13.

## 2. ROS2-Gazebo bridge config (includes IMU at 250Hz)
Copy to ros2_ws root:
```bash
cp assets/camera_bridge.yaml ~/ros2_ws/camera_bridge.yaml
```
Start the bridge with:
```bash
ros2 run ros_gz_bridge parameter_bridge \
  --ros-args -p config_file:=$HOME/ros2_ws/camera_bridge.yaml
```

## 3. PX4 parameter — set once in PX4 console (fixes preflight arming block)
In the terminal running PX4 SITL, type:
```
param set COM_RCL_EXCEPT 4
param set NAV_RCL_ACT 0
param save
```
This allows arming in offboard mode without RC/GCS connection.
Only needed once — PX4 saves it to parameters.bson.
