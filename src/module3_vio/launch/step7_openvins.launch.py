"""
PHASE 3 — Step 7: OpenVINS Launch File
=======================================

Launches:
  1. sensor_bridge  — remaps Gazebo topics to /imu0, /cam0/image_raw
  2. open_vins_msckf node  — the actual VIO estimator
  3. vio_comparator  — compares OpenVINS vs ground truth

PREREQUISITE — Install OpenVINS for ROS2 Humble:
-------------------------------------------------
  cd ~/ros2_ws/src
  git clone https://github.com/rpng/open_vins.git
  cd ~/ros2_ws
  rosdep install --from-paths src/open_vins --ignore-src -r -y
  colcon build --packages-select ov_core ov_init ov_msckf \\
      --cmake-args -DCMAKE_BUILD_TYPE=Release
  source install/setup.bash

Then run this launch file:
  ros2 launch module3_vio step7_openvins.launch.py

Or run nodes individually (easier for debugging):
  Terminal 3:  ros2 run module3_vio sensor_bridge
  Terminal 4:  ros2 run ov_msckf run_subscribe_msckf \\
                 --ros-args -p config_path:=<path_to_openvins_mono.yaml>
  Terminal 5:  ros2 run module3_vio vio_comparator
  Terminal 6:  ros2 run module2_visual_odometry circle_flight
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    pkg_dir = get_package_share_directory('module3_vio')
    config_path = os.path.join(pkg_dir, 'config', 'openvins_mono.yaml')

    return LaunchDescription([
        LogInfo(msg='=== PHASE 3 Step 7: OpenVINS VIO Study ==='),

        # ── 1. Sensor bridge ──────────────────────────────────────────────────
        Node(
            package='module3_vio',
            executable='sensor_bridge',
            name='sensor_bridge',
            output='screen',
        ),

        # ── 2. OpenVINS MSCKF node ─────────────────────────────────────────────
        # This node implements the full MSCKF VIO algorithm.
        # Study these source files after running:
        #   open_vins/ov_msckf/src/ros/RosSubscriber.cpp   — sensor callbacks
        #   open_vins/ov_msckf/src/VioManager.cpp          — EKF state machine
        #   open_vins/ov_core/src/track/TrackKLT.cpp       — feature tracker
        #   open_vins/ov_core/src/feat/FeatureInitializer.cpp — triangulation
        Node(
            package='ov_msckf',
            executable='run_subscribe_msckf',
            name='ov_msckf',
            output='screen',
            parameters=[{
                'config_path': config_path,
            }],
            remappings=[
                # Our sensor_bridge already publishes on these exact topics,
                # so no remapping needed.
                ('/imu0',           '/imu0'),
                ('/cam0/image_raw', '/cam0/image_raw'),
            ],
        ),

        # ── 3. VIO Comparator ─────────────────────────────────────────────────
        Node(
            package='module3_vio',
            executable='vio_comparator',
            name='vio_comparator',
            output='screen',
        ),
    ])
