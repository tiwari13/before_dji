---
name: VIO Project Current State
description: Current progress in the drone VIO project — Steps 7/8/9 done, next is Step 10 multi-camera MSCKF
type: project
---

## All completed (as of 2026-04-13)

### module3_vio → github.com/tiwari13/module3_vio  (branch: master)
- step7_sensor_bridge.py — remaps Gazebo topics to /imu0, /cam0/image_raw, /cam1/image_raw
  - Camera intrinsics scaled to 640x480: FX=FY=465.74, CX=320, CY=180
- step7_vio_comparator.py — ATE/RPE vs PX4 ground truth
- config/openvins_mono.yaml — full OpenVINS config, every parameter documented with math
- step8_ekf_vio.py — 15-state error-state EKF (p+v+phi+b_a+b_g)
- assets/OakD-Lite_model.sdf — 640x480 @ 10Hz camera (fixes Gazebo real-time factor)
- assets/camera_bridge.yaml — bridge config with IMU at 250 Hz
- assets/SETUP.md — deploy instructions

### module2_visual_odometry → github.com/tiwari13/module2_visual_odometry  (branch: main)
- monocular_vo.py — publishes PoseStamped on /monocular_vo/pose (feeds EKF)
- circle_flight.py — VehicleStatus feedback, real arm confirm, force-arm fallback (param2=21196)

## All bugs fixed

### Bug 1 — Drone never takes off (FIXED)
- Root cause: no VehicleStatus check; arming assumed from z-movement
- Fix: VehicleStatus subscriber, arming_state==2 confirms arm, force-arm after 3 attempts
- Also requires PX4 params: COM_RCL_EXCEPT=4, NAV_RCL_ACT=0 (set once in SITL console)

### Bug 2 — EKF drifts 555m in 185s (FIXED)
- Root cause: identity quaternion init, drone pitched 11.5 degrees in Gazebo
- Fix: _imu_cb reads msg.orientation on first sample to init self._state.q

### Bug 3 — IMU 35Hz, camera 4Hz (FIXED)
- Root cause: 1920x1080 cameras overwhelmed Gazebo; IMU was commented in bridge config
- Fix: OakD-Lite model at 640x480@10Hz + camera_bridge.yaml with IMU at 250Hz

## ONE-TIME setup on each machine

Deploy assets:
  mkdir -p ~/PX4-Autopilot/Tools/simulation/gz/models/OakD-Lite
  cp ~/ros2_ws/src/module3_vio/assets/OakD-Lite_model.sdf ~/PX4-Autopilot/Tools/simulation/gz/models/OakD-Lite/model.sdf
  cp ~/ros2_ws/src/module3_vio/assets/camera_bridge.yaml ~/ros2_ws/camera_bridge.yaml

PX4 SITL console (once, saved to disk):
  param set COM_RCL_EXCEPT 4
  param set NAV_RCL_ACT 0
  param save

## Full startup sequence
  T1: MicroXRCEAgent UDP4 --port 8888
  T2: Gazebo + PX4 SITL
  T3: ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=$HOME/ros2_ws/camera_bridge.yaml
  T4: ros2 run module2_visual_odometry spawn_landmarks
  T5: ros2 run module3_vio sensor_bridge
  T6: ros2 run module2_visual_odometry monocular_vo
  T7: ros2 run module3_vio ekf_vio
  T8: ros2 run module2_visual_odometry circle_flight

## Step 9 MSCKF — DONE (gravity bug fixed 2026-04-16)
File: module3_vio/step9_msckf.py — run as: ros2 run module3_vio msckf_vio
- IMU propagation + sliding window camera state augmentation
- KLT tracking, DLT triangulation, null-space projection (feature coords eliminated)
- Chi-squared Mahalanobis gate, batch EKF update, oldest-state marginalization
- CONFIRMED live rates: IMU=248Hz, cam0=7.7Hz

### Gravity bug fix (commit a4a62f2)
- OLD (WRONG): GRAVITY=[0,0,-9.81] with acc=R@ab-GRAVITY → adds +9.81, double-gravity
- NEW (FIXED): GRAVITY=[0,0,+9.81] with acc=R@ab-GRAVITY → correct: R@ab+[0,0,-9.81]
- Verified: at rest pitch=-11.5°, acc is now zero (was 19.62 m/s²!)
- Same fix applied to step10_multicam_msckf.py

### Remaining issues (NOT yet fixed — next session)
- |b_a|=0.000 always: accel bias never updates → pure IMU diverges when drone not flying
- Position drifts ~0.5m/s² at rest before flight (covariance grows fast without motion)
- cov_p explodes from 22→1900 when EKF updates become inconsistent
- Root cause: MSCKF needs actual 3D motion (parallax) to observe bias
- Diagnosis: drone was NOT flying during tests (circle_flight failed to arm)
  → need to test with drone actually flying circles to see real MSCKF performance

## Step 10 — Multi-Camera MSCKF — DONE (gravity fix applied same session)
File: module3_vio/step10_multicam_msckf.py — ros2 run module3_vio multicam_msckf
- Same gravity fix applied
- Not yet tested during actual flight

## Integration test — PASSED (2026-04-29)
Ran full stack:
  T1: ros2 run module3_vio sensor_bridge
  T2: ros2 run module3_vio msckf_vio
  T3: ros2 run ai_navigator autonomous_navigation --ros-args -p primary_pose_source:=msckf -p vio_topic:=/msckf_vio/odometry

Result: drone armed, took off, entered autonomous navigation.

Two bugs found and fixed same session:
  BUG A — ThreatLevel(Enum) → ThreatLevel(IntEnum)
    Planning loop crashed at ~4Hz post-takeoff: '>=' not supported between ThreatLevel instances
    Fix: obstacle_data.py: class ThreatLevel(IntEnum) + added IntEnum to imports
  BUG B — executor.shutdown(wait=False) invalid on Jazzy
    MultiThreadedExecutor.shutdown() takes no 'wait' kwarg in Jazzy
    Fix: main.py: both calls changed to executor.shutdown() (no args)
  Both committed as bae533a on branch phase0-bug-fixes

## Next session priorities (Skydio-level upgrade plan)
Agreed plan — implement in this order:

### Phase 1 — Fix remaining critical bugs in smart_navigator.py
  C1: _setup_enhanced_computer_vision() sets EMERGENCY state in __init__ on YOLO fail → use flag instead
  C2+C3: lidar_down_callback defined but never subscribed; _lidar_processing_loop always passes empty array
  C4: _pixel_to_world_coordinates() hardcoded intrinsics fx=465.74,cx=320,cy=240 → ROS params
  H1: _vision/_lidar/_planning_processing_loop bare while True → add stop events
  H2: KalmanFilter._update() np.linalg.inv(S) → np.linalg.solve for numerical stability
  M1: cv2.imshow from vision thread (needs main thread) → queue frames to main thread or disable
  M4: __del__ calls cv2.destroyAllWindows → move to destroy_node()

### Phase 2 — SLAM / loop closure
  - Pose graph node using MSCKF odometry + keyframe detection
  - Loop closure via descriptor matching (BoW style, pure numpy/scipy — no open3d/g2o installed)
  - New ROS2 node: module3_vio/slam_backend.py

### Phase 3 — 3D occupancy & better planning
  - Replace flat obstacle grid with 3D voxel map (numpy/scipy only)
  - Path planner upgrade: use real-time 3D costmap from stereo depth
  - Trajectory optimization: minimum-snap in numpy

### Phase 4 — Autonomous mission execution
  - Waypoint sequencer with proper state machine
  - ActiveTrack: wire YOLO tracking loop to setpoints
  - Terrain follow: wire lidar-down subscription (currently dead)

### Phase 5 — Tighter VIO integration
  - MSCKF tightly feeds navigation EKF in smart_navigator (currently loosely coupled via topic)

## Available libraries (checked 2026-04-29)
  torch 2.7.0+cu126, torchvision 0.22.0+cu126
  scipy 1.15.2, sklearn 1.7.0, ultralytics (YOLO)
  open3d: NOT installed, g2o: NOT installed, gtsam: NOT installed, sophus: NOT installed
  → All SLAM/3D geometry must be implemented in numpy/scipy

## ai_navigator — github.com/tiwari13/before_dji (branch: phase0-bug-fixes)
- smart_navigator.py — main autonomy node; SensorData refactored for NED convention
  - attitude_quaternion (w,x,y,z), per-source timestamps, vio_* fields for MSCKF
  - validity flags: local_pose_valid, gps_valid, vio_valid
- drone_state.py — lifecycle states: IDLE/TAKEOFF/MOVE/HOVER/LANDING/DISARMED/EMERGENCY/RTH/GPS_DENIED
- start_sensor_bridges.sh — convenience script to launch all sensor bridges
- Python env for ai_navigator: source ~/drone_venv/bin/activate (NOT yolo9_env)

## Environment
Always: source ~/drone_venv/bin/activate && source ~/ros2_ws/install/setup.bash
Build:  source /opt/ros/jazzy/setup.bash && colcon build --packages-select <pkg>
ROS2:   Jazzy (/opt/ros/jazzy)
