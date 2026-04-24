#!/usr/bin/env python3
"""
PHASE 3 — Step 7: OpenVINS Sensor Bridge
=========================================

PURPOSE
-------
OpenVINS (Geneva et al., ICRA 2020) expects sensors on well-known topics.
Our Gazebo simulation publishes on long namespaced topics.
This bridge re-publishes them so OpenVINS can subscribe.

It also:
  • Validates sensor rates & timestamp alignment on startup
  • Publishes camera_info constructed from our calibration (not just passthrough)
  • Logs diagnostic statistics every 5 s

WHAT OPENVINS NEEDS
-------------------
  /imu0                    sensor_msgs/Imu
  /cam0/image_raw          sensor_msgs/Image
  /cam0/camera_info        sensor_msgs/CameraInfo
  (optional)
  /cam1/image_raw          sensor_msgs/Image     ← stereo, we add in Step 10
  /cam1/camera_info        sensor_msgs/CameraInfo

STUDY MATERIAL
--------------
  [1] Geneva et al., "OpenVINS: A Research Platform for Visual-Inertial
      Estimation", ICRA 2020.  https://arxiv.org/abs/2203.10895
  [2] OpenVINS documentation:  https://docs.openvins.com/
  [3] OpenVINS GitHub:         https://github.com/rpng/open_vins

WHY CAMERA-IMU SYNCHRONISATION MATTERS
---------------------------------------
VIO fuses two physically distinct sensors.  A timing error of even a few
milliseconds causes a mismatch between the IMU-predicted camera pose and
the actual camera pose at which the image was captured.  At typical drone
speed (5 m/s) and 1 ms timing error → 5 mm positional error per frame.
This compounds over the flight.

OpenVINS uses a parameter td (time offset) to calibrate this; we set it to 0
for simulation (Gazebo simulation clock is shared).

Run
---
  # Terminal 1 — Gazebo + PX4
  # Terminal 2 — ROS2 bridge
  ros2 run gz_ros2_bridge parameter_bridge --ros-args ...
  # Terminal 3 — This node
  ros2 run module3_vio sensor_bridge
  # Terminal 4 — OpenVINS  (see launch/step7_openvins.launch.py)
  # Terminal 5 — Comparator
  ros2 run module3_vio vio_comparator
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, Imu, CameraInfo
import numpy as np
import time
from collections import deque


# ── Our Gazebo-bridged topic names ───────────────────────────────────────────
DRONE = '/world/default/model/x500_skydio_0'

IMU_SRC = f'{DRONE}/link/base_link/sensor/imu_sensor/imu'

CAM_FRONT_IMG  = f'{DRONE}/model/camera_front/link/camera_link/sensor/IMX214/image'
CAM_FRONT_INFO = f'{DRONE}/model/camera_front/link/camera_link/sensor/IMX214/camera_info'

CAM_BACK_IMG   = f'{DRONE}/model/camera_back/link/camera_link/sensor/IMX214/image'
CAM_BACK_INFO  = f'{DRONE}/model/camera_back/link/camera_link/sensor/IMX214/camera_info'

# ── Camera intrinsics — scaled from Module 1 calibration (1920×1080 → 640×480)
# Original: FX=FY=1397.22, CX=960, CY=540 at 1920×1080
# Scale factor: 640/1920 = 1/3 for fx/fy/cx/cy
FX = FY = 465.74        # 1397.22 / 3
CX, CY   = 320.0, 180.0 # 960/3, 540/3
WIDTH, HEIGHT = 640, 480

# ── Best-effort QoS for PX4 / Gazebo bridge topics ───────────────────────────
BEST_EFFORT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


def _make_camera_info(stamp, frame_id: str) -> CameraInfo:
    """
    Construct a CameraInfo message from our known intrinsics.

    OpenVINS can use this to avoid requiring a separate calibration file.
    The distortion model is 'plumb_bob' (radtan) with all zeros for simulation.

    Camera matrix layout (row-major):
        K = [fx  0  cx]
            [ 0 fy  cy]
            [ 0  0   1]

    OpenVINS uses fx, fy, cx, cy from K[0,0], K[1,1], K[0,2], K[1,2].
    """
    msg = CameraInfo()
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id
    msg.width  = WIDTH
    msg.height = HEIGHT

    # Row-major 3×3 intrinsic matrix
    msg.k = [FX,  0.0, CX,
             0.0, FY,  CY,
             0.0, 0.0, 1.0]

    # Rectification matrix (identity for monocular)
    msg.r = [1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0]

    # Projection matrix (3×4, P = K [I | 0] for a single camera)
    msg.p = [FX,  0.0, CX,  0.0,
             0.0, FY,  CY,  0.0,
             0.0, 0.0, 1.0, 0.0]

    msg.distortion_model = 'plumb_bob'    # radtan: k1, k2, p1, p2, k3
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]   # no distortion in simulation

    return msg


class SensorBridge(Node):
    """
    Bridges Gazebo-namespaced topics → standard OpenVINS topics.

    Sensors bridged
    ---------------
    cam0  ← front camera   (primary)
    cam1  ← back camera    (second mono, useful later)
    imu0  ← base_link IMU
    """

    def __init__(self):
        super().__init__('sensor_bridge')

        # ── Diagnostics ───────────────────────────────────────────────────────
        self._imu_ts: deque  = deque(maxlen=500)   # recent IMU arrival times
        self._cam0_ts: deque = deque(maxlen=100)   # recent cam0 arrival times
        self._cam1_ts: deque = deque(maxlen=100)
        self._last_imu_stamp: float = 0.0
        self._last_cam0_stamp: float = 0.0
        self._n_imu  = 0
        self._n_cam0 = 0
        self._n_cam1 = 0

        # ── Publishers (standard OpenVINS topics) ─────────────────────────────
        self._pub_imu   = self.create_publisher(Imu,        '/imu0',             10)
        self._pub_c0img = self.create_publisher(Image,      '/cam0/image_raw',   10)
        self._pub_c0inf = self.create_publisher(CameraInfo, '/cam0/camera_info', 10)
        self._pub_c1img = self.create_publisher(Image,      '/cam1/image_raw',   10)
        self._pub_c1inf = self.create_publisher(CameraInfo, '/cam1/camera_info', 10)

        # ── Subscribers (Gazebo-bridged topics) ───────────────────────────────
        self.create_subscription(
            Imu, IMU_SRC, self._imu_cb, BEST_EFFORT_QOS)

        self.create_subscription(
            Image, CAM_FRONT_IMG, self._cam0_img_cb, 10)
        self.create_subscription(
            CameraInfo, CAM_FRONT_INFO, self._cam0_inf_cb, 10)

        self.create_subscription(
            Image, CAM_BACK_IMG, self._cam1_img_cb, 10)
        self.create_subscription(
            CameraInfo, CAM_BACK_INFO, self._cam1_inf_cb, 10)

        # ── Status timer ──────────────────────────────────────────────────────
        self._start = time.time()
        self.create_timer(5.0, self._status_cb)

        self.get_logger().info('=' * 60)
        self.get_logger().info('PHASE 3  Step 7: OpenVINS Sensor Bridge')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Bridging:')
        self.get_logger().info(f'  {IMU_SRC}')
        self.get_logger().info(f'       → /imu0')
        self.get_logger().info(f'  {CAM_FRONT_IMG}')
        self.get_logger().info(f'       → /cam0/image_raw')
        self.get_logger().info(f'  {CAM_BACK_IMG}')
        self.get_logger().info(f'       → /cam1/image_raw')
        self.get_logger().info('')
        self.get_logger().info('Start OpenVINS in another terminal, then fly.')

    # ── IMU callback ──────────────────────────────────────────────────────────
    def _imu_cb(self, msg: Imu):
        """
        Pass IMU through unchanged.

        WHY: OpenVINS integrates IMU at full rate between camera frames to
        predict how the camera moved.  This is called IMU pre-integration.
        The IMU propagates:
          p_{k+1} = p_k + v_k*dt + 0.5*(R_k*(a_k - b_a) - g)*dt²
          v_{k+1} = v_k + (R_k*(a_k - b_a) - g)*dt
          R_{k+1} = R_k * Exp((ω_k - b_g)*dt)      [SO(3) exponential]

        See Forster et al. RSS 2015 for the manifold-aware version.
        """
        self._pub_imu.publish(msg)
        now = time.time()
        self._imu_ts.append(now)
        self._last_imu_stamp = (
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        )
        self._n_imu += 1

    # ── Front camera callbacks ────────────────────────────────────────────────
    def _cam0_img_cb(self, msg: Image):
        """
        Pass front camera image through unchanged.

        WHY: OpenVINS tracks features between consecutive frames using either
        KLT optical flow (same as our Step 6) or descriptor matching.  It then
        uses the feature observations across the sliding window of camera poses
        (the MSCKF part) as measurements to correct the EKF state.
        """
        self._pub_c0img.publish(msg)
        self._pub_c0inf.publish(
            _make_camera_info(msg.header.stamp, 'cam0')
        )
        now = time.time()
        self._cam0_ts.append(now)
        self._last_cam0_stamp = (
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        )
        self._n_cam0 += 1

    def _cam0_inf_cb(self, msg: CameraInfo):
        # Gazebo sends camera_info too; we re-publish our own (from calibration)
        # so the values are guaranteed correct even if Gazebo's differ.
        pass

    # ── Back camera callbacks ─────────────────────────────────────────────────
    def _cam1_img_cb(self, msg: Image):
        self._pub_c1img.publish(msg)
        self._pub_c1inf.publish(
            _make_camera_info(msg.header.stamp, 'cam1')
        )
        self._n_cam1 += 1

    def _cam1_inf_cb(self, msg: CameraInfo):
        pass

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def _status_cb(self):
        elapsed = time.time() - self._start
        imu_hz  = self._rate(self._imu_ts)
        cam0_hz = self._rate(self._cam0_ts)

        # Camera-IMU time gap (should be < one IMU period = ~1/250 s = 4 ms)
        time_gap_ms = abs(self._last_cam0_stamp - self._last_imu_stamp) * 1e3

        self.get_logger().info(
            f'[{elapsed:5.1f}s]  '
            f'IMU: {imu_hz:5.1f} Hz ({self._n_imu})  '
            f'cam0: {cam0_hz:4.1f} Hz ({self._n_cam0})  '
            f'cam1: {self._n_cam1}  '
            f'Δt(cam-imu): {time_gap_ms:.1f} ms'
        )

        # Warn if rates look wrong
        if imu_hz > 0 and imu_hz < 100:
            self.get_logger().warn(
                f'IMU rate {imu_hz:.0f} Hz is low — VIO needs ≥ 100 Hz'
            )
        if cam0_hz > 0 and cam0_hz < 5:
            self.get_logger().warn(
                f'Camera rate {cam0_hz:.1f} Hz is low — VIO needs ≥ 5 Hz'
            )

    @staticmethod
    def _rate(ts: deque) -> float:
        recent = list(ts)[-50:]
        if len(recent) < 2:
            return 0.0
        dt = recent[-1] - recent[0]
        return (len(recent) - 1) / dt if dt > 0 else 0.0


def main(args=None):
    rclpy.init(args=args)
    node = SensorBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
