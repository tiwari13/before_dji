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
  • Passes camera_info through from Gazebo (guarantees calibration consistency)
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
import copy
import time
from collections import deque


# ── Our Gazebo-bridged topic names ───────────────────────────────────────────
DRONE = '/world/default/model/x500_skydio_0'

IMU_SRC = f'{DRONE}/link/base_link/sensor/imu_sensor/imu'

CAM_FRONT_IMG  = f'{DRONE}/model/camera_front/link/camera_link/sensor/IMX214/image'
CAM_FRONT_INFO = f'{DRONE}/model/camera_front/link/camera_link/sensor/IMX214/camera_info'

CAM_BACK_IMG   = f'{DRONE}/model/camera_back/link/camera_link/sensor/IMX214/image'
CAM_BACK_INFO  = f'{DRONE}/model/camera_back/link/camera_link/sensor/IMX214/camera_info'

# ── Best-effort QoS for PX4 / Gazebo bridge topics ───────────────────────────
BEST_EFFORT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


class SensorBridge(Node):
    """
    Bridges Gazebo-namespaced topics → standard OpenVINS topics.

    Sensors bridged
    ---------------
    cam0  ← front camera   (primary)
    cam1  ← back camera    (second mono, useful later)
    imu0  ← base_link IMU

    Calibration
    -----------
    CameraInfo is passed through directly from Gazebo — both the image and
    the calibration always come from the same source, so they are guaranteed
    to be consistent regardless of the actual sensor resolution.
    """

    def __init__(self):
        super().__init__('sensor_bridge')

        # ── Latest CameraInfo from Gazebo (populated on first message) ─────────
        self._cam0_info: CameraInfo | None = None
        self._cam1_info: CameraInfo | None = None
        self._cam0_info_logged = False
        self._cam1_info_logged = False

        # ── Diagnostics ───────────────────────────────────────────────────────
        self._imu_ts:  deque = deque(maxlen=500)   # recent IMU arrival times
        self._cam0_ts: deque = deque(maxlen=100)   # recent cam0 arrival times
        self._cam1_ts: deque = deque(maxlen=100)   # recent cam1 arrival times
        self._last_imu_stamp:  float = 0.0
        self._last_cam0_stamp: float = 0.0
        self._last_cam1_stamp: float = 0.0
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
        self.get_logger().info('Waiting for CameraInfo from Gazebo...')
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

        The CameraInfo is published co-temporally from the stored Gazebo info
        so that the image and its calibration are always consistent.
        """
        # Update frame_id to match OpenVINS convention, keep everything else
        msg.header.frame_id = 'cam0'
        self._pub_c0img.publish(msg)

        # Republish matching CameraInfo with updated stamp and frame_id
        if self._cam0_info is not None:
            info = copy.deepcopy(self._cam0_info)
            info.header.stamp    = msg.header.stamp
            info.header.frame_id = 'cam0'
            self._pub_c0inf.publish(info)

        now = time.time()
        self._cam0_ts.append(now)
        self._last_cam0_stamp = (
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        )
        self._n_cam0 += 1

    def _cam0_inf_cb(self, msg: CameraInfo):
        """Store Gazebo's CameraInfo for co-temporal republishing with images."""
        self._cam0_info = msg
        if not self._cam0_info_logged:
            self._cam0_info_logged = True
            self.get_logger().info(
                f'cam0 CameraInfo received: '
                f'{msg.width}×{msg.height}  '
                f'fx={msg.k[0]:.2f}  fy={msg.k[4]:.2f}  '
                f'cx={msg.k[2]:.2f}  cy={msg.k[5]:.2f}'
            )

    # ── Back camera callbacks ─────────────────────────────────────────────────
    def _cam1_img_cb(self, msg: Image):
        msg.header.frame_id = 'cam1'
        self._pub_c1img.publish(msg)

        if self._cam1_info is not None:
            info = copy.deepcopy(self._cam1_info)
            info.header.stamp    = msg.header.stamp
            info.header.frame_id = 'cam1'
            self._pub_c1inf.publish(info)

        now = time.time()
        self._cam1_ts.append(now)
        self._last_cam1_stamp = (
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        )
        self._n_cam1 += 1

    def _cam1_inf_cb(self, msg: CameraInfo):
        """Store Gazebo's CameraInfo for co-temporal republishing with images."""
        self._cam1_info = msg
        if not self._cam1_info_logged:
            self._cam1_info_logged = True
            self.get_logger().info(
                f'cam1 CameraInfo received: '
                f'{msg.width}×{msg.height}  '
                f'fx={msg.k[0]:.2f}  fy={msg.k[4]:.2f}  '
                f'cx={msg.k[2]:.2f}  cy={msg.k[5]:.2f}'
            )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def _status_cb(self):
        elapsed = time.time() - self._start
        imu_hz  = self._rate(self._imu_ts)
        cam0_hz = self._rate(self._cam0_ts)
        cam1_hz = self._rate(self._cam1_ts)

        # Time gap between camera header stamps and the last IMU header stamp.
        # In simulation the shared clock keeps this near zero; a large value
        # (> ~100 ms) suggests a topic is stalled, not a calibration error.
        gap0_ms = abs(self._last_cam0_stamp - self._last_imu_stamp) * 1e3
        gap1_ms = abs(self._last_cam1_stamp - self._last_imu_stamp) * 1e3

        self.get_logger().info(
            f'[{elapsed:5.1f}s]  '
            f'IMU: {imu_hz:5.1f} Hz ({self._n_imu})  '
            f'cam0: {cam0_hz:4.1f} Hz ({self._n_cam0})  '
            f'cam1: {cam1_hz:4.1f} Hz ({self._n_cam1})  '
            f'Δt(cam0-imu): {gap0_ms:.1f} ms  '
            f'Δt(cam1-imu): {gap1_ms:.1f} ms'
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
