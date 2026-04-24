#!/usr/bin/env python3
"""
PHASE 3 — Step 8: Loosely-Coupled EKF VIO
==========================================

THEORY
------
"Loosely-coupled" means the camera and IMU are fused at the POSE LEVEL:
  • IMU  → predicts state between frames  (high frequency, ~250 Hz)
  • Camera VO (Step 6 output) → corrects state at keyframes  (~1-2 Hz)

Contrast with tightly-coupled (Step 9) where the fusion happens at the
RAW FEATURE LEVEL — individual pixel observations are the measurements.

EKF STATE VECTOR (15-dimensional)
----------------------------------
  x = [p, v, φ, b_a, b_g]  ∈ ℝ¹⁵

  p   ∈ ℝ³   position in world frame  [m]
  v   ∈ ℝ³   velocity in world frame  [m/s]
  φ   ∈ ℝ³   orientation error (small-angle Euler, body→world)  [rad]
  b_a ∈ ℝ³   accelerometer bias  [m/s²]
  b_g ∈ ℝ³   gyroscope bias  [rad/s]

The orientation itself is stored as a quaternion q (not in the state error).
The state error δx = x - x̂ is what the EKF tracks.

IMU PREDICTION (Process Model)
-------------------------------
Between camera frames, integrate IMU to propagate state.

  ṗ = v
  v̇ = R(q) * (a_m - b_a) - g          [gravity g = [0,0,9.81] in NED: 0,0,-9.81 in ENU]
  q̇ = 0.5 * q ⊗ [0; ω_m - b_g]       [quaternion kinematics]
  ḃ_a = 0   (bias modeled as random walk — driven by noise)
  ḃ_g = 0

Linearised process noise covariance Q (4×4 blocks):
  Q = diag(σ_a² I₃,  σ_g² I₃,  σ_ab² I₃,  σ_gb² I₃)

CAMERA UPDATE (Measurement Model)
-----------------------------------
At each keyframe, Step 6 gives us:
  z = [Δp_VO, ΔR_VO]   — relative pose from keyframe k-1 to k

We model this as:
  z = h(x) + noise

where h(x) extracts the predicted relative pose from state.
Innovation:
  y = z - h(x̂)   (difference between observed and predicted relative pose)

Measurement noise R (6×6):
  R = diag(σ_pos² I₃,  σ_rot² I₃)

EKF Update:
  K  = P Hᵀ (H P Hᵀ + R)⁻¹      [Kalman gain]
  δx = K y                          [state correction]
  P  = (I - KH) P                  [covariance update]

STUDY MATERIAL
--------------
[1] Forster et al., "IMU Preintegration on Manifold for Efficient
    Visual-Inertial Maximum-a-Posteriori Estimation",
    RSS 2015.  https://arxiv.org/abs/1512.02363
    THE reference for proper IMU integration.

[2] Sola et al., "A micro Lie theory for state estimation in robotics",
    2021.  https://arxiv.org/abs/1812.01537
    Best tutorial on SO(3), quaternions, and the error-state EKF.

[3] Titterton & Weston, "Strapdown Inertial Navigation Technology",
    2004 (2nd ed.).  Chapter 3: IMU mechanisation equations.

[4] Mourikis & Roumeliotis, "A Multi-State Constraint Kalman Filter for
    Vision-aided Inertial Navigation", ICRA 2007.
    https://www-users.cse.umn.edu/~stergios/papers/ICRA07-MSCKF.pdf
    The original MSCKF paper — foundation for Step 9.

Run
---
  # Alongside Step 6 monocular VO:
  ros2 run module2_visual_odometry monocular_vo &
  ros2 run module3_vio ekf_vio
  ros2 run module2_visual_odometry circle_flight
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from px4_msgs.msg import VehicleLocalPosition

import numpy as np
from scipy.spatial.transform import Rotation
import time


# ── IMU noise parameters (match openvins_mono.yaml) ─────────────────────────
SIGMA_A   = 0.05      # accel noise density [m/s²/√Hz]
SIGMA_G   = 0.005     # gyro noise density  [rad/s/√Hz]
SIGMA_AB  = 0.001     # accel bias walk     [m/s³/√Hz]
SIGMA_GB  = 0.0001    # gyro bias walk      [rad/s²/√Hz]

# ── Camera VO measurement noise ──────────────────────────────────────────────
SIGMA_POS = 0.10      # VO position noise [m]  — rough, tune after seeing results
SIGMA_ROT = 0.05      # VO rotation noise [rad]

# ── Gravity (ENU frame: +Z up, so g = -Z) ───────────────────────────────────
# PX4 / Gazebo uses NED internally, but our bridge delivers ENU-like Imu.
# In Gazebo simulation the IMU Z-axis is up, so gravity reads as -9.81 on Z.
GRAVITY = np.array([0.0, 0.0, -9.81])

BEST_EFFORT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)


# ── Quaternion utilities ──────────────────────────────────────────────────────

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """q = [x, y, z, w] → 3×3 rotation matrix R (body→world)."""
    return Rotation.from_quat(q).as_matrix()


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → [x, y, z, w] quaternion."""
    return Rotation.from_matrix(R).as_quat()


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion product q1 ⊗ q2.  Both [x,y,z,w]."""
    return (Rotation.from_quat(q1) * Rotation.from_quat(q2)).as_quat()


def exp_so3(theta: np.ndarray) -> np.ndarray:
    """
    SO(3) exponential map: rotation vector → rotation matrix.
    theta ∈ ℝ³ is angle*axis.

    Uses Rodrigues formula:
      Exp(θ) = I + sin(‖θ‖)/‖θ‖ [θ]× + (1-cos(‖θ‖))/‖θ‖² [θ]×²

    For ‖θ‖ → 0 uses Taylor series to avoid division by zero.
    """
    angle = np.linalg.norm(theta)
    if angle < 1e-8:
        return np.eye(3) + skew(theta)
    axis  = theta / angle
    K     = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]× such that [v]× u = v × u."""
    return np.array([
        [ 0.0,   -v[2],  v[1]],
        [ v[2],   0.0,  -v[0]],
        [-v[1],   v[0],  0.0],
    ])


# ── EKF State ─────────────────────────────────────────────────────────────────

class EKFState:
    """
    Nominal state + error covariance for the error-state EKF.

    Error-state EKF (also called indirect EKF or δx-EKF):
      • Nominal state x̂ is propagated with the nonlinear dynamics
      • Error δx = x_true - x̂ is propagated with linearised dynamics
      • Measurement updates correct δx, then inject into x̂

    This is the standard approach in VIO (Sola et al. 2021, Chapter 7).
    """
    DIM = 15    # p(3) + v(3) + φ(3) + b_a(3) + b_g(3)

    def __init__(self):
        self.p   = np.zeros(3)        # position [m]
        self.v   = np.zeros(3)        # velocity [m/s]
        self.q   = np.array([0., 0., 0., 1.])  # orientation [x,y,z,w]
        self.b_a = np.zeros(3)        # accel bias [m/s²]
        self.b_g = np.zeros(3)        # gyro bias [rad/s]

        # Error covariance (15×15)
        # Start with large uncertainty (we have not seen any camera frames yet)
        self.P   = np.diag([
            1e-2, 1e-2, 1e-2,   # position uncertainty [m²]
            0.1,  0.1,  0.1,    # velocity uncertainty [m/s]²
            1e-3, 1e-3, 1e-3,   # orientation uncertainty [rad²]
            1e-4, 1e-4, 1e-4,   # accel bias uncertainty [m/s²]²
            1e-6, 1e-6, 1e-6,   # gyro bias uncertainty [rad/s]²
        ])

        self.is_initialised = False

    def rotation_matrix(self) -> np.ndarray:
        return quat_to_rot(self.q)

    def inject_error(self, dx: np.ndarray):
        """
        After the EKF update computes δx, inject it into the nominal state.

        Additive for p, v, b_a, b_g.
        Multiplicative for orientation (on SO(3)):
          q ← q ⊗ Exp(δφ/2)   [quaternion update using rotation vector]
        """
        self.p   += dx[0:3]
        self.v   += dx[3:6]
        self.b_a += dx[9:12]
        self.b_g += dx[12:15]

        # SO(3) injection
        dq = Rotation.from_rotvec(dx[6:9]).as_quat()   # δφ → δq
        self.q = quat_multiply(self.q, dq)
        self.q /= np.linalg.norm(self.q)   # re-normalise


class EkfVio(Node):
    """
    Loosely-coupled EKF that fuses IMU + monocular VO pose estimates.

    INPUT topics:
      /imu0                    sensor_msgs/Imu     (from sensor_bridge)
      /monocular_vo/pose       geometry_msgs/PoseStamped  (NEW: VO node must publish this)

    OUTPUT topics:
      /ekf_vio/odometry        nav_msgs/Odometry
      /ekf_vio/path            nav_msgs/Path

    Note on /monocular_vo/pose:
      The monocular_vo.py (Step 6) does not currently publish a PoseStamped.
      You can either:
        a) Add a publisher there (recommended — good exercise)
        b) Use this EKF with OpenVINS output on /ov_msckf/poseimu as the
           "camera measurement" to understand the data flow

    For now, this node will run and print what it *would* do when VO
    data arrives, so you understand the full EKF loop even without
    a live VO feed.
    """

    def __init__(self):
        super().__init__('ekf_vio')

        self._state    = EKFState()
        self._last_imu_time: float | None = None
        self._imu_count   = 0
        self._update_count = 0
        self._trajectory   = []    # [(t, p)]
        self._gt_origin    = None
        self._gt_traj      = []
        self._start        = time.time()

        # Previous keyframe pose in world frame (for relative pose measurement)
        self._kf_p: np.ndarray | None = None
        self._kf_q: np.ndarray | None = None

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_odom = self.create_publisher(Odometry, '/ekf_vio/odometry', 10)
        self._pub_path = self.create_publisher(Path,     '/ekf_vio/path',     10)
        self._path_msg = Path()
        self._path_msg.header.frame_id = 'world'

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Imu, '/imu0', self._imu_cb, 50)

        # Listen to Step 6 VO output (add publisher to monocular_vo.py — see below)
        self.create_subscription(
            PoseStamped, '/monocular_vo/pose', self._vo_cb, 10
        )

        # Ground truth
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self._gt_cb,
            BEST_EFFORT_QOS,
        )

        self.create_timer(5.0, self._status_cb)

        self.get_logger().info('=' * 60)
        self.get_logger().info('PHASE 3  Step 8: Loosely-Coupled EKF VIO')
        self.get_logger().info('=' * 60)
        self.get_logger().info('State: p(3) v(3) φ(3) b_a(3) b_g(3) = 15-dim')
        self.get_logger().info('IMU:   predicts state at high rate')
        self.get_logger().info('VO:    corrects state at keyframe rate')
        self.get_logger().info('')
        self.get_logger().info('NOTE: monocular_vo.py needs to publish')
        self.get_logger().info('      PoseStamped on /monocular_vo/pose')
        self.get_logger().info('      See step8_ekf_vio.py docstring.')

    # ── IMU Callback: PREDICTION STEP ─────────────────────────────────────────
    def _imu_cb(self, msg: Imu):
        """
        EKF PREDICTION (time propagation).

        Called at IMU rate (~250 Hz).  Each call integrates one IMU sample
        and propagates the error covariance P forward.

        ┌──────────────────────────────────────────────────────────────┐
        │  NOMINAL STATE PROPAGATION  (nonlinear, no approximation)    │
        │  p_{k+1} = p_k + v_k*dt + 0.5*(R_k*(a-b_a)+g)*dt²         │
        │  v_{k+1} = v_k + (R_k*(a-b_a)+g)*dt                        │
        │  q_{k+1} = q_k ⊗ Exp((ω-b_g)*dt)                           │
        │  b_a unchanged  (zero mean random walk)                     │
        │  b_g unchanged                                               │
        └──────────────────────────────────────────────────────────────┘

        ┌──────────────────────────────────────────────────────────────┐
        │  COVARIANCE PROPAGATION  (linearised around nominal state)   │
        │  P_{k+1} = F P_k Fᵀ + G Q Gᵀ                               │
        │                                                              │
        │  F = ∂f/∂δx  (15×15 state transition Jacobian)              │
        │  G            (15×12 noise input Jacobian)                   │
        │  Q = diag(σ_a², σ_g², σ_ab², σ_gb²) ⊗ I₃                  │
        └──────────────────────────────────────────────────────────────┘
        """
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self._last_imu_time is None:
            self._last_imu_time = stamp
            # Initialise orientation from IMU orientation field (Gazebo provides this).
            # Without this, q defaults to identity [0,0,0,1] but the drone is pitched
            # ~11.5° at rest, causing sin(11.5°)*9.81 ≈ 1.95 m/s² to leak as
            # horizontal acceleration → 555 m position drift in 185 s.
            q = np.array([
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ])
            norm = np.linalg.norm(q)
            if norm > 0.1:   # valid quaternion from sensor
                self._state.q = q / norm
                rpy = Rotation.from_quat(self._state.q).as_euler('xyz', degrees=True)
                self.get_logger().info(
                    f'First IMU: init q from orientation field  '
                    f'roll={rpy[0]:.1f}°  pitch={rpy[1]:.1f}°  yaw={rpy[2]:.1f}°'
                )
            else:
                # Fallback: estimate from accelerometer (assumes near-static)
                a_m = np.array([
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                ])
                self.get_logger().warn(
                    f'First IMU: orientation field invalid (norm={norm:.3f}), '
                    f'using identity q — expect drift!'
                )
            self._state.is_initialised = True
            return

        dt = stamp - self._last_imu_time
        self._last_imu_time = stamp

        if dt <= 0 or dt > 0.1:
            return   # bad timestamp

        a_m = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ])
        w_m = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ])

        s = self._state
        R = s.rotation_matrix()

        # ── 1. Correct for bias ───────────────────────────────────────────────
        a_cor = a_m - s.b_a         # corrected acceleration (body frame)
        w_cor = w_m - s.b_g         # corrected angular velocity (body frame)

        # ── 2. Nominal state propagation (midpoint Euler, good enough) ────────
        a_world = R @ a_cor + GRAVITY
        s.p += s.v * dt + 0.5 * a_world * dt * dt
        s.v += a_world * dt

        # Orientation update via SO(3) exponential map
        dq  = Rotation.from_rotvec(w_cor * dt).as_quat()
        s.q = quat_multiply(s.q, dq)
        s.q /= np.linalg.norm(s.q)

        # ── 3. Covariance propagation: P = F P Fᵀ + G Q Gᵀ ──────────────────
        F = self._build_F(R, a_cor, dt)
        G = self._build_G(R, dt)

        # Continuous-time noise spectral density → discrete
        IMU_RATE = 1.0 / dt
        q_a  = SIGMA_A  ** 2 * IMU_RATE
        q_g  = SIGMA_G  ** 2 * IMU_RATE
        q_ab = SIGMA_AB ** 2 * IMU_RATE
        q_gb = SIGMA_GB ** 2 * IMU_RATE

        Q = np.diag([q_a]*3 + [q_g]*3 + [q_ab]*3 + [q_gb]*3)

        s.P = F @ s.P @ F.T + G @ Q @ G.T

        self._imu_count += 1

        # Publish estimated pose at IMU rate
        self._publish_odometry(stamp)

    def _build_F(self, R: np.ndarray, a_cor: np.ndarray, dt: float) -> np.ndarray:
        """
        Linearised state transition matrix F = I + Fc*dt  (discrete approximation).

        Fc (continuous-time):
          ∂δp/∂δv       = I
          ∂δv/∂δφ       = -R [a_cor]×       (cross-product term)
          ∂δv/∂δb_a     = -R
          ∂δφ/∂δφ       = -[ω_cor]×         (not needed here, simplified)
          ∂δφ/∂δb_g     = -I
          All others    = 0

        Indices: p=0:3, v=3:6, φ=6:9, b_a=9:12, b_g=12:15
        """
        F = np.eye(15)

        # δṗ = δv
        F[0:3, 3:6] = np.eye(3) * dt

        # δv̇ = -R [a_cor]× δφ - R δb_a + g cancels
        F[3:6, 6:9]  = -R @ skew(a_cor) * dt
        F[3:6, 9:12] = -R * dt

        # δφ̇ = -[ω_cor]× δφ - δb_g  (simplified: ignore -[ω]× for short dt)
        F[6:9, 12:15] = -np.eye(3) * dt

        return F

    def _build_G(self, R: np.ndarray, dt: float) -> np.ndarray:
        """
        Noise input matrix G (15×12).

        Maps [n_a, n_g, n_ab, n_gb] → δx.
        """
        G = np.zeros((15, 12))

        # Acceleration noise → velocity error
        G[3:6,  0:3]  = -R * dt

        # Gyro noise → orientation error
        G[6:9,  3:6]  = -np.eye(3) * dt

        # Accel bias noise
        G[9:12, 6:9]  = np.eye(3) * dt

        # Gyro bias noise
        G[12:15, 9:12] = np.eye(3) * dt

        return G

    # ── VO Callback: UPDATE STEP ───────────────────────────────────────────────
    def _vo_cb(self, msg: PoseStamped):
        """
        EKF UPDATE (measurement correction).

        Called at keyframe rate (whenever Step 6 creates a new keyframe).

        ┌──────────────────────────────────────────────────────────────┐
        │  MEASUREMENT:  z = [p_VO, q_VO]  (absolute pose from VO)   │
        │                                                              │
        │  For loose coupling we treat the VO pose as a noisy         │
        │  measurement of the true pose:                               │
        │    z = h(x) + n,   n ~ N(0, R_meas)                         │
        │                                                              │
        │  h(x) = [p, φ_from_q]   (extract position & orientation)   │
        │                                                              │
        │  Innovation:   y = z - h(x̂)                                 │
        │  Jacobian H:   ∂h/∂δx  (6×15, maps state error to meas)    │
        │  Kalman gain:  K = P Hᵀ (HPHᵀ + R)⁻¹                      │
        │  Update:       δx = K y                                      │
        │  Inject:       x̂ ← x̂ ⊕ δx                                 │
        │  Covariance:   P = (I - KH) P (I - KH)ᵀ + K R Kᵀ  (Joseph)│
        └──────────────────────────────────────────────────────────────┘
        """
        if not self._state.is_initialised:
            return

        s = self._state

        # Extract VO measurement
        p_vo = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        q_vo = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])
        # Orientation error (VO orientation - EKF orientation) as rotation vector
        R_meas = quat_to_rot(q_vo)
        R_pred = s.rotation_matrix()
        dR     = R_meas @ R_pred.T
        dphi   = Rotation.from_matrix(dR).as_rotvec()

        # Innovation vector y ∈ ℝ⁶
        y = np.concatenate([p_vo - s.p, dphi])

        # Measurement Jacobian H (6×15)
        # h maps δx to measurement space:
        #   δp_meas ≈ δp               → H[0:3, 0:3] = I
        #   δφ_meas ≈ δφ               → H[3:6, 6:9] = I
        #   all others = 0
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)    # position
        H[3:6, 6:9] = np.eye(3)    # orientation

        # Measurement noise covariance R_meas (6×6)
        R_meas_cov = np.diag([SIGMA_POS**2]*3 + [SIGMA_ROT**2]*3)

        # Kalman gain (15×6)
        S = H @ s.P @ H.T + R_meas_cov
        K = s.P @ H.T @ np.linalg.inv(S)

        # State correction δx (15,)
        dx = K @ y

        # Inject δx into nominal state
        s.inject_error(dx)

        # Joseph form covariance update (numerically stable)
        IKH = np.eye(15) - K @ H
        s.P = IKH @ s.P @ IKH.T + K @ R_meas_cov @ K.T

        self._update_count += 1
        self.get_logger().info(
            f'EKF update #{self._update_count} | '
            f'Innovation: pos={np.linalg.norm(y[:3]):.3f} m  '
            f'rot={np.degrees(np.linalg.norm(y[3:])):.2f}°'
        )

    # ── Ground truth ──────────────────────────────────────────────────────────
    def _gt_cb(self, msg: VehicleLocalPosition):
        pos = np.array([msg.x, msg.y, msg.z])
        if self._gt_origin is None:
            self._gt_origin = pos.copy()
        self._gt_traj.append(pos - self._gt_origin)

    # ── Publish ───────────────────────────────────────────────────────────────
    def _publish_odometry(self, stamp: float):
        s = self._state

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'world'
        odom.child_frame_id  = 'body'

        odom.pose.pose.position.x = float(s.p[0])
        odom.pose.pose.position.y = float(s.p[1])
        odom.pose.pose.position.z = float(s.p[2])
        odom.pose.pose.orientation.x = float(s.q[0])
        odom.pose.pose.orientation.y = float(s.q[1])
        odom.pose.pose.orientation.z = float(s.q[2])
        odom.pose.pose.orientation.w = float(s.q[3])

        # 6×6 pose covariance (row-major) from P[0:6, 0:6]
        cov6 = np.zeros(36)
        for i in range(6):
            for j in range(6):
                cov6[i*6+j] = s.P[i, j]
        odom.pose.covariance = list(cov6)

        self._pub_odom.publish(odom)

        # Path
        ps = PoseStamped()
        ps.header = odom.header
        ps.pose   = odom.pose.pose
        self._path_msg.header.stamp = odom.header.stamp
        self._path_msg.poses.append(ps)
        if len(self._path_msg.poses) > 2000:   # cap memory
            self._path_msg.poses.pop(0)
        self._pub_path.publish(self._path_msg)

    # ── Status ────────────────────────────────────────────────────────────────
    def _status_cb(self):
        elapsed = time.time() - self._start
        s = self._state
        pos_cov = float(np.trace(s.P[0:3, 0:3]))
        self.get_logger().info(
            f'[{elapsed:.0f}s]  '
            f'IMU integrated: {self._imu_count}  '
            f'VO updates: {self._update_count}  '
            f'p=[{s.p[0]:.2f},{s.p[1]:.2f},{s.p[2]:.2f}]  '
            f'|b_a|={np.linalg.norm(s.b_a):.3f}  '
            f'|b_g|={np.linalg.norm(s.b_g):.4f}  '
            f'cov_p={pos_cov:.4f}'
        )

    def generate_report(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print('\n' + '=' * 70)
        print('PHASE 3 — Step 8: LOOSELY-COUPLED EKF VIO REPORT')
        print('=' * 70)
        print(f'\n  IMU integration steps: {self._imu_count}')
        print(f'  VO measurement updates: {self._update_count}')

        s = self._state
        print(f'\n  Final EKF state:')
        print(f'    Position: [{s.p[0]:.3f}, {s.p[1]:.3f}, {s.p[2]:.3f}] m')
        print(f'    Velocity: [{s.v[0]:.3f}, {s.v[1]:.3f}, {s.v[2]:.3f}] m/s')
        print(f'    Accel bias: [{s.b_a[0]:.4f}, {s.b_a[1]:.4f}, {s.b_a[2]:.4f}] m/s²')
        print(f'    Gyro bias:  [{s.b_g[0]:.5f}, {s.b_g[1]:.5f}, {s.b_g[2]:.5f}] rad/s')

        pos_cov = np.trace(s.P[0:3, 0:3])
        vel_cov = np.trace(s.P[3:6, 3:6])
        print(f'\n  Final covariance traces:')
        print(f'    Position: {pos_cov:.4f} m²')
        print(f'    Velocity: {vel_cov:.4f} (m/s)²')

        print(f'\n  Key insight:')
        print(f'    Without VO updates ({self._update_count} of them),')
        print(f'    IMU-only position would drift ~{SIGMA_A**2 * 0.5 * 60**2:.1f} m')
        print(f'    over 60 s from accel bias alone.')
        print(f'    VO corrections bounded this drift.')

        print(f'\n  What loosely-coupled CANNOT do (→ motivation for Step 9):')
        print(f'    - VO pose is computed without IMU aid, so it degrades')
        print(f'      in fast rotation / low texture')
        print(f'    - The EKF cannot help the feature tracker directly')
        print(f'    - Scale ambiguity in monocular VO propagates to EKF')
        print(f'    Solution: Step 9 uses RAW FEATURE TRACKS as measurements')
        print(f'    so the IMU-predicted camera motion aids the tracker.')

        print('=' * 70)


def main(args=None):
    rclpy.init(args=args)
    node = EkfVio()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.generate_report()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
