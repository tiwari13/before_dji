#!/usr/bin/env python3
"""
PHASE 3 — Step 9: Multi-State Constraint Kalman Filter (MSCKF)
==============================================================

THEORY
------
Step 8 (loosely-coupled) fused IMU with VO POSE estimates.
Step 9 (tightly-coupled) fuses IMU with raw PIXEL observations directly.

Key insight (Mourikis & Roumeliotis, ICRA 2007):
  A 3-D feature observed from N camera poses gives 2N pixel constraints.
  The feature's 3-D position appears in all of them, but can be
  ANALYTICALLY ELIMINATED via null-space projection (left null space of
  the feature Jacobian).  This yields 2N-3 constraints that depend only
  on camera poses — the feature coords never enter the state vector.

  Result: state size = 15 (IMU) + 6*N (camera window) — bounded regardless
  of how many features are tracked.

STATE VECTOR
-----------
  x = [x_IMU | x_C0 | x_C1 | ... | x_C(N-1)]

  x_IMU = [p(3)  v(3)  φ(3)  b_a(3)  b_g(3)]   15-dim
  x_Ci  = [p_Ci(3)  φ_Ci(3)]                      6-dim each

PIPELINE (per image frame)
--------------------------
  1. IMU propagation   — same as Step 8
  2. State augmentation — append new camera pose p_C = p_I + R_I*T_IC
  3. KLT feature tracking — track features, find lost ones
  4. For each lost feature (observed ≥ MIN_TRACK_LEN times):
       a. Triangulate 3-D position (linear DLT)
       b. Compute residuals + Jacobians across all camera observations
       c. Null-space project out 3-D coords → H_o, r_o  (2N-3 constraints)
       d. Mahalanobis chi-squared outlier test
       e. Stack into batch update
  5. EKF update (batch)
  6. Marginalize oldest camera state if window is full

PROCESS NOISE DISCRETISATION
------------------------------
σ_a, σ_g are continuous-time spectral densities (units: m/s²/√Hz, rad/s/√Hz).

The discrete covariance contribution for a zero-order hold over dt is:

  Q_d = G @ Q_c @ G.T * dt

where Q_c = diag(σ²) (the continuous-time noise matrix) and G maps noise
inputs to state derivatives (units of 1/s after dt scaling, so Q_d has
correct state² units).

The previous form G @ (Q_c/dt) @ G.T * dt = G @ Q_c @ G.T is algebraically
equivalent but obscures the intent — the /dt and *dt cancel, leaving Q_c
unchanged.  The correct first-order discretisation is simply G @ Q_c @ G.T * dt.

STUDY MATERIAL
--------------
[1] Mourikis & Roumeliotis, "A Multi-State Constraint Kalman Filter for
    Vision-aided Inertial Navigation", ICRA 2007.
    https://www-users.cse.umn.edu/~stergios/papers/ICRA07-MSCKF.pdf
    Equations 13-22 are implemented here.

[2] Li & Mourikis, "High-precision, consistent EKF-based visual-inertial
    odometry", IJRR 2013.  https://doi.org/10.1177/0278364913481563
    Observability-constrained linearisation.

[3] Geneva et al., "OpenVINS: A Research Platform for Visual-Inertial
    Estimation", ICRA 2020.  https://arxiv.org/abs/2203.10895
    Open-source reference implementation.

[4] Sola et al., "A micro Lie theory for state estimation in robotics"
    https://arxiv.org/abs/1812.01537
    Background on SO(3) and right-perturbation conventions used here.

FEATURE JACOBIAN CONVENTION
-----------------------------
Right-perturbation is used throughout: R_C → R_C @ exp(skew(δφ_C)).

For a point z_c = R_C.T @ (p_f - p_C) in the camera frame:
  ∂z_c/∂δφ_C = skew(z_c)   (from d/dε [R @ exp(εK)]^T X at ε=0)
  ∂z_c/∂δp_C = -R_C.T
  ∂z_c/∂δp_f =  R_C.T

Run
---
  ros2 run module3_vio msckf_vio
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu, Image, CameraInfo
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import chi2
import cv2
import time
from collections import defaultdict

# ── Camera-IMU extrinsics (camera position/orientation in IMU body frame) ─────
# Front camera: 0.15 m forward, 0.10 m up; axes aligned with IMU
T_IC = np.array([0.15, 0.0, 0.10])   # camera origin in IMU frame [m]
R_IC = np.eye(3)                       # camera axes = IMU axes

# ── IMU continuous-time noise parameters ─────────────────────────────────────
SIGMA_G  = 0.005    # gyro noise density    [rad/s / √Hz]
SIGMA_A  = 0.05     # accel noise density   [m/s² / √Hz]
SIGMA_GB = 0.0001   # gyro bias walk        [rad/s² / √Hz]
SIGMA_AB = 0.001    # accel bias walk       [m/s³ / √Hz]

# ── MSCKF tuning ──────────────────────────────────────────────────────────────
MAX_CAM_STATES = 20   # sliding window size N
MIN_TRACK_LEN  = 3    # min observations before a feature is used for update
MAX_TRACK_LEN  = 25   # max observations before force-marginalizing
PIXEL_STD      = 1.5  # pixel noise std [px] — increase if tracking is noisy
MAX_FEATURES   = 200  # max simultaneously tracked features

# Gravity in world frame.  ENU convention: x=East, y=North, z=Up.
# g_world = [0, 0, -9.81]  (gravity pulls DOWN = negative Z in ENU)
#
# IMU specific force model (standard):
#   a_imu = R_body_to_world.T @ (a_true - g_world)
#   → a_true = R_body_to_world @ a_imu + g_world
#
# In code: acc = R @ ab - GRAVITY, so GRAVITY must be -g_world = +9.81 in Z.
GRAVITY = np.array([0.0, 0.0, 9.81])

BEST_EFFORT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SO(3) utilities  (right-perturbation convention throughout)
# ═══════════════════════════════════════════════════════════════════════════════

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """[x,y,z,w] → 3×3 rotation matrix (body → world)."""
    return Rotation.from_quat(q).as_matrix()


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_quat()


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]])


def exp_so3(theta: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation vector → 3×3 rotation matrix."""
    angle = np.linalg.norm(theta)
    if angle < 1e-8:
        return np.eye(3) + skew(theta)
    ax = theta / angle
    K  = skew(ax)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# ═══════════════════════════════════════════════════════════════════════════════
# Linear triangulation (DLT)
# ═══════════════════════════════════════════════════════════════════════════════

def triangulate_dlt(obs_norm, cam_poses):
    """
    Triangulate a 3-D point from N ≥ 2 normalised pixel observations.

    obs_norm  : list of (u_n, v_n)  — (pixel - principal point) / focal length
    cam_poses : list of (p_C, R_C)
                  p_C: camera origin in world frame
                  R_C: body→world rotation (same convention as R_I)

    Returns point in world frame, or None if degenerate.

    DLT derivation:
      λ [u;v;1] = K [R|t] X    (projection)
      In normalised coords (K=I): [u;v;1] × (R_C.T*(X-p_C)) = 0
      → each observation gives 2 linear rows in X
    """
    A = []
    for (u_n, v_n), (p_C, R_C) in zip(obs_norm, cam_poses):
        # Transform: world→camera.  R_C is body→world, so R_C.T = world→camera
        R_wc = R_C.T                      # world→camera
        t_wc = -(R_C.T @ p_C)            # translation in world→camera convention
        P    = np.hstack([R_wc, t_wc[:, None]])   # 3×4 projection matrix

        A.append(u_n * P[2] - P[0])
        A.append(v_n * P[2] - P[1])

    A = np.array(A)   # 2N × 4
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    X = Vt[-1]
    if abs(X[3]) < 1e-8:
        return None
    return X[:3] / X[3]   # world frame


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Tracker  (KLT optical flow with forward-backward verification)
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureTracker:
    """
    Tracks image features across frames using Lucas-Kanade optical flow.

    tracks[fid] = list of {'cam_id': int, 'u': float, 'v': float}
      where u, v are NORMALISED coords: u = (px - cx)/fx, v = (py - cy)/fy.

    cam_id is the globally-unique ID assigned to each camera state at
    augmentation time.  This lets us match observations to camera states
    even after old states are marginalized (removing them from the window
    changes indices but not IDs).

    Intrinsics (fx, fy, cx, cy, img_w, img_h) are passed in at construction
    and must match the live CameraInfo received by MsckfVio.
    """

    LK_PARAMS = dict(
        winSize   = (21, 21),
        maxLevel  = 3,
        criteria  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 img_w: int, img_h: int):
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._img_w = img_w
        self._img_h = img_h

        self._next_fid = 0
        self._prev_img: np.ndarray | None = None
        self._pts:   np.ndarray | None = None   # Nx1x2 float32
        self._ids:   list[int]          = []
        self.tracks: dict[int, list]    = defaultdict(list)

    def process(self, img_gray: np.ndarray, cam_id: int) -> list[int]:
        """
        Track existing features into the new frame, detect replacements.

        Returns list of feature IDs that are no longer tracked and have
        enough observations to be used for an EKF update.
        """
        lost_ids: list[int] = []

        # First frame — just detect
        if self._prev_img is None:
            self._prev_img = img_gray
            self._detect(img_gray, cam_id)
            return lost_ids

        # ── Forward KLT ──────────────────────────────────────────────────────
        if self._pts is None or len(self._pts) == 0:
            self._prev_img = img_gray
            self._detect(img_gray, cam_id)
            return lost_ids

        next_pts, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_img, img_gray, self._pts, None, **self.LK_PARAMS)

        # ── Backward KLT (forward-backward error check) ───────────────────────
        prev_pts2, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
            img_gray, self._prev_img, next_pts, None, **self.LK_PARAMS)

        fb_err = np.abs(self._pts - prev_pts2).reshape(-1, 2).max(axis=1)
        ok = (st_fwd.flatten() == 1) & (st_bwd.flatten() == 1) & (fb_err < 2.0)

        # In-bounds check using live image dimensions
        nx, ny = next_pts[:, 0, 0], next_pts[:, 0, 1]
        ok &= (nx >= 5) & (nx < self._img_w - 5) & (ny >= 5) & (ny < self._img_h - 5)

        # ── Update tracks ─────────────────────────────────────────────────────
        kept_pts, kept_ids = [], []

        for i, (fid, tracked) in enumerate(zip(self._ids, ok)):
            if tracked:
                px, py = float(next_pts[i, 0, 0]), float(next_pts[i, 0, 1])
                self.tracks[fid].append({
                    'cam_id': cam_id,
                    'u': (px - self._cx) / self._fx,
                    'v': (py - self._cy) / self._fy,
                })
                if len(self.tracks[fid]) >= MAX_TRACK_LEN:
                    # Force marginalize long tracks
                    lost_ids.append(fid)
                else:
                    kept_pts.append(next_pts[i])
                    kept_ids.append(fid)
            else:
                lost_ids.append(fid)

        self._pts = np.array(kept_pts, dtype=np.float32) if kept_pts else None
        self._ids = kept_ids

        # ── Detect new features to fill up to MAX_FEATURES ───────────────────
        n_new = MAX_FEATURES - len(kept_ids)
        if n_new > 10:
            existing = np.array(kept_pts, dtype=np.float32) if kept_pts else None
            self._detect(img_gray, cam_id, n_new=n_new, existing=existing)

        self._prev_img = img_gray
        return lost_ids

    def _detect(self, img: np.ndarray, cam_id: int,
                n_new: int = MAX_FEATURES, existing=None):
        """Detect new Shi-Tomasi corners, avoiding existing feature regions."""
        mask = np.ones(img.shape[:2], np.uint8) * 255
        if existing is not None and len(existing) > 0:
            for pt in existing.reshape(-1, 2):
                cv2.circle(mask, (int(pt[0]), int(pt[1])), 20, 0, -1)

        new_pts = cv2.goodFeaturesToTrack(
            img, maxCorners=n_new, qualityLevel=0.01,
            minDistance=15, mask=mask)
        if new_pts is None:
            return

        for pt in new_pts:
            px, py = float(pt[0, 0]), float(pt[0, 1])
            fid = self._next_fid
            self._next_fid += 1
            self.tracks[fid].append({
                'cam_id': cam_id,
                'u': (px - self._cx) / self._fx,
                'v': (py - self._cy) / self._fy,
            })
            if self._pts is None:
                self._pts = pt.reshape(1, 1, 2)
            else:
                self._pts = np.vstack([self._pts, pt.reshape(1, 1, 2)])
            self._ids.append(fid)


# ═══════════════════════════════════════════════════════════════════════════════
# MSCKF State
# ═══════════════════════════════════════════════════════════════════════════════

class MsckfState:
    """
    Full MSCKF state: IMU state + sliding window of camera poses.

    Error-state EKF with right-perturbation on SO(3):
      R_true ≈ R_nominal @ exp(skew(δφ))

    State ordering in covariance P:
      [ IMU block (15×15) | IMU-Cam cross (15×6N) ]
      [ Cam-IMU cross     | Cam-Cam block  (6N×6N) ]

    Each camera state occupies 6 consecutive rows/cols: [δp_C(3), δφ_C(3)]
    """
    IMU_DIM = 15
    CAM_DIM = 6

    def __init__(self):
        # Nominal IMU state
        self.p   = np.zeros(3)
        self.v   = np.zeros(3)
        self.q   = np.array([0., 0., 0., 1.])   # [x,y,z,w]
        self.b_a = np.zeros(3)
        self.b_g = np.zeros(3)

        # Sliding window of camera states
        # Each entry: {'id': int, 'p': np.array(3), 'q': np.array(4)}
        self.cam_states: list[dict] = []
        self._cam_id_counter = 0

        # Error covariance (starts 15×15, grows with augmentation)
        self.P = np.diag([
            1e-2, 1e-2, 1e-2,    # position [m²]
            0.1,  0.1,  0.1,     # velocity [m²/s²]
            1e-3, 1e-3, 1e-3,    # orientation [rad²]
            1e-4, 1e-4, 1e-4,    # accel bias [m²/s⁴]
            1e-6, 1e-6, 1e-6,    # gyro bias [rad²/s²]
        ])
        self.is_init = False

    @property
    def n(self) -> int:
        """Total error-state dimension."""
        return self.IMU_DIM + self.CAM_DIM * len(self.cam_states)

    def rotation_matrix(self) -> np.ndarray:
        return quat_to_rot(self.q)

    # ── State augmentation ────────────────────────────────────────────────────

    def augment(self) -> int:
        """
        Augment state with a new camera pose derived from current IMU state.

        Camera pose in world frame:
          p_C = p_I + R_I @ T_IC          (camera origin)
          R_C = R_IC.T @ R_I = R_I        (since R_IC = I)

        Augmentation Jacobian J (6 × current_n):
          δp_C = δp_I  +  (-R_I @ [T_IC]×) @ δφ_I
          δφ_C =                        I  @ δφ_I

        Covariance grows as:
          P_new = [P        P @ J.T ]
                  [J @ P    J @ P @ J.T]
        """
        R_I = self.rotation_matrix()
        p_C = self.p + R_I @ T_IC
        q_C = rot_to_quat(R_IC.T @ R_I)

        cam_id = self._cam_id_counter
        self._cam_id_counter += 1
        self.cam_states.append({'id': cam_id, 'p': p_C.copy(), 'q': q_C.copy()})

        # Build augmentation Jacobian (6 × current_n before augmentation)
        n_old = self.P.shape[0]
        J = np.zeros((6, n_old))
        J[0:3, 0:3]  = np.eye(3)              # δp_C / δp_I
        J[0:3, 6:9]  = -R_I @ skew(T_IC)     # δp_C / δφ_I
        J[3:6, 6:9]  = np.eye(3)              # δφ_C / δφ_I  (R_IC = I)

        # Grow covariance
        n_new = n_old + 6
        P_new = np.zeros((n_new, n_new))
        P_new[:n_old, :n_old] = self.P
        P_new[:n_old, n_old:] = self.P @ J.T
        P_new[n_old:, :n_old] = J @ self.P
        P_new[n_old:, n_old:] = J @ self.P @ J.T
        self.P = P_new

        return cam_id

    # ── Marginalization ───────────────────────────────────────────────────────

    def marginalize_cam(self, idx: int):
        """
        Remove camera state at window index idx.
        Deletes corresponding rows/cols from P.
        """
        base = self.IMU_DIM + idx * self.CAM_DIM
        keep = list(range(base)) + list(range(base + self.CAM_DIM, self.n))
        self.P = self.P[np.ix_(keep, keep)]
        del self.cam_states[idx]

    # ── Error injection ───────────────────────────────────────────────────────

    def inject_error(self, dx: np.ndarray):
        """
        Apply error-state correction dx to nominal state.
        Additive for p, v, b_a, b_g.
        Right-multiplicative for orientations.
        """
        self.p   += dx[0:3]
        self.v   += dx[3:6]
        self.b_a += dx[9:12]
        self.b_g += dx[12:15]

        # IMU orientation (right perturbation)
        dR_I = exp_so3(dx[6:9])
        self.q = rot_to_quat(quat_to_rot(self.q) @ dR_I)
        self.q /= np.linalg.norm(self.q)

        # Camera orientations
        for i, cs in enumerate(self.cam_states):
            base = self.IMU_DIM + i * self.CAM_DIM
            cs['p'] += dx[base:base + 3]
            dR_C = exp_so3(dx[base + 3:base + 6])
            cs['q'] = rot_to_quat(quat_to_rot(cs['q']) @ dR_C)
            cs['q'] /= np.linalg.norm(cs['q'])


# ═══════════════════════════════════════════════════════════════════════════════
# MSCKF VIO Node
# ═══════════════════════════════════════════════════════════════════════════════

class MsckfVio(Node):

    def __init__(self):
        super().__init__('msckf_vio')

        self._state   = MsckfState()
        self._tracker: FeatureTracker | None = None   # created after CameraInfo

        # Intrinsics — populated from live /cam0/camera_info
        self._fx = self._fy = self._cx = self._cy = None
        self._img_w = self._img_h = None
        self._cam_info_received = False

        self._last_imu_t: float | None = None
        self._imu_count    = 0
        self._update_count = 0
        self._feat_count   = 0
        self._start        = time.time()

        # Static IMU initialisation: buffer readings while drone is stationary,
        # then estimate biases from mean.
        self._static_buf: list[tuple]  = []   # (am, wm)
        self._static_start_t: float | None = None   # first IMU sensor timestamp
        self._static_init_done = False
        self._static_init_dur  = 2.0         # seconds of static data to collect

        # Continuous-time IMU process noise (12-dim: g_noise, a_noise, g_bias, a_bias)
        self._Q_c = np.diag([
            SIGMA_G**2,  SIGMA_G**2,  SIGMA_G**2,
            SIGMA_A**2,  SIGMA_A**2,  SIGMA_A**2,
            SIGMA_GB**2, SIGMA_GB**2, SIGMA_GB**2,
            SIGMA_AB**2, SIGMA_AB**2, SIGMA_AB**2,
        ])

        # Precompute chi-squared thresholds for degrees of freedom 1..200
        self._chi2_lut = {df: chi2.ppf(0.95, df) for df in range(1, 201)}

        # ── Publishers ─────────────────────────────────────────────────────────
        self._pub_odom = self.create_publisher(Odometry, '/msckf_vio/odometry', 10)
        self._pub_path = self.create_publisher(Path,     '/msckf_vio/path',     10)
        self._path_msg = Path()
        self._path_msg.header.frame_id = 'world'

        # ── Subscribers ────────────────────────────────────────────────────────
        self.create_subscription(Imu,        '/imu0',            self._imu_cb,     50)
        self.create_subscription(CameraInfo, '/cam0/camera_info',self._cam_info_cb, 1)
        self.create_subscription(Image,      '/cam0/image_raw',  self._img_cb,     10)

        self.create_timer(5.0, self._status_cb)

        self.get_logger().info('=' * 60)
        self.get_logger().info('PHASE 3  Step 9: MSCKF Visual-Inertial Odometry')
        self.get_logger().info('=' * 60)
        self.get_logger().info(
            f'Window: {MAX_CAM_STATES}  Min track: {MIN_TRACK_LEN}  '
            f'Pixel σ: {PIXEL_STD} px  Max feats: {MAX_FEATURES}')
        self.get_logger().info('Waiting for /cam0/camera_info...')

    # ── CameraInfo callback ────────────────────────────────────────────────────

    def _cam_info_cb(self, msg: CameraInfo):
        if self._cam_info_received:
            return
        self._fx    = msg.k[0]
        self._fy    = msg.k[4]
        self._cx    = msg.k[2]
        self._cy    = msg.k[5]
        self._img_w = msg.width
        self._img_h = msg.height
        self._cam_info_received = True
        self._tracker = FeatureTracker(
            self._fx, self._fy, self._cx, self._cy,
            self._img_w, self._img_h,
        )
        self.get_logger().info(
            f'CameraInfo received: {msg.width}×{msg.height}  '
            f'fx={self._fx:.2f}  fy={self._fy:.2f}  '
            f'cx={self._cx:.2f}  cy={self._cy:.2f}'
        )

    # ── IMU propagation ───────────────────────────────────────────────────────

    def _imu_cb(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        am = np.array([msg.linear_acceleration.x,
                       msg.linear_acceleration.y,
                       msg.linear_acceleration.z])
        wm = np.array([msg.angular_velocity.x,
                       msg.angular_velocity.y,
                       msg.angular_velocity.z])

        # ── Static initialisation: collect IMU while drone is stationary ──────
        if not self._static_init_done:
            q0 = np.array([msg.orientation.x, msg.orientation.y,
                           msg.orientation.z, msg.orientation.w])
            if np.linalg.norm(q0) > 0.1 and self._state.q[3] == 1.0:
                self._state.q = q0 / np.linalg.norm(q0)

            self._static_buf.append((am.copy(), wm.copy()))

            if self._static_start_t is None:
                self._static_start_t = t

            # Use actual elapsed sensor time for the collection window
            elapsed = t - self._static_start_t
            if elapsed >= self._static_init_dur:
                # Enough data — compute biases from mean
                am_mean = np.mean([s[0] for s in self._static_buf], axis=0)
                wm_mean = np.mean([s[1] for s in self._static_buf], axis=0)

                R0 = self._state.rotation_matrix()
                # b_g: gyro reads bias when stationary
                self._state.b_g = wm_mean.copy()
                # b_a: accel reads specific force = R.T @ (-g_world) + b_a
                # g_world = [0,0,-9.81] → R.T @ g_world subtracted below
                # acc = R @ (am - b_a) - GRAVITY = 0  →  b_a = am - R.T @ GRAVITY
                self._state.b_a = am_mean - R0.T @ GRAVITY

                self._static_init_done = True
                self._state.is_init = True
                rpy = Rotation.from_quat(self._state.q).as_euler('xyz', degrees=True)
                self.get_logger().info(
                    f'Static init done ({len(self._static_buf)} samples over {elapsed:.2f}s)  '
                    f'roll={rpy[0]:.1f}°  pitch={rpy[1]:.1f}°  yaw={rpy[2]:.1f}°  '
                    f'b_a=[{self._state.b_a[0]:.3f},{self._state.b_a[1]:.3f},{self._state.b_a[2]:.3f}]  '
                    f'b_g=[{self._state.b_g[0]:.4f},{self._state.b_g[1]:.4f},{self._state.b_g[2]:.4f}]')

            self._last_imu_t = t
            return

        if self._last_imu_t is None:
            self._last_imu_t = t
            return

        dt = t - self._last_imu_t
        if dt <= 0 or dt > 0.1:
            self._last_imu_t = t
            return
        self._last_imu_t = t
        self._imu_count += 1

        self._propagate(am, wm, dt)

    def _propagate(self, am: np.ndarray, wm: np.ndarray, dt: float):
        """
        Propagate IMU state + covariance.

        Nominal dynamics:
          ṗ = v
          v̇ = R*(a_m - b_a) - g
          Ṙ = R * [ω_m - b_g]×
          ḃ_a = 0,  ḃ_g = 0

        Error-state transition (first-order discretisation):
          F = d(ẋ)/d(δx)  →  Φ ≈ I + F*dt

        Process noise (first-order discretisation):
          Q_d = G @ Q_c @ G.T * dt
          (see "PROCESS NOISE DISCRETISATION" in module docstring)
        """
        s  = self._state
        R  = s.rotation_matrix()
        ab = am - s.b_a
        wb = wm - s.b_g

        # Nominal propagation
        acc = R @ ab - GRAVITY
        s.p  = s.p + s.v * dt + 0.5 * acc * dt**2
        s.v  = s.v + acc * dt
        R_new = R @ exp_so3(wb * dt)
        s.q  = rot_to_quat(R_new)
        s.q /= np.linalg.norm(s.q)

        # Error-state Jacobian F (15×15)
        F = np.zeros((15, 15))
        F[0:3,  3:6]  = np.eye(3)
        F[3:6,  6:9]  = -R @ skew(ab)
        F[3:6,  9:12] = -R
        F[6:9,  6:9]  = -skew(wb)
        F[6:9, 12:15] = -np.eye(3)

        Phi = np.eye(15) + F * dt

        # Process noise injection G (15×12)
        G = np.zeros((15, 12))
        G[3:6,  3:6]  = -R          # accel noise → velocity
        G[6:9,  0:3]  = -np.eye(3)  # gyro noise  → orientation
        G[9:12, 6:9]  = np.eye(3)   # accel bias walk
        G[12:15, 9:12]= np.eye(3)   # gyro bias walk

        # Discrete noise: Q_d = G @ Q_c @ G.T * dt
        Q_d = G @ self._Q_c @ G.T * dt

        # Propagate full covariance (IMU + camera states)
        n   = s.n
        n_I = s.IMU_DIM
        n_C = n - n_I

        P = s.P
        P_new = np.empty_like(P)
        P_new[:n_I, :n_I] = Phi @ P[:n_I, :n_I] @ Phi.T + Q_d
        if n_C > 0:
            # IMU-camera cross blocks: only IMU rows change
            P_new[:n_I, n_I:] = Phi @ P[:n_I, n_I:]
            P_new[n_I:, :n_I] = P[n_I:, :n_I] @ Phi.T
            P_new[n_I:, n_I:] = P[n_I:, n_I:]   # camera-camera block unchanged
        s.P = P_new

    # ── Image callback (augment → track → update → marginalize) ──────────────

    def _img_cb(self, msg: Image):
        if not self._state.is_init or not self._cam_info_received:
            return

        # Decode image to grayscale — branch on encoding explicitly
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        enc = msg.encoding.lower()
        try:
            if enc in ('mono8', '8uc1'):
                gray = arr.reshape(msg.height, msg.width)
            elif enc in ('bgr8', 'rgb8', 'bgra8', 'rgba8'):
                channels = 4 if enc.endswith('a8') else 3
                img = arr.reshape(msg.height, msg.width, channels)
                code = cv2.COLOR_BGRA2GRAY if enc.startswith('bgra') else \
                       cv2.COLOR_RGBA2GRAY if enc.startswith('rgba') else \
                       cv2.COLOR_RGB2GRAY  if enc.startswith('rgb')  else \
                       cv2.COLOR_BGR2GRAY
                gray = cv2.cvtColor(img, code)
            else:
                # Fallback: assume 3-channel BGR
                img = arr.reshape(msg.height, msg.width, -1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().warn(f'Image decode failed ({enc}): {e}')
            return

        # 1. Augment state with current camera pose
        cam_id = self._state.augment()

        # 2. Track features; get IDs of features no longer visible
        lost_ids = self._tracker.process(gray, cam_id)

        # 3. Build camera-ID → window-index lookup
        cam_id_to_idx = {cs['id']: i for i, cs in enumerate(self._state.cam_states)}

        # 4. Process each lost feature → compute H_o, r_o
        H_list, r_list = [], []

        for fid in lost_ids:
            track = self._tracker.tracks.pop(fid, [])

            # Keep only observations whose camera state is still in the window
            valid = [(obs['cam_id'], obs['u'], obs['v'])
                     for obs in track if obs['cam_id'] in cam_id_to_idx]

            if len(valid) < MIN_TRACK_LEN:
                continue

            cids   = [v[0] for v in valid]
            obs_uv = [(v[1], v[2]) for v in valid]
            cam_ps = [(self._state.cam_states[cam_id_to_idx[cid]]['p'],
                       quat_to_rot(self._state.cam_states[cam_id_to_idx[cid]]['q']))
                      for cid in cids]

            # 4a. Triangulate feature position
            p_f = triangulate_dlt(obs_uv, cam_ps)
            if p_f is None:
                continue

            # 4b. Verify: feature must be in front of all cameras
            depths = [float((R_C.T @ (p_f - p_C))[2])
                      for p_C, R_C in cam_ps]
            if min(depths) < 0.1:
                continue

            # 4c. Build residuals and Jacobians
            #
            # For camera state i observing feature j:
            #   z_c = R_C.T @ (p_f - p_C)          — feature in camera frame
            #   predicted: [X/Z, Y/Z]
            #
            #   Jacobian of measurement w.r.t. z_c (J_proj, 2×3):
            #     [[1/Z, 0,  -X/Z²],
            #      [0,   1/Z,-Y/Z²]]
            #
            #   Right-perturbation: R_C → R_C @ exp(skew(δφ_C))
            #   δz_c / δφ_C = skew(z_c)           (see module docstring)
            #   δz_c / δp_C = -R_C.T               (world→camera rotation)
            #   δz_c / δp_f =  R_C.T
            #
            #   H_cam_i (2×6): J_proj @ [-R_C.T | skew(z_c)]  ([δp_C | δφ_C])
            #   H_f_i   (2×3): J_proj @ R_C.T

            N_obs = len(valid)
            H_f   = np.zeros((2 * N_obs, 3))
            H_X   = np.zeros((2 * N_obs, self._state.n))
            res   = np.zeros(2 * N_obs)

            ok = True
            for k, (cid, (u_n, v_n), (p_C, R_C)) in enumerate(
                    zip(cids, obs_uv, cam_ps)):

                z_c = R_C.T @ (p_f - p_C)   # feature in camera frame
                X, Y, Z = z_c
                if Z < 0.01:
                    ok = False
                    break

                u_hat, v_hat = X / Z, Y / Z
                res[2*k]   = u_n - u_hat
                res[2*k+1] = v_n - v_hat

                # 2×3 projection Jacobian
                J_p = np.array([[1/Z, 0,   -X/Z**2],
                                 [0,   1/Z, -Y/Z**2]])

                # Feature Jacobian (world frame)
                H_f[2*k:2*k+2, :] = J_p @ R_C.T   # R_C.T = world→camera

                # Camera state Jacobian
                win_idx = cam_id_to_idx[cid]
                base    = self._state.IMU_DIM + win_idx * self._state.CAM_DIM

                # δz_c/δp_C = -R_C.T  (camera origin moves → z_c changes)
                H_X[2*k:2*k+2, base:base+3]   = J_p @ (-R_C.T)
                # δz_c/δφ_C = skew(z_c)  (camera rotates → z_c rotates)
                H_X[2*k:2*k+2, base+3:base+6] = J_p @ skew(z_c)

            if not ok:
                continue

            # 4d. Null-space projection — eliminate feature position
            #
            # QR of H_f (2N × 3):
            #   H_f = Q @ [R_tri; 0]    Q is 2N×2N orthogonal
            #   Q2 = Q[:, 3:]           left null space of H_f (2N × 2N-3)
            #   Q2.T @ H_f = 0
            #
            # Projected:
            #   H_o = Q2.T @ H_X    (2N-3 × state_dim)
            #   r_o = Q2.T @ res    (2N-3,)
            if 2 * N_obs <= 3:
                continue

            Q_full, _ = np.linalg.qr(H_f, mode='complete')
            Q2 = Q_full[:, 3:]          # 2N × (2N-3)

            H_o = Q2.T @ H_X           # (2N-3 × n)
            r_o = Q2.T @ res           # (2N-3,)

            # 4e. Mahalanobis chi-squared outlier test
            R_o = PIXEL_STD**2 * np.eye(H_o.shape[0])
            S   = H_o @ self._state.P @ H_o.T + R_o
            try:
                gamma = float(r_o @ np.linalg.solve(S, r_o))
            except np.linalg.LinAlgError:
                continue

            df        = H_o.shape[0]
            threshold = self._chi2_lut.get(df, self._chi2_lut[200])
            if gamma > threshold:
                continue   # outlier — discard

            H_list.append(H_o)
            r_list.append(r_o)
            self._feat_count += 1

        # 5. Batch EKF update
        if H_list:
            self._ekf_update(H_list, r_list)

        # 6. Marginalize oldest camera state when window is full
        while len(self._state.cam_states) > MAX_CAM_STATES:
            self._state.marginalize_cam(0)

        self._publish(msg.header.stamp)

    def _ekf_update(self, H_list: list, r_list: list):
        """
        Stack all feature constraints and apply one EKF update.

        K  = P @ H.T @ inv(H @ P @ H.T + R)
        dx = K @ r
        P  = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T  [Joseph form]
        """
        H = np.vstack(H_list)
        r = np.concatenate(r_list)
        R = PIXEL_STD**2 * np.eye(H.shape[0])

        P = self._state.P
        S = H @ P @ H.T + R
        try:
            K = np.linalg.solve(S.T, (P @ H.T).T).T
        except np.linalg.LinAlgError:
            return

        dx = K @ r
        self._state.inject_error(dx)

        I_KH = np.eye(P.shape[0]) - K @ H
        P_new = I_KH @ P @ I_KH.T + K @ R @ K.T
        self._state.P = 0.5 * (P_new + P_new.T)   # enforce symmetry

        self._update_count += 1
        self.get_logger().debug(
            f'Update #{self._update_count}  '
            f'feats={len(H_list)}  |r|={np.linalg.norm(r):.3f}')

    # ── Publishers ────────────────────────────────────────────────────────────

    def _publish(self, stamp):
        s = self._state
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = 'world'
        odom.child_frame_id  = 'imu'
        odom.pose.pose.position.x = float(s.p[0])
        odom.pose.pose.position.y = float(s.p[1])
        odom.pose.pose.position.z = float(s.p[2])
        odom.pose.pose.orientation.x = float(s.q[0])
        odom.pose.pose.orientation.y = float(s.q[1])
        odom.pose.pose.orientation.z = float(s.q[2])
        odom.pose.pose.orientation.w = float(s.q[3])

        # Pose covariance (6×6 row-major): [x,y,z, rot_x,rot_y,rot_z]
        # Map from EKF state indices [p(0:3), phi(6:9)]
        cov6 = np.zeros(36)
        pose_idx = [0, 1, 2, 6, 7, 8]
        for i, src_i in enumerate(pose_idx):
            for j, src_j in enumerate(pose_idx):
                cov6[i * 6 + j] = s.P[src_i, src_j]
        odom.pose.covariance = list(cov6)

        self._pub_odom.publish(odom)

        pose = PoseStamped()
        pose.header = odom.header
        pose.pose   = odom.pose.pose
        self._path_msg.header.stamp = stamp
        self._path_msg.poses.append(pose)
        if len(self._path_msg.poses) > 2000:   # cap memory
            self._path_msg.poses.pop(0)
        self._pub_path.publish(self._path_msg)

    def _status_cb(self):
        s       = self._state
        elapsed = time.time() - self._start
        cov_p   = float(np.trace(s.P[:3, :3]))
        cam_info = 'yes' if self._cam_info_received else 'no'
        self.get_logger().info(
            f'[{elapsed:.0f}s]  '
            f'cam_info: {cam_info}  '
            f'Updates: {self._update_count}  '
            f'Feats used: {self._feat_count}  '
            f'Cam states: {len(s.cam_states)}  '
            f'p=[{s.p[0]:.2f},{s.p[1]:.2f},{s.p[2]:.2f}]  '
            f'cov_p={cov_p:.4f}  '
            f'|b_a|={np.linalg.norm(s.b_a):.3f}  '
            f'|b_g|={np.linalg.norm(s.b_g):.4f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = MsckfVio()
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
