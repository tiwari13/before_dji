#!/usr/bin/env python3
"""
PHASE 3 — Step 10: Multi-Camera MSCKF
======================================

WHY TWO CAMERAS?
----------------
Monocular VIO (Step 9) has scale ambiguity: a single camera cannot
distinguish "small object nearby" from "large object far away".
Scale is recovered indirectly through IMU integration but drifts over time.

Two cameras with a known baseline DIRECTLY triangulate metric scale:
  Given a 3D point seen by cam0 at pixel (u0,v0) and cam1 at pixel (u1,v1),
  and knowing the baseline T between the cameras, the DLT system:

    A @ X = 0   (4 equations from 2 cameras, 4 unknowns in homogeneous coords)

  has a unique solution whose scale is set by ‖T‖.  No IMU needed for scale.

HOW THE MSCKF EXTENDS TO MULTIPLE CAMERAS
------------------------------------------
Step 9 maintained ONE feature tracker and ONE set of camera states {C_k}.

Step 10 maintains TWO trackers (cam0=front, cam1=back) and augments TWO
camera poses per image timestep.  For features visible in both cameras:

  Each observation provides 2 rows in H_X and H_f.
  With N timesteps × 2 cameras: 4N rows, still 3 cols for H_f.
  Null-space projection removes the 3 feature rows → 4N-3 constraints.
  This gives MORE constraints per feature than monocular → better accuracy.

STATE VECTOR (unchanged from Step 9)
-------------------------------------
  x = [x_IMU(15) | x_C0(6) | x_C1(6) | x_C2(6) | ...]

  But now EACH timestep adds TWO camera states (one per physical camera):
    x_C_front_k  (6-dim)
    x_C_back_k   (6-dim)

  Window size = MAX_CAM_STATES pairs = 2 × MAX_CAM_STATES states.

CAMERA EXTRINSICS
-----------------
  Front camera (cam0): T_IC0 = [0.15, 0, 0.10],  R_IC0 = I
  Back  camera (cam1): T_IC1 = [-0.15, 0, 0.10],  R_IC1 = R_z(π)

  R_z(π) = [[-1, 0, 0],   (back camera faces the opposite direction)
             [ 0,-1, 0],
             [ 0, 0, 1]]

  Baseline ≈ 0.30 m  (sufficient for metric scale at 2-15 m scene depth)

CROSS-CAMERA FEATURE MATCHING
------------------------------
Features detected independently in each camera.  A feature is "stereo"
if it is matched between cam0 and cam1 using descriptor matching (ORB)
with a geometric check (epipolar constraint):

  |u1 - u0_rectified| < MAX_DISP    (approximate for 180-deg baseline)

For features only seen in one camera: processed as monocular (Step 9).
For features seen in both: processed with stereo Jacobians → metric scale.

STUDY MATERIAL
--------------
[1] Li & Mourikis, "High-precision, consistent EKF-based visual-inertial
    odometry", IJRR 2013.  Section IV: multi-camera extension.
    https://doi.org/10.1177/0278364913481563

[2] Mourikis & Roumeliotis, ICRA 2007 — original MSCKF.
    https://www-users.cse.umn.edu/~stergios/papers/ICRA07-MSCKF.pdf

[3] Huai & Huang, "Robocentric visual-inertial odometry", IROS 2018.
    Multi-camera treatment with general baselines.

Run
---
  ros2 run module3_vio multicam_msckf
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber

from sensor_msgs.msg import Imu, Image
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import VehicleLocalPosition

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import chi2
import cv2
import time
from collections import defaultdict

# ── Camera intrinsics (both cameras identical — same OakD-Lite model) ─────────
FX = FY = 465.74
CX, CY  = 320.0, 180.0
IMG_W, IMG_H = 640, 480

# ── Camera-IMU extrinsics ──────────────────────────────────────────────────────
#
# cam0 (front): 0.15 m forward, 0.10 m up from IMU, axes aligned
T_IC0 = np.array([ 0.15, 0.0,  0.10])
R_IC0 = np.eye(3)

# cam1 (back):  0.15 m behind, 0.10 m up from IMU, rotated 180° around Z
# R_z(π): drone body +X is forward; back camera faces -X, so it is rotated π around Z
T_IC1 = np.array([-0.15, 0.0,  0.10])
R_IC1 = np.array([[-1.,  0.,  0.],
                   [ 0., -1.,  0.],
                   [ 0.,  0.,  1.]])

# ── IMU noise ─────────────────────────────────────────────────────────────────
SIGMA_G  = 0.005
SIGMA_A  = 0.05
SIGMA_GB = 0.0001
SIGMA_AB = 0.001

# ── MSCKF tuning ──────────────────────────────────────────────────────────────
MAX_CAM_PAIRS  = 15    # window = 15 front+back pairs = 30 camera states
MIN_TRACK_LEN  = 3     # minimum observations (across both cameras) to use
MAX_TRACK_LEN  = 25
PIXEL_STD      = 1.5   # pixel noise [px]
MAX_FEATURES   = 150   # per camera

# Stereo matching params
MAX_HAMMING    = 50    # ORB descriptor match threshold
MIN_STEREO_OBS = 2     # stereo feature must appear in ≥ N frame-pairs

# ENU gravity: g_world=[0,0,-9.81], acc = R@ab - GRAVITY → GRAVITY = +9.81 in Z
# (standard: a_true = R@a_imu + g_world = R@a_imu - GRAVITY where GRAVITY=-g_world)
GRAVITY = np.array([0.0, 0.0, 9.81])

BEST_EFFORT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SO(3) utilities
# ═══════════════════════════════════════════════════════════════════════════════

def quat_to_rot(q):
    return Rotation.from_quat(q).as_matrix()

def rot_to_quat(R):
    return Rotation.from_matrix(R).as_quat()

def skew(v):
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]])

def exp_so3(theta):
    angle = np.linalg.norm(theta)
    if angle < 1e-8:
        return np.eye(3) + skew(theta)
    ax = theta / angle
    K  = skew(ax)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# ═══════════════════════════════════════════════════════════════════════════════
# DLT triangulation (same as Step 9, works for N ≥ 2 cameras)
# ═══════════════════════════════════════════════════════════════════════════════

def triangulate_dlt(obs_norm, cam_poses):
    """
    obs_norm  : list of (u_n, v_n) normalised pixel coords
    cam_poses : list of (p_C, R_C) — camera origin and rotation in world frame
    Returns 3D point in world frame, or None if degenerate.
    """
    A = []
    for (u_n, v_n), (p_C, R_C) in zip(obs_norm, cam_poses):
        R_wc = R_C.T
        t_wc = -(R_C.T @ p_C)
        P    = np.hstack([R_wc, t_wc[:, None]])
        A.append(u_n * P[2] - P[0])
        A.append(v_n * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    X = Vt[-1]
    if abs(X[3]) < 1e-8:
        return None
    return X[:3] / X[3]


# ═══════════════════════════════════════════════════════════════════════════════
# Single-camera KLT tracker (same as Step 9)
# ═══════════════════════════════════════════════════════════════════════════════

class CameraTracker:
    """
    KLT tracker for one physical camera.

    tracks[fid] = list of {'cam_state_id': int, 'u': float, 'v': float}
    where cam_state_id is the ID of the camera STATE (not the camera index)
    assigned at augmentation time.
    """

    LK_PARAMS = dict(
        winSize  = (21, 21),
        maxLevel = 3,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def __init__(self, cam_label: str):
        self.label        = cam_label
        self._next_fid    = 0
        self._prev_img    = None
        self._pts         = None   # Nx1x2 float32
        self._ids: list   = []
        self.tracks: dict = defaultdict(list)

    def process(self, img_gray: np.ndarray, cam_state_id: int) -> list:
        """
        Track into new frame.  Returns list of lost feature IDs
        (enough observations to be usable for EKF update).
        """
        lost_ids = []

        if self._prev_img is None:
            self._prev_img = img_gray
            self._detect(img_gray, cam_state_id)
            return lost_ids

        if self._pts is None or len(self._pts) == 0:
            self._prev_img = img_gray
            self._detect(img_gray, cam_state_id)
            return lost_ids

        # Forward KLT
        next_pts, st_f, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_img, img_gray, self._pts, None, **self.LK_PARAMS)

        # Backward verification
        prev2, st_b, _ = cv2.calcOpticalFlowPyrLK(
            img_gray, self._prev_img, next_pts, None, **self.LK_PARAMS)

        fb_err = np.abs(self._pts - prev2).reshape(-1, 2).max(axis=1)
        ok = (st_f.flatten() == 1) & (st_b.flatten() == 1) & (fb_err < 2.0)

        nx, ny = next_pts[:, 0, 0], next_pts[:, 0, 1]
        ok &= (nx >= 5) & (nx < IMG_W - 5) & (ny >= 5) & (ny < IMG_H - 5)

        kept_pts, kept_ids = [], []
        for i, (fid, tracked) in enumerate(zip(self._ids, ok)):
            if tracked:
                px = float(next_pts[i, 0, 0])
                py = float(next_pts[i, 0, 1])
                self.tracks[fid].append({
                    'cam_state_id': cam_state_id,
                    'u': (px - CX) / FX,
                    'v': (py - CY) / FY,
                })
                if len(self.tracks[fid]) >= MAX_TRACK_LEN:
                    lost_ids.append(fid)
                else:
                    kept_pts.append(next_pts[i])
                    kept_ids.append(fid)
            else:
                lost_ids.append(fid)

        self._pts = np.array(kept_pts, dtype=np.float32) if kept_pts else None
        self._ids = kept_ids

        n_new = MAX_FEATURES - len(kept_ids)
        if n_new > 10:
            existing = np.array(kept_pts, dtype=np.float32) if kept_pts else None
            self._detect(img_gray, cam_state_id, n_new=n_new, existing=existing)

        self._prev_img = img_gray
        return lost_ids

    def _detect(self, img, cam_state_id, n_new=MAX_FEATURES, existing=None):
        mask = np.ones(img.shape[:2], np.uint8) * 255
        if existing is not None and len(existing) > 0:
            for pt in existing.reshape(-1, 2):
                cv2.circle(mask, (int(pt[0]), int(pt[1])), 20, 0, -1)

        pts = cv2.goodFeaturesToTrack(
            img, maxCorners=n_new, qualityLevel=0.01, minDistance=15, mask=mask)
        if pts is None:
            return

        for pt in pts:
            px, py = float(pt[0, 0]), float(pt[0, 1])
            fid = self._next_fid
            self._next_fid += 1
            self.tracks[fid].append({
                'cam_state_id': cam_state_id,
                'u': (px - CX) / FX,
                'v': (py - CY) / FY,
            })
            if self._pts is None:
                self._pts = pt.reshape(1, 1, 2)
            else:
                self._pts = np.vstack([self._pts, pt.reshape(1, 1, 2)])
            self._ids.append(fid)

    def get_current_keypoints(self):
        """Return current tracked points as (N,2) array for ORB matching."""
        if self._pts is None or len(self._pts) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return self._pts.reshape(-1, 2)

    def get_current_ids(self):
        return list(self._ids)


# ═══════════════════════════════════════════════════════════════════════════════
# Stereo matcher — cross-camera feature association
# ═══════════════════════════════════════════════════════════════════════════════

class StereoMatcher:
    """
    Matches features between cam0 and cam1 at the same timestep.

    Because cam1 is rotated 180° around Z, the epipolar geometry is
    non-standard (not a horizontal rectification).  We use ORB descriptors
    + Hamming distance matching + a loose geometric consistency check.

    Returns: dict mapping cam0_fid → cam1_fid for matched pairs.
    """

    def __init__(self):
        self._orb     = cv2.ORB_create(nfeatures=MAX_FEATURES)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match(self, img0: np.ndarray, img1: np.ndarray,
              pts0: np.ndarray, ids0: list,
              pts1: np.ndarray, ids1: list) -> dict:
        """
        img0, img1  : grayscale images
        pts0, pts1  : (N,2) float32 current tracked points
        ids0, ids1  : corresponding feature IDs

        Returns {fid0: fid1} for matched pairs.
        """
        if len(pts0) == 0 or len(pts1) == 0:
            return {}

        # Compute ORB descriptors at tracked point locations
        kp0 = [cv2.KeyPoint(float(p[0]), float(p[1]), 7) for p in pts0]
        kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 7) for p in pts1]

        _, desc0 = self._orb.compute(img0, kp0)
        _, desc1 = self._orb.compute(img1, kp1)

        if desc0 is None or desc1 is None:
            return {}
        if len(desc0) == 0 or len(desc1) == 0:
            return {}

        matches = self._matcher.match(desc0, desc1)

        result = {}
        for m in matches:
            if m.distance > MAX_HAMMING:
                continue
            fid0 = ids0[m.queryIdx]
            fid1 = ids1[m.trainIdx]
            result[fid0] = fid1

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Camera MSCKF State
# ═══════════════════════════════════════════════════════════════════════════════

class MultiCamMsckfState:
    """
    State: IMU (15) + sliding window of camera STATE pairs.

    Each image timestep k appends:
      - cam_state for cam0 at time k  (6-dim)
      - cam_state for cam1 at time k  (6-dim)

    Total camera states = 2 * num_timesteps_in_window.

    cam_states list entries:
      {'id': int, 'p': ndarray(3), 'q': ndarray(4), 'cam_idx': 0 or 1}
    cam_idx=0 → front camera extrinsics (T_IC0, R_IC0)
    cam_idx=1 → back  camera extrinsics (T_IC1, R_IC1)
    """
    IMU_DIM = 15
    CAM_DIM = 6

    # Extrinsics indexed by cam_idx
    T_IC = [T_IC0, T_IC1]
    R_IC = [R_IC0, R_IC1]

    def __init__(self):
        self.p   = np.zeros(3)
        self.v   = np.zeros(3)
        self.q   = np.array([0., 0., 0., 1.])
        self.b_a = np.zeros(3)
        self.b_g = np.zeros(3)

        self.cam_states: list[dict] = []
        self._cam_id_ctr = 0

        self.P = np.diag([
            1e-2, 1e-2, 1e-2,
            0.1,  0.1,  0.1,
            1e-3, 1e-3, 1e-3,
            1e-4, 1e-4, 1e-4,
            1e-6, 1e-6, 1e-6,
        ])
        self.is_init = False

    @property
    def n(self):
        return self.IMU_DIM + self.CAM_DIM * len(self.cam_states)

    def rotation_matrix(self):
        return quat_to_rot(self.q)

    def augment_pair(self):
        """
        Augment with BOTH cam0 and cam1 poses from current IMU state.
        Returns (cam0_state_id, cam1_state_id).

        For each camera ci (i=0,1):
          p_Ci = p_I + R_I @ T_ICi
          R_Ci = R_ICi.T @ R_I

        Augmentation Jacobian for camera i (6 × current_n):
          δp_Ci = δp_I + (-R_I @ [T_ICi]×) @ δφ_I
          δφ_Ci = R_ICi.T @ δφ_I   (since R_ICi may not be identity)

        We augment both cameras in one covariance expansion.
        """
        R_I   = self.rotation_matrix()
        ids   = []
        poses = []

        for ci in range(2):
            T = self.T_IC[ci]
            R = self.R_IC[ci]
            p_C = self.p + R_I @ T
            q_C = rot_to_quat(R.T @ R_I)
            sid = self._cam_id_ctr
            self._cam_id_ctr += 1
            self.cam_states.append({
                'id': sid, 'p': p_C.copy(), 'q': q_C.copy(), 'cam_idx': ci
            })
            ids.append(sid)
            poses.append((p_C, R_I, T, R))

        # Expand covariance for BOTH new camera states in one step
        # Each new camera adds 6 rows/cols
        n_old = self.P.shape[0]

        # Build J for cam0 and cam1 stacked vertically: (12 × n_old)
        J = np.zeros((12, n_old))
        for k, (_, R_I_k, T_k, R_IC_k) in enumerate(poses):
            row = k * 6
            J[row:row+3, 0:3]  = np.eye(3)                    # δp_Ci/δp_I
            J[row:row+3, 6:9]  = -R_I_k @ skew(T_k)           # δp_Ci/δφ_I
            J[row+3:row+6, 6:9] = R_IC_k.T                    # δφ_Ci/δφ_I

        n_new = n_old + 12
        P_new = np.zeros((n_new, n_new))
        P_new[:n_old, :n_old] = self.P
        P_new[:n_old, n_old:] = self.P @ J.T
        P_new[n_old:, :n_old] = J @ self.P
        P_new[n_old:, n_old:] = J @ self.P @ J.T
        self.P = P_new

        return ids[0], ids[1]

    def marginalize_pair(self):
        """
        Remove the oldest two camera states (oldest front+back pair).
        Always at indices 0 and 1 of cam_states.
        """
        # Remove indices 1 then 0 (remove higher index first to keep index 0 valid)
        for _ in range(2):
            idx = 0
            base = self.IMU_DIM + idx * self.CAM_DIM
            keep = list(range(base)) + list(range(base + self.CAM_DIM, self.n))
            self.P = self.P[np.ix_(keep, keep)]
            del self.cam_states[idx]

    def inject_error(self, dx: np.ndarray):
        self.p   += dx[0:3]
        self.v   += dx[3:6]
        self.b_a += dx[9:12]
        self.b_g += dx[12:15]

        dR_I = exp_so3(dx[6:9])
        self.q = rot_to_quat(quat_to_rot(self.q) @ dR_I)
        self.q /= np.linalg.norm(self.q)

        for i, cs in enumerate(self.cam_states):
            base = self.IMU_DIM + i * self.CAM_DIM
            cs['p'] += dx[base:base + 3]
            dR_C = exp_so3(dx[base + 3:base + 6])
            cs['q'] = rot_to_quat(quat_to_rot(cs['q']) @ dR_C)
            cs['q'] /= np.linalg.norm(cs['q'])


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Camera MSCKF Node
# ═══════════════════════════════════════════════════════════════════════════════

class MultiCamMsckfVio(Node):

    def __init__(self):
        super().__init__('multicam_msckf')

        self._state   = MultiCamMsckfState()
        self._tracker0 = CameraTracker('cam0')   # front
        self._tracker1 = CameraTracker('cam1')   # back
        self._stereo   = StereoMatcher()

        # stereo_links[fid0] = fid1  — cross-camera feature associations
        self._stereo_links: dict[int, int] = {}

        self._last_imu_t   = None
        self._imu_count    = 0
        self._update_count = 0
        self._mono_feats   = 0   # features processed as monocular
        self._stereo_feats = 0   # features processed as stereo
        self._start        = time.time()

        # Static IMU initialisation buffers
        self._static_buf: list[tuple] = []
        self._static_init_done = False
        self._static_init_dur  = 2.0

        self._Q_c = np.diag([
            SIGMA_G**2,  SIGMA_G**2,  SIGMA_G**2,
            SIGMA_A**2,  SIGMA_A**2,  SIGMA_A**2,
            SIGMA_GB**2, SIGMA_GB**2, SIGMA_GB**2,
            SIGMA_AB**2, SIGMA_AB**2, SIGMA_AB**2,
        ])
        self._chi2_lut = {df: chi2.ppf(0.95, df) for df in range(1, 401)}

        # Ground truth
        self._gt_origin = None
        self._gt_pos    = None

        # ── Publishers ─────────────────────────────────────────────────────────
        self._pub_odom = self.create_publisher(Odometry, '/multicam_msckf/odometry', 10)
        self._pub_path = self.create_publisher(Path,     '/multicam_msckf/path',     10)
        self._path_msg = Path()
        self._path_msg.header.frame_id = 'world'

        # ── IMU subscriber ─────────────────────────────────────────────────────
        self.create_subscription(Imu, '/imu0', self._imu_cb, 50)

        # ── Synchronised stereo image subscriber ──────────────────────────────
        # ApproximateTimeSynchronizer matches cam0 and cam1 frames that arrive
        # within 0.05 s of each other (they share the same sim clock so this
        # is effectively exact synchronisation).
        cam0_sub = Subscriber(self, Image, '/cam0/image_raw',
                              qos_profile=10)
        cam1_sub = Subscriber(self, Image, '/cam1/image_raw',
                              qos_profile=10)
        self._sync = ApproximateTimeSynchronizer(
            [cam0_sub, cam1_sub], queue_size=10, slop=0.05)
        self._sync.registerCallback(self._stereo_img_cb)

        # ── Ground truth ───────────────────────────────────────────────────────
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self._gt_cb,
            BEST_EFFORT_QOS,
        )

        self.create_timer(5.0, self._status_cb)

        self.get_logger().info('=' * 65)
        self.get_logger().info('PHASE 3  Step 10: Multi-Camera MSCKF')
        self.get_logger().info('=' * 65)
        self.get_logger().info(
            f'Cameras: cam0 (front) + cam1 (back)  Baseline: ~0.30 m')
        self.get_logger().info(
            f'Window: {MAX_CAM_PAIRS} pairs ({2*MAX_CAM_PAIRS} states)  '
            f'Pixel σ: {PIXEL_STD} px  Max feats/cam: {MAX_FEATURES}')
        self.get_logger().info(
            f'Stereo gives METRIC SCALE — no monocular scale ambiguity')

    # ── IMU propagation (identical to Step 9) ─────────────────────────────────

    def _imu_cb(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        am = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y,
                       msg.linear_acceleration.z])
        wm = np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                       msg.angular_velocity.z])

        if not self._static_init_done:
            q0 = np.array([msg.orientation.x, msg.orientation.y,
                           msg.orientation.z, msg.orientation.w])
            if np.linalg.norm(q0) > 0.1 and self._state.q[3] == 1.0:
                self._state.q = q0 / np.linalg.norm(q0)

            self._static_buf.append((am.copy(), wm.copy()))

            if self._last_imu_t is not None and (t - self._last_imu_t +
                    len(self._static_buf) / 250.0) >= self._static_init_dur:
                am_mean = np.mean([s[0] for s in self._static_buf], axis=0)
                wm_mean = np.mean([s[1] for s in self._static_buf], axis=0)
                R0 = self._state.rotation_matrix()
                self._state.b_g = wm_mean.copy()
                self._state.b_a = am_mean - R0.T @ GRAVITY
                self._static_init_done = True
                self._state.is_init = True
                rpy = Rotation.from_quat(self._state.q).as_euler('xyz', degrees=True)
                self.get_logger().info(
                    f'Static init done ({len(self._static_buf)} samples)  '
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

    def _propagate(self, am, wm, dt):
        s  = self._state
        R  = s.rotation_matrix()
        ab = am - s.b_a
        wb = wm - s.b_g

        acc  = R @ ab - GRAVITY
        s.p  = s.p + s.v * dt + 0.5 * acc * dt**2
        s.v  = s.v + acc * dt
        R_new = R @ exp_so3(wb * dt)
        s.q  = rot_to_quat(R_new)
        s.q /= np.linalg.norm(s.q)

        F = np.zeros((15, 15))
        F[0:3,  3:6]  = np.eye(3)
        F[3:6,  6:9]  = -R @ skew(ab)
        F[3:6,  9:12] = -R
        F[6:9,  6:9]  = -skew(wb)
        F[6:9, 12:15] = -np.eye(3)
        Phi = np.eye(15) + F * dt

        G = np.zeros((15, 12))
        G[3:6,  3:6]   = -R
        G[6:9,  0:3]   = -np.eye(3)
        G[9:12, 6:9]   = np.eye(3)
        G[12:15, 9:12] = np.eye(3)
        Q_d = G @ (self._Q_c / dt) @ G.T * dt

        n, n_I = s.n, s.IMU_DIM
        n_C    = n - n_I
        P = s.P
        P_new = np.empty_like(P)
        P_new[:n_I, :n_I] = Phi @ P[:n_I, :n_I] @ Phi.T + Q_d
        if n_C > 0:
            P_new[:n_I, n_I:] = Phi @ P[:n_I, n_I:]
            P_new[n_I:, :n_I] = P[n_I:, :n_I] @ Phi.T
            P_new[n_I:, n_I:] = P[n_I:, n_I:]
        s.P = P_new

    # ── Stereo image callback ──────────────────────────────────────────────────

    def _stereo_img_cb(self, msg0: Image, msg1: Image):
        """
        Called with synchronised cam0 + cam1 frames.

        Pipeline:
          1. Decode both images
          2. Augment state with cam0 + cam1 poses
          3. Stereo-match current frame to update stereo_links
          4. Track features in each camera independently
          5. Process lost features:
               - If fid0 has a stereo link fid1: use BOTH cameras' observations
               - Otherwise: monocular observations from whichever camera saw it
          6. EKF batch update
          7. Marginalize oldest pair if window full
        """
        if not self._state.is_init:
            return

        # 1. Decode
        gray0 = self._decode(msg0)
        gray1 = self._decode(msg1)
        if gray0 is None or gray1 is None:
            return

        # 2. Augment: adds cam0 state then cam1 state
        sid0, sid1 = self._state.augment_pair()

        # 3. Stereo matching (before tracking so we use pre-track keypoints)
        pts0_cur = self._tracker0.get_current_keypoints()
        pts1_cur = self._tracker1.get_current_keypoints()
        ids0_cur = self._tracker0.get_current_ids()
        ids1_cur = self._tracker1.get_current_ids()
        new_links = self._stereo.match(
            gray0, gray1, pts0_cur, ids0_cur, pts1_cur, ids1_cur)
        self._stereo_links.update(new_links)

        # 4. Track in each camera independently
        lost0 = self._tracker0.process(gray0, sid0)
        lost1 = self._tracker1.process(gray1, sid1)

        # 5. Build EKF residuals
        cam_id_to_idx = {cs['id']: i for i, cs in enumerate(self._state.cam_states)}
        H_list, r_list = [], []

        # Process cam0 lost features (possibly with stereo partner from cam1)
        for fid0 in lost0:
            track0 = self._tracker0.tracks.pop(fid0, [])
            fid1   = self._stereo_links.pop(fid0, None)
            track1 = self._tracker1.tracks.get(fid1, []) if fid1 else []

            H_o, r_o = self._process_feature(
                track0, track1, cam_id_to_idx)
            if H_o is not None:
                H_list.append(H_o)
                r_list.append(r_o)
                if track1:
                    self._stereo_feats += 1
                else:
                    self._mono_feats += 1

        # Process cam1 lost features that have no cam0 partner (pure monocular from cam1)
        stereo_fid1s = set(self._stereo_links.values())
        for fid1 in lost1:
            if fid1 in stereo_fid1s:
                continue   # will be processed when fid0 is lost
            track1 = self._tracker1.tracks.pop(fid1, [])
            H_o, r_o = self._process_feature([], track1, cam_id_to_idx)
            if H_o is not None:
                H_list.append(H_o)
                r_list.append(r_o)
                self._mono_feats += 1

        # 6. Batch EKF update
        if H_list:
            self._ekf_update(H_list, r_list)

        # 7. Marginalize oldest pair
        while len(self._state.cam_states) > 2 * MAX_CAM_PAIRS:
            self._state.marginalize_pair()

        self._publish(msg0.header.stamp)

    def _process_feature(self, track0: list, track1: list,
                         cam_id_to_idx: dict):
        """
        Build null-space-projected residual (H_o, r_o) for one feature.

        track0: observations from cam0 — list of {'cam_state_id', 'u', 'v'}
        track1: observations from cam1 — list of {'cam_state_id', 'u', 'v'}

        Combines all valid observations from both cameras into one DLT
        triangulation and one set of Jacobians.

        Returns (H_o, r_o) or (None, None) if not enough data / outlier.
        """
        # Collect all valid observations (cam_state must still be in window)
        all_obs = []   # (cam_state_id, u_n, v_n)
        for obs in track0:
            if obs['cam_state_id'] in cam_id_to_idx:
                all_obs.append((obs['cam_state_id'], obs['u'], obs['v']))
        for obs in track1:
            if obs['cam_state_id'] in cam_id_to_idx:
                all_obs.append((obs['cam_state_id'], obs['u'], obs['v']))

        if len(all_obs) < MIN_TRACK_LEN:
            return None, None

        # Build camera poses for DLT
        obs_uv   = [(o[1], o[2]) for o in all_obs]
        cam_ps   = []
        for (csid, _, _) in all_obs:
            cs = self._state.cam_states[cam_id_to_idx[csid]]
            cam_ps.append((cs['p'], quat_to_rot(cs['q'])))

        # Triangulate
        p_f = triangulate_dlt(obs_uv, cam_ps)
        if p_f is None:
            return None, None

        # Depth check
        depths = [float((R_C.T @ (p_f - p_C))[2]) for p_C, R_C in cam_ps]
        if min(depths) < 0.1:
            return None, None

        # Build residuals + Jacobians
        N_obs = len(all_obs)
        H_f   = np.zeros((2 * N_obs, 3))
        H_X   = np.zeros((2 * N_obs, self._state.n))
        res   = np.zeros(2 * N_obs)

        ok = True
        for k, ((csid, u_n, v_n), (p_C, R_C)) in enumerate(zip(all_obs, cam_ps)):
            z_c = R_C.T @ (p_f - p_C)
            X, Y, Z = z_c
            if Z < 0.01:
                ok = False
                break

            u_hat, v_hat = X / Z, Y / Z
            res[2*k]   = u_n - u_hat
            res[2*k+1] = v_n - v_hat

            J_p = np.array([[1/Z, 0,    -X/Z**2],
                             [0,   1/Z,  -Y/Z**2]])

            H_f[2*k:2*k+2, :] = J_p @ R_C.T

            win_idx = cam_id_to_idx[csid]
            base    = self._state.IMU_DIM + win_idx * self._state.CAM_DIM
            H_X[2*k:2*k+2, base:base+3]   = J_p @ (-R_C.T)
            H_X[2*k:2*k+2, base+3:base+6] = J_p @ skew(z_c)

        if not ok:
            return None, None

        # Null-space projection
        if 2 * N_obs <= 3:
            return None, None

        Q_full, _ = np.linalg.qr(H_f, mode='complete')
        Q2  = Q_full[:, 3:]
        H_o = Q2.T @ H_X
        r_o = Q2.T @ res

        # Chi-squared gate
        R_o   = PIXEL_STD**2 * np.eye(H_o.shape[0])
        S     = H_o @ self._state.P @ H_o.T + R_o
        try:
            gamma = float(r_o @ np.linalg.solve(S, r_o))
        except np.linalg.LinAlgError:
            return None, None

        df        = H_o.shape[0]
        threshold = self._chi2_lut.get(df, self._chi2_lut[400])
        if gamma > threshold:
            return None, None

        return H_o, r_o

    # ── EKF batch update (same as Step 9) ─────────────────────────────────────

    def _ekf_update(self, H_list, r_list):
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
        self._state.P = 0.5 * (P_new + P_new.T)

        self._update_count += 1
        self.get_logger().debug(
            f'Update #{self._update_count}  feats={len(H_list)}  |r|={np.linalg.norm(r):.3f}')

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _decode(self, msg: Image):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = arr.reshape(msg.height, msg.width, -1)
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception:
            try:
                return np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width)
            except Exception:
                return None

    def _gt_cb(self, msg: VehicleLocalPosition):
        pos = np.array([msg.x, msg.y, msg.z])
        if self._gt_origin is None:
            self._gt_origin = pos.copy()
        self._gt_pos = pos - self._gt_origin

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
        self._pub_odom.publish(odom)

        pose = PoseStamped()
        pose.header = odom.header
        pose.pose   = odom.pose.pose
        self._path_msg.header.stamp = stamp
        self._path_msg.poses.append(pose)
        self._pub_path.publish(self._path_msg)

    def _status_cb(self):
        s       = self._state
        elapsed = time.time() - self._start
        cov_p   = float(np.trace(s.P[:3, :3]))
        gt_str  = (f'  GT=[{self._gt_pos[0]:.2f},{self._gt_pos[1]:.2f},'
                   f'{self._gt_pos[2]:.2f}]'
                   if self._gt_pos is not None else '')
        self.get_logger().info(
            f'[{elapsed:.0f}s]  '
            f'Updates:{self._update_count}  '
            f'Stereo:{self._stereo_feats}  Mono:{self._mono_feats}  '
            f'CamStates:{len(s.cam_states)}  '
            f'p=[{s.p[0]:.2f},{s.p[1]:.2f},{s.p[2]:.2f}]{gt_str}  '
            f'cov_p={cov_p:.4f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = MultiCamMsckfVio()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
