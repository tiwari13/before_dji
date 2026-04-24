#!/usr/bin/env python3
"""
PHASE 3 — Step 7: VIO Comparator
==================================

Subscribes to:
  /ov_msckf/poseimu          — OpenVINS estimated pose (PoseWithCovarianceStamped)
  /fmu/out/vehicle_local_position_v1  — PX4 ground truth

Compares OpenVINS trajectory against ground truth and generates a report
on Ctrl+C showing:
  • ATE  (Absolute Trajectory Error) — global accuracy
  • RPE  (Relative Pose Error)       — local drift rate

STUDY MATERIAL
--------------
[1] Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM Systems",
    IROS 2012.  Defines ATE and RPE used in all VIO benchmarks.
    https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools

[2] Burri et al., "The EuRoC MAV Dataset", IJRR 2016.
    The standard benchmark dataset for VIO algorithms.
    https://rpg.ifi.uzh.ch/docs/IJRR17_Burri.pdf

[3] OpenVINS output topics:
    /ov_msckf/poseimu          PoseWithCovarianceStamped  (IMU pose in world)
    /ov_msckf/odomimu          Odometry
    /ov_msckf/trackhist        Image  (feature track visualisation)
    /ov_msckf/loop_feats       Image  (loop-closure feature matches, if enabled)

WHY ATE AND RPE?
----------------
ATE measures how far the trajectory drifts in an absolute sense — it requires
aligning the estimated trajectory to the ground truth first (SE3 alignment).
RPE measures how much the trajectory drifts per unit distance — independent
of the alignment, it captures local consistency.

A good VIO should have:
  ATE < 0.5% of path length
  RPE < 0.1% of path length per meter

Run
---
  ros2 run module3_vio vio_comparator
  (keep running alongside OpenVINS and circle_flight; Ctrl+C for report)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseWithCovarianceStamped
from px4_msgs.msg import VehicleLocalPosition
import numpy as np
import time


BEST_EFFORT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)


def _align_trajectories(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Umeyama alignment — finds the similarity transform (s, R, t) that
    minimises sum of squared distances between est and gt trajectories.

    Returns est transformed into the gt frame.

    Reference: Umeyama, "Least-squares estimation of transformation
    parameters between two point patterns", PAMI 1991.

    Both est and gt are (N, 3) arrays of corresponding positions.
    """
    n = est.shape[0]
    mu_e = est.mean(axis=0)
    mu_g = gt.mean(axis=0)

    est_c = est - mu_e
    gt_c  = gt  - mu_g

    sigma2_e = np.mean(np.sum(est_c ** 2, axis=1))
    H = (gt_c.T @ est_c) / n       # covariance

    U, S, Vt = np.linalg.svd(H)
    det_sign = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, det_sign])

    R = U @ D @ Vt
    s = np.sum(S * np.diag(D)) / sigma2_e
    t = mu_g - s * R @ mu_e

    aligned = (s * (R @ est_c.T)).T + mu_g
    return aligned


def _compute_ate(est: np.ndarray, gt: np.ndarray) -> dict:
    """
    Absolute Trajectory Error after SE3 alignment.
    Returns: {rmse, mean, max, std} in metres.
    """
    if len(est) < 4:
        return {'rmse': float('nan'), 'mean': float('nan'),
                'max': float('nan'), 'std': float('nan')}

    n = min(len(est), len(gt))
    aligned = _align_trajectories(est[:n], gt[:n])
    err = np.linalg.norm(aligned - gt[:n], axis=1)
    return {
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'mean': float(np.mean(err)),
        'max':  float(np.max(err)),
        'std':  float(np.std(err)),
    }


def _compute_rpe(est: np.ndarray, gt: np.ndarray, delta: int = 5) -> dict:
    """
    Relative Pose Error — translational part only.
    delta: number of poses between pairs (default 5).

    For each pair (i, i+delta) compute the difference between
    the estimated relative translation and the ground truth.
    """
    n = min(len(est), len(gt))
    if n <= delta:
        return {'rmse': float('nan'), 'mean': float('nan')}

    errs = []
    for i in range(n - delta):
        d_est = np.linalg.norm(est[i + delta] - est[i])
        d_gt  = np.linalg.norm(gt[i + delta]  - gt[i])
        errs.append(abs(d_est - d_gt))

    return {
        'rmse': float(np.sqrt(np.mean(np.array(errs) ** 2))),
        'mean': float(np.mean(errs)),
    }


class VioComparator(Node):

    def __init__(self):
        super().__init__('vio_comparator')

        # ── Trajectory buffers ────────────────────────────────────────────────
        self._vio_traj: list  = []   # [(stamp, x, y, z)] from OpenVINS
        self._gt_traj:  list  = []   # [(stamp, x, y, z)] from PX4
        self._gt_origin        = None
        self._vio_origin       = None

        # ── Covariance tracking (how confident is OpenVINS in its estimate?) ──
        self._cov_trace: list = []   # trace of position covariance 3×3 block

        self._start = time.time()

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/ov_msckf/poseimu',
            self._vio_cb,
            10,
        )
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self._gt_cb,
            BEST_EFFORT_QOS,
        )

        self.create_timer(10.0, self._status_cb)

        self.get_logger().info('=' * 60)
        self.get_logger().info('PHASE 3  Step 7: VIO Comparator')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Listening for OpenVINS on /ov_msckf/poseimu')
        self.get_logger().info('Listening for GT on /fmu/out/vehicle_local_position_v1')
        self.get_logger().info('Ctrl+C to generate report')

    def _vio_cb(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = np.array([p.x, p.y, p.z])

        if self._vio_origin is None:
            self._vio_origin = pos.copy()
        self._vio_traj.append(
            np.concatenate([[stamp], pos - self._vio_origin])
        )

        # Trace of 3x3 position covariance block (rows 0-2, cols 0-2)
        cov = msg.pose.covariance   # 6×6 row-major flat array
        pos_cov_trace = cov[0] + cov[7] + cov[14]   # (0,0), (1,1), (2,2)
        self._cov_trace.append(float(pos_cov_trace))

    def _gt_cb(self, msg: VehicleLocalPosition):
        pos = np.array([msg.x, msg.y, msg.z])
        stamp = msg.timestamp * 1e-6   # microseconds → seconds

        if self._gt_origin is None:
            self._gt_origin = pos.copy()
        self._gt_traj.append(
            np.concatenate([[stamp], pos - self._gt_origin])
        )

    def _status_cb(self):
        elapsed = time.time() - self._start
        n_vio = len(self._vio_traj)
        n_gt  = len(self._gt_traj)

        if n_vio == 0:
            self.get_logger().info(
                f'[{elapsed:.0f}s] Waiting for OpenVINS output '
                f'(GT pts: {n_gt})'
            )
        else:
            latest_cov = self._cov_trace[-1] if self._cov_trace else float('nan')
            self.get_logger().info(
                f'[{elapsed:.0f}s] VIO pts: {n_vio}  GT pts: {n_gt}  '
                f'Pos-cov trace: {latest_cov:.4f} m²'
            )

    def generate_report(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print('\n' + '=' * 70)
        print('PHASE 3 — Step 7: VIO COMPARATOR REPORT')
        print('=' * 70)

        vio_arr = np.array(self._vio_traj) if self._vio_traj else None
        gt_arr  = np.array(self._gt_traj)  if self._gt_traj  else None

        if vio_arr is None or len(vio_arr) < 4:
            print('\n  No OpenVINS data received.')
            print('  Make sure OpenVINS is running and the drone is moving.')
            print('  Check: ros2 topic echo /ov_msckf/poseimu')
            return

        if gt_arr is None or len(gt_arr) < 4:
            print('\n  No ground truth received — cannot compute ATE/RPE.')
            return

        # Align timestamps: find nearest GT for each VIO pose
        gt_times  = gt_arr[:, 0]
        vio_times = vio_arr[:, 0]

        gt_matched = []
        for vt in vio_times:
            idx = np.argmin(np.abs(gt_times - vt))
            gt_matched.append(gt_arr[idx, 1:])
        gt_matched = np.array(gt_matched)

        vio_pos = vio_arr[:, 1:]

        ate = _compute_ate(vio_pos, gt_matched)
        rpe = _compute_rpe(vio_pos, gt_matched, delta=5)

        # Path length
        path_len = float(np.sum(np.linalg.norm(np.diff(gt_matched, axis=0), axis=1)))

        print(f'\n  VIO poses:        {len(vio_pos)}')
        print(f'  GT poses matched: {len(gt_matched)}')
        print(f'  Path length (GT): {path_len:.2f} m')

        print(f'\n  ── Absolute Trajectory Error (ATE) ──')
        print(f'  RMSE:  {ate["rmse"]:.4f} m')
        print(f'  Mean:  {ate["mean"]:.4f} m')
        print(f'  Max:   {ate["max"]:.4f} m')
        print(f'  Std:   {ate["std"]:.4f} m')
        if path_len > 0:
            print(f'  ATE/path: {ate["rmse"]/path_len*100:.2f}%  '
                  f'(good VIO < 0.5%)')

        print(f'\n  ── Relative Pose Error (RPE, δ=5 poses) ──')
        print(f'  RMSE: {rpe["rmse"]:.4f} m')
        print(f'  Mean: {rpe["mean"]:.4f} m')

        if self._cov_trace:
            cov_arr = np.array(self._cov_trace)
            print(f'\n  ── EKF Covariance (position trace) ──')
            print(f'  Initial: {cov_arr[0]:.4f} m²')
            print(f'  Final:   {cov_arr[-1]:.4f} m²')
            print(f'  (decreasing covariance = EKF gaining confidence)')

        # Assessment
        print('\n  ── Assessment ──')
        if ate['rmse'] < 0.5:
            print('  ATE EXCELLENT — OpenVINS tracking is very accurate')
        elif ate['rmse'] < 2.0:
            print('  ATE GOOD — acceptable for most navigation tasks')
        else:
            print('  ATE POOR — check calibration and sensor rates')

        print('\n  ── What to study next ──')
        print('  1. Open openvins_mono.yaml and read every parameter comment')
        print('  2. Compare T_imu_cam with SDF: is it correct?')
        print('  3. Try: ros2 topic echo /ov_msckf/poseimu')
        print('     Watch the covariance shrink after good feature tracks')
        print('  4. View tracks: rqt → /ov_msckf/trackhist')
        print('  5. Now you understand what Step 8 (your EKF) must replicate')

        # ── Plot ──────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Trajectory comparison
        ax = axes[0]
        aligned_vio = _align_trajectories(vio_pos, gt_matched)
        ax.plot(gt_matched[:, 0],      gt_matched[:, 1],
                'r--', lw=2, label='GT (PX4)')
        ax.plot(aligned_vio[:, 0],     aligned_vio[:, 1],
                'b-',  lw=1.5, label='OpenVINS (aligned)')
        ax.scatter(*gt_matched[0, :2],  c='g', s=100, zorder=5, label='Start')
        ax.scatter(*gt_matched[-1, :2], c='r', s=100, marker='x', zorder=5)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title(f'Trajectory  ATE={ate["rmse"]:.3f} m')
        ax.legend(fontsize=8); ax.grid(True); ax.set_aspect('equal')

        # ATE over time
        ax2 = axes[1]
        errs = np.linalg.norm(aligned_vio - gt_matched, axis=1)
        ax2.plot(vio_times - vio_times[0], errs, 'b-', lw=1.5)
        ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Position error (m)')
        ax2.set_title('ATE over time\n(should stay flat or grow slowly)')
        ax2.grid(True)

        # Covariance
        ax3 = axes[2]
        if self._cov_trace:
            cov_arr = np.array(self._cov_trace)
            ax3.plot(cov_arr, 'g-', lw=1.5)
            ax3.set_xlabel('VIO update index')
            ax3.set_ylabel('Position covariance trace (m²)')
            ax3.set_title('EKF Confidence\n(↓ = EKF getting surer)')
            ax3.set_yscale('log')
            ax3.grid(True)

        plt.suptitle('Phase 3 — Step 7: OpenVINS vs Ground Truth',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        out = 'vio_comparator_step7.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f'\n  Saved: {out}')
        print('=' * 70)


def main(args=None):
    rclpy.init(args=args)
    node = VioComparator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nGenerating comparison report...')
        node.generate_report()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
