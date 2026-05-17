#!/usr/bin/env python3
"""
STEP 4: Camera-IMU Temporal Calibration
Task 4a: Sensor Rate Analysis & Timestamp Synchronization

Goals:
  1. Measure actual publishing rates of camera and IMU
  2. Analyze timestamp alignment between the two sensors
  3. Build an IMU pre-integration buffer (foundation for VIO)
  4. Compute the camera-IMU time offset (td)

Run:
  ros2 run module1_camera_verification cam_imu_sync

Keep the drone hovering for ~30 seconds, then Ctrl+C for the full report.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
import numpy as np
from collections import deque
import csv
import time


class CamIMUSync(Node):
    """
    Analyze temporal relationship between camera and IMU.
    This is the foundation for Visual-Inertial Odometry.
    """

    def __init__(self):
        super().__init__('cam_imu_sync')

        # ── Timestamp storage ──────────────────────────────────
        self.cam_arrival_times = deque(maxlen=2000)   # wall-clock arrival times
        self.imu_arrival_times = deque(maxlen=20000)  # wall-clock arrival times

        # Raw header timestamps (from Gazebo simulation clock)
        self.cam_header_times = deque(maxlen=2000)
        self.imu_header_times = deque(maxlen=20000)

        # ── IMU Pre-integration Buffer ─────────────────────────
        # Between each camera frame, we collect all IMU samples
        self.imu_buffer = []              # IMU msgs since last camera frame
        self.preintegrated_segments = deque(maxlen=2000)  # list of (cam_time, [imu_msgs]) pairs
        self.last_cam_time = None

        # ── Synchronization pairs ──────────────────────────────
        # For each camera frame, find the nearest IMU measurement
        self.sync_pairs = deque(maxlen=2000)  # (cam_time, nearest_imu_time, offset)

        # ── Subscribers ────────────────────────────────────────
        # Topics match camera_bridge_oakd_pro_w_front.yaml
        cam_topic = '/cam_front/left/image_raw'
        imu_topic = '/cam_front/imu'

        self.cam_sub = self.create_subscription(
            Image, cam_topic, self.cam_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, imu_topic, self.imu_callback, 50
        )

        # ── Status timer ───────────────────────────────────────
        self.create_timer(5.0, self.print_live_status)
        self.start_time = time.time()

        self.get_logger().info("=" * 60)
        self.get_logger().info("STEP 4: Camera-IMU Temporal Calibration")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Hover the drone for ~30 seconds, then Ctrl+C")
        self.get_logger().info("Analyzing camera and IMU timing...")
        self.get_logger().info("")

    # ── Callbacks ──────────────────────────────────────────────

    def cam_callback(self, msg):
        """Process each camera frame arrival"""
        now = time.time()
        header_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self.cam_arrival_times.append(now)
        self.cam_header_times.append(header_time)

        # ── Build pre-integration segment ──────────────────────
        # Collect all IMU measurements between this frame and the last
        if self.last_cam_time is not None and len(self.imu_buffer) > 0:
            dt = header_time - self.last_cam_time
            imu_count = len(self.imu_buffer)
            # Only keep segments with realistic dt (50–500 ms) and ≥3 IMU samples
            if 0.05 <= dt <= 0.5 and imu_count >= 3:
                self.preintegrated_segments.append({
                    'cam_time': header_time,
                    'prev_cam_time': self.last_cam_time,
                    'imu_count': imu_count,
                    'imu_samples': list(self.imu_buffer),  # copy
                    'dt': dt,
                })

        self.imu_buffer.clear()
        self.last_cam_time = header_time

        # ── Find nearest IMU timestamp (recent window only) ────
        # Search only the last 100 IMU samples (~400 ms at 250 Hz)
        # to avoid stale matches from the far past.
        if len(self.imu_header_times) > 0:
            recent_imu = np.array(list(self.imu_header_times)[-100:])
            idx = np.argmin(np.abs(recent_imu - header_time))
            nearest_imu_time = recent_imu[idx]
            offset_ms = (header_time - nearest_imu_time) * 1000  # ms

            # Bracket check: find prev and next IMU around this camera frame
            prev_imu = recent_imu[recent_imu <= header_time]
            next_imu = recent_imu[recent_imu > header_time]
            bracketed = len(prev_imu) > 0 and len(next_imu) > 0
            bracket_width_ms = (
                (next_imu[0] - prev_imu[-1]) * 1000 if bracketed else float('nan')
            )

            self.sync_pairs.append({
                'cam_time': header_time,
                'imu_time': nearest_imu_time,
                'offset_ms': offset_ms,
                'bracketed': bracketed,
                'bracket_width_ms': bracket_width_ms,
            })

    def imu_callback(self, msg):
        """Process each IMU measurement"""
        now = time.time()
        header_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self.imu_arrival_times.append(now)
        self.imu_header_times.append(header_time)

        # Store in pre-integration buffer
        self.imu_buffer.append({
            'time': header_time,
            'accel': [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ],
            'gyro': [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ],
        })

    # ── Live Status ────────────────────────────────────────────

    def print_live_status(self):
        elapsed = time.time() - self.start_time

        cam_count = len(self.cam_arrival_times)
        imu_count = len(self.imu_arrival_times)

        # Compute rates
        cam_rate = self._compute_rate(self.cam_arrival_times)
        imu_rate = self._compute_rate(self.imu_arrival_times)

        # Latest offset
        latest_offset = "N/A"
        if self.sync_pairs:
            latest_offset = f"{self.sync_pairs[-1]['offset_ms']:.2f} ms"

        # Pre-integration info
        seg_count = len(self.preintegrated_segments)
        avg_imu_per_frame = 0
        if seg_count > 0:
            avg_imu_per_frame = np.mean([s['imu_count'] for s in self.preintegrated_segments])

        self.get_logger().info(
            f"[{elapsed:5.1f}s] "
            f"CAM: {cam_rate:5.1f} Hz ({cam_count} frames) | "
            f"IMU: {imu_rate:5.1f} Hz ({imu_count} samples) | "
            f"Offset: {latest_offset} | "
            f"IMU/frame: {avg_imu_per_frame:.1f}"
        )

    def _compute_rate(self, timestamps):
        """Compute Hz from a deque of timestamps"""
        if len(timestamps) < 2:
            return 0.0
        # Use last 100 samples for rate calculation
        recent = list(timestamps)[-100:]
        if len(recent) < 2:
            return 0.0
        dt = recent[-1] - recent[0]
        if dt <= 0:
            return 0.0
        return (len(recent) - 1) / dt

    # ── Full Report ────────────────────────────────────────────

    def generate_report(self):
        """Generate comprehensive temporal calibration report"""
        print("\n" + "=" * 70)
        print("CAMERA-IMU TEMPORAL CALIBRATION REPORT")
        print("=" * 70)

        # ── 1. Sensor Rates ────────────────────────────────────
        print("\n┌─ 1. SENSOR RATES ─────────────────────────────────┐")

        cam_rate = self._compute_rate(self.cam_arrival_times)
        imu_rate = self._compute_rate(self.imu_arrival_times)

        print(f"  Camera:  {cam_rate:6.2f} Hz  ({len(self.cam_arrival_times)} frames)")
        print(f"  IMU:     {imu_rate:6.2f} Hz  ({len(self.imu_arrival_times)} samples)")
        print(f"  Ratio:   {imu_rate/cam_rate:.1f}x (IMU samples per camera frame)" if cam_rate > 0 else "")

        # Camera frame intervals
        if len(self.cam_header_times) > 1:
            cam_dts = np.diff(list(self.cam_header_times)) * 1000  # ms
            print(f"\n  Camera frame intervals:")
            print(f"    Mean:   {np.mean(cam_dts):6.2f} ms")
            print(f"    Std:    {np.std(cam_dts):6.2f} ms")
            print(f"    Min:    {np.min(cam_dts):6.2f} ms")
            print(f"    Max:    {np.max(cam_dts):6.2f} ms")

        # IMU intervals
        if len(self.imu_header_times) > 1:
            imu_dts = np.diff(list(self.imu_header_times)) * 1000  # ms
            print(f"\n  IMU sample intervals:")
            print(f"    Mean:   {np.mean(imu_dts):6.2f} ms")
            print(f"    Std:    {np.std(imu_dts):6.2f} ms")
            print(f"    Min:    {np.min(imu_dts):6.2f} ms")
            print(f"    Max:    {np.max(imu_dts):6.2f} ms")

        print("└──────────────────────────────────────────────────┘")

        # ── 2. Time Offset Analysis ────────────────────────────
        print("\n┌─ 2. TIME OFFSET (td) ANALYSIS ────────────────────┐")

        if self.sync_pairs:
            offsets = np.array([p['offset_ms'] for p in self.sync_pairs])
            bracketed_count = sum(1 for p in self.sync_pairs if p['bracketed'])
            bracket_widths = np.array([
                p['bracket_width_ms'] for p in self.sync_pairs
                if p['bracketed'] and not np.isnan(p['bracket_width_ms'])
            ])

            print(f"  Analyzed {len(offsets)} camera-IMU pairs (recent window)")
            print(f"  Bracketed pairs: {bracketed_count}/{len(offsets)} "
                  f"(camera ts inside valid IMU interval)")
            print(f"\n  Camera-to-nearest-IMU offset:")
            print(f"    Median: {np.median(offsets):7.3f} ms")
            print(f"    P5/P95: {np.percentile(offsets, 5):7.3f} / "
                  f"{np.percentile(offsets, 95):7.3f} ms")
            print(f"    Min:    {np.min(offsets):7.3f} ms")
            print(f"    Max:    {np.max(offsets):7.3f} ms")
            if len(bracket_widths) > 0:
                print(f"\n  IMU bracket width (prev→next around camera frame):")
                print(f"    Median: {np.median(bracket_widths):7.3f} ms")
                print(f"    Max:    {np.max(bracket_widths):7.3f} ms")

            td = np.median(offsets)
            print(f"\n  ► Estimated td = {td:.3f} ms")

            if abs(td) < 1.0:
                print(f"    ✔ Excellent! Near-zero offset (simulation ideal)")
            elif abs(td) < 5.0:
                print(f"    ✔ Good. Small offset, manageable in VIO")
            elif abs(td) < 20.0:
                print(f"    ⚠ Moderate offset. Should be compensated in VIO")
            else:
                print(f"    ✗ Large offset! Must be corrected before VIO")

        print("└──────────────────────────────────────────────────┘")

        # ── 3. Pre-integration Analysis ────────────────────────
        print("\n┌─ 3. IMU PRE-INTEGRATION SEGMENTS ─────────────────┐")

        if self.preintegrated_segments:
            # Segments are pre-validated (dt 50–500 ms, ≥3 IMU samples)
            seg_counts = np.array([s['imu_count'] for s in self.preintegrated_segments])
            seg_dts = np.array([s['dt'] * 1000 for s in self.preintegrated_segments])

            print(f"  Valid segments: {len(self.preintegrated_segments)}")
            print(f"\n  IMU samples per camera frame:")
            print(f"    Median: {np.median(seg_counts):5.1f}")
            print(f"    P5/P95: {np.percentile(seg_counts, 5):.0f} / "
                  f"{np.percentile(seg_counts, 95):.0f}")
            print(f"    Min:    {np.min(seg_counts):5.0f}")
            print(f"    Max:    {np.max(seg_counts):5.0f}")
            print(f"\n  Segment duration (between camera frames):")
            print(f"    Median: {np.median(seg_dts):6.2f} ms")
            print(f"    P5/P95: {np.percentile(seg_dts, 5):.2f} / "
                  f"{np.percentile(seg_dts, 95):.2f} ms")

            # Demonstrate pre-integration on one segment
            if len(self.preintegrated_segments) > 5:
                seg = self.preintegrated_segments[5]
                print(f"\n  Example pre-integration (segment #5):")
                print(f"    Duration: {seg['dt']*1000:.2f} ms")
                print(f"    IMU samples: {seg['imu_count']}")

                # Actually pre-integrate this segment
                delta_v, delta_p, delta_angle = self._preintegrate_segment(seg)
                print(f"    Δvelocity: [{delta_v[0]:.4f}, {delta_v[1]:.4f}, {delta_v[2]:.4f}] m/s")
                print(f"    Δposition: [{delta_p[0]:.6f}, {delta_p[1]:.6f}, {delta_p[2]:.6f}] m")
                print(f"    Δrotation: [{delta_angle[0]:.6f}, {delta_angle[1]:.6f}, {delta_angle[2]:.6f}] rad")

        print("└──────────────────────────────────────────────────┘")

        # ── 4. VIO Readiness ───────────────────────────────────
        print("\n┌─ 4. VIO READINESS CHECK ──────────────────────────┐")

        issues = []
        recommendations = []

        if cam_rate < 9.5:
            issues.append("Camera rate too low (< 9.5 Hz)")
        elif cam_rate < 20:
            recommendations.append("Camera rate is OK but 20+ Hz is better for VIO")

        if imu_rate < 100:
            issues.append("IMU rate too low (< 100 Hz)")
        elif imu_rate < 200:
            recommendations.append("IMU rate is OK but 200+ Hz is ideal")

        if self.sync_pairs:
            td = abs(np.median([p['offset_ms'] for p in self.sync_pairs]))
            if td > 20:
                issues.append(f"Time offset too large ({td:.1f} ms)")
            elif td > 5:
                recommendations.append(f"Time offset ({td:.1f} ms) should be compensated")

        if self.preintegrated_segments:
            avg_imu = np.mean([s['imu_count'] for s in self.preintegrated_segments])
            if avg_imu < 3:
                issues.append(f"Too few IMU samples per frame ({avg_imu:.1f})")

        if not issues:
            print("  ✔ ALL CHECKS PASSED — Ready for VIO!")
        else:
            for issue in issues:
                print(f"  ✗ {issue}")

        for rec in recommendations:
            print(f"  △ {rec}")

        print("└──────────────────────────────────────────────────┘")

        # ── 5. What This Means for VIO ─────────────────────────
        print("\n┌─ 5. WHY THIS MATTERS FOR VIO ────────────────────┐")
        print("                                                    ")
        print("  Camera frame k          Camera frame k+1          ")
        print("       │                        │                   ")
        print("       ▼                        ▼                   ")
        print("  ─────●────────────────────────●─────── time       ")
        print("       │  ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑  │                  ")
        print("       │  IMU measurements       │                  ")
        print("       │  (pre-integrated)        │                  ")
        print("       │                          │                  ")
        print("       └── These IMU samples ─────┘                  ")
        print("           predict how features                      ")
        print("           should move between                       ")
        print("           frames k and k+1                          ")
        print("                                                    ")
        print("  If timestamps are wrong, the prediction is wrong, ")
        print("  and VIO cannot match features to IMU motion.      ")
        print("└──────────────────────────────────────────────────┘")

        print("\n" + "=" * 70)

    def _preintegrate_segment(self, segment):
        """
        Simple IMU pre-integration between two camera frames.
        This is a simplified version — full pre-integration accounts
        for rotation, but this gives you the idea.

        Returns: (delta_velocity, delta_position, delta_angle)
        """
        delta_v = np.array([0.0, 0.0, 0.0])
        delta_p = np.array([0.0, 0.0, 0.0])
        delta_angle = np.array([0.0, 0.0, 0.0])

        gravity = np.array([0.0, 0.0, 9.81])

        samples = segment['imu_samples']
        for i in range(1, len(samples)):
            dt = samples[i]['time'] - samples[i-1]['time']
            if dt <= 0 or dt > 0.1:
                continue

            accel = np.array(samples[i]['accel'])
            gyro = np.array(samples[i]['gyro'])

            # Remove gravity (simplified — assumes level orientation)
            accel_corrected = accel - gravity

            # Integrate
            delta_p += delta_v * dt + 0.5 * accel_corrected * dt * dt
            delta_v += accel_corrected * dt
            delta_angle += gyro * dt

        return delta_v, delta_p, delta_angle

    def save_sync_data(self):
        """Save synchronization data for later analysis"""
        if not self.sync_pairs:
            return

        filename = 'cam_imu_sync_data_step4.csv'
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cam_time', 'imu_time', 'offset_ms'])
            for pair in self.sync_pairs:
                writer.writerow([
                    pair['cam_time'],
                    pair['imu_time'],
                    pair['offset_ms'],
                ])
        print(f"✔ Saved sync data: {filename}")

        # Also save pre-integration segments summary
        if self.preintegrated_segments:
            filename2 = 'preintegration_segments_step4.csv'
            with open(filename2, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['cam_time', 'prev_cam_time', 'dt_ms', 'imu_count'])
                for seg in self.preintegrated_segments:
                    writer.writerow([
                        seg['cam_time'],
                        seg['prev_cam_time'],
                        seg['dt'] * 1000,
                        seg['imu_count'],
                    ])
            print(f"✔ Saved pre-integration data: {filename2}")


def main(args=None):
    rclpy.init(args=args)
    node = CamIMUSync()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nStopping sensor analysis...")
        print("Generating report...\n")
        node.generate_report()
        node.save_sync_data()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()