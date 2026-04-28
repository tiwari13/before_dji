#!/usr/bin/env python3
"""
STEP 5: Camera Projection Testing
Task 5a: Verify camera intrinsics and 3D→2D projection

Reads intrinsics live from /camera_info so it stays correct if the
SDF or calibration changes.

Run:
  ros2 run module1_camera_verification projection_tester
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ProjectionTester(Node):
    """Test 3D→2D projection using live camera intrinsics from CameraInfo."""

    def __init__(self):
        super().__init__('projection_tester')

        self._intrinsics_received = False

        base_path = '/world/default/model/x500_skydio_0'
        info_topic = (
            f'{base_path}/model/camera_front/link/camera_link'
            f'/sensor/IMX214/camera_info'
        )

        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self._camera_info_callback, 1
        )

        self.get_logger().info('Waiting for CameraInfo on:')
        self.get_logger().info(f'  {info_topic}')

    # ── CameraInfo callback ────────────────────────────────────────

    def _camera_info_callback(self, msg):
        if self._intrinsics_received:
            return

        self._intrinsics_received = True

        # K is row-major: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.image_width = msg.width
        self.image_height = msg.height

        self.K = np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1      ]
        ])

        self.get_logger().info('CameraInfo received:')
        self.get_logger().info(f'  fx={self.fx:.2f}  fy={self.fy:.2f}')
        self.get_logger().info(f'  cx={self.cx:.2f}  cy={self.cy:.2f}')
        self.get_logger().info(f'  resolution={self.image_width}x{self.image_height}')

        # Run all tests now that intrinsics are known
        self._run_all_tests()

        # Done — no need to keep spinning
        raise SystemExit

    # ── Projection helpers ─────────────────────────────────────────

    def project_point(self, point_3d):
        """
        Project a 3D point to 2D pixel coordinates.

        Args:
            point_3d: [x, y, z] in camera frame (metres)
                      x: right, y: down, z: forward

        Returns:
            pixel: [u, v] in image (pixels)
        """
        p = np.array(point_3d, dtype=float)
        if p[2] <= 0:
            raise ValueError(
                f'project_point: z={p[2]:.3f} <= 0 — point is behind or on the camera plane'
            )
        h = self.K @ p
        return h[:2] / h[2]

    def is_in_image(self, pixel):
        u, v = pixel
        return 0 <= u < self.image_width and 0 <= v < self.image_height

    # ── Tests ──────────────────────────────────────────────────────

    def _run_all_tests(self):
        print('\n' + '=' * 70)
        print('MODULE 1 — STEP 5: CAMERA PROJECTION TESTING')
        print('=' * 70)
        self._test_fov()
        self._test_basic_projection()
        self._test_grid_projection()
        self._visualize_frustum()
        print('=' * 70)
        print('✔ All tests complete!')
        print('=' * 70 + '\n')

    def _test_fov(self):
        print('\n┌─ TEST 1: Field of View ───────────────────────────────┐')

        fov_h = 2 * np.arctan(self.image_width  / (2 * self.fx))
        fov_v = 2 * np.arctan(self.image_height / (2 * self.fy))

        print(f'  Horizontal FOV: {np.degrees(fov_h):.2f}°')
        print(f'  Vertical FOV:   {np.degrees(fov_v):.2f}°')
        print()
        print('  Visible area at range:')
        for d in [1, 3, 5, 10, 20]:
            w = 2 * d * np.tan(fov_h / 2)
            h = 2 * d * np.tan(fov_v / 2)
            print(f'    {d:2d}m:  {w:.2f}m wide × {h:.2f}m tall')

        print('└──────────────────────────────────────────────────────┘')

    def _test_basic_projection(self):
        print('\n┌─ TEST 2: Basic Projection ────────────────────────────┐')

        test_points = [
            ([0, 0, 5],  '5m straight ahead, centre'),
            ([1, 0, 5],  '5m ahead, 1m right'),
            ([-1, 0, 5], '5m ahead, 1m left'),
            ([0, 1, 5],  '5m ahead, 1m down'),
            ([0, -1, 5], '5m ahead, 1m up'),
            ([0, 0, 10], '10m straight ahead'),
        ]

        for point_3d, description in test_points:
            pixel = self.project_point(point_3d)
            status = '✔ VISIBLE' if self.is_in_image(pixel) else '✗ OUT OF VIEW'
            print(f'  {description}')
            print(f'    3D: {point_3d}  →  pixel: ({pixel[0]:.1f}, {pixel[1]:.1f})  {status}')

        print('└──────────────────────────────────────────────────────┘')

    def _test_grid_projection(self):
        print('\n┌─ TEST 3: Grid Projection ─────────────────────────────┐')

        visible = 0
        total = 0
        for x in np.linspace(-3, 3, 7):
            for y in np.linspace(-2, 2, 5):
                for z in [3, 5, 10]:
                    total += 1
                    if self.is_in_image(self.project_point([x, y, z])):
                        visible += 1

        print(f'  Grid: 7×5 points at 3 depths = {total} total')
        print(f'  Visible: {visible}/{total}  ({visible/total*100:.1f}%)')
        print('└──────────────────────────────────────────────────────┘')

    def _visualize_frustum(self):
        print('\n┌─ TEST 4: Camera Frustum ──────────────────────────────┐')

        try:
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(0, 0, 0, c='red', s=300, marker='^',
                       label='Camera', edgecolors='black', linewidths=2)

            K_inv = np.linalg.inv(self.K)
            colors = ['blue', 'green', 'orange']
            depths = [3, 5, 10]

            for depth, color in zip(depths, colors):
                corners_px = [
                    [0,                 0                 ],
                    [self.image_width,  0                 ],
                    [self.image_width,  self.image_height ],
                    [0,                 self.image_height ],
                ]
                corners_3d = []
                for u, v in corners_px:
                    ray = K_inv @ np.array([u, v, 1.0])
                    corners_3d.append(ray * depth / ray[2])
                corners_3d = np.array(corners_3d)

                for i in range(4):
                    j = (i + 1) % 4
                    ax.plot(
                        [corners_3d[i, 0], corners_3d[j, 0]],
                        [corners_3d[i, 1], corners_3d[j, 1]],
                        [corners_3d[i, 2], corners_3d[j, 2]],
                        c=color, linewidth=2,
                        label=f'{depth}m' if i == 0 else '',
                    )
                for corner in corners_3d:
                    ax.plot([0, corner[0]], [0, corner[1]], [0, corner[2]],
                            c=color, linestyle='--', alpha=0.3, linewidth=1)

            ax.set_xlabel('X (m) — Right')
            ax.set_ylabel('Y (m) — Down')
            ax.set_zlabel('Z (m) — Forward')
            ax.set_title('Camera Viewing Frustum')
            ax.legend()
            half = 5
            ax.set_xlim([-half, half])
            ax.set_ylim([-half, half])
            ax.set_zlim([0, 10])

            plt.tight_layout()
            plt.savefig('camera_frustum_module1.png', dpi=150, bbox_inches='tight')
            print('  ✔ Saved: camera_frustum_module1.png')

            try:
                plt.show()
            except Exception:
                pass

        except Exception as e:
            print(f'  ⚠ 3D frustum visualization skipped: {e}')
            print('    (Other tests above are unaffected)')

        print('└──────────────────────────────────────────────────────┘')


def main(args=None):
    rclpy.init(args=args)
    node = ProjectionTester()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
