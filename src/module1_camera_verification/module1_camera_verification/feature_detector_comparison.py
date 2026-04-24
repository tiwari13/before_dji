
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class FeatureDetectorComparison(Node):
    """
    Compare different feature detectors on YOUR camera images
    Goal: Find best detector for VIO
    """

    def __init__(self):
        super().__init__('feature_detector_comparison')

        self.bridge = CvBridge()

        # Initialize all detectors — SIFT guarded for builds where it is absent
        self.detectors = {
            'ORB': cv2.ORB_create(nfeatures=1000),
            'FAST': cv2.FastFeatureDetector_create(threshold=25),
        }
        try:
            self.detectors['SIFT'] = cv2.SIFT_create(nfeatures=1000)
        except AttributeError:
            self.get_logger().warn("SIFT not available in this OpenCV build — skipping")

        # Results storage
        self.results = {
            name: {'counts': [], 'times': []}
            for name in self.detectors.keys()
        }
                
        # Subscribe to front camera
        base_path = '/world/default/model/x500_skydio_0/model'
        image_topic = f'{base_path}/camera_front/link/camera_link/sensor/IMX214/image'

        self.image_sub = self.create_subscription(
            Image, image_topic,
            self.image_callback, 10
        )

        self.frame_count = 0
        self.max_frames = 50  # Test on 50 frames
        self.report_done = False  # guard: run report+visualize exactly once

        # For visualization
        self.last_frame = None

        self.get_logger().info("Feature detector comparison starting...")
        self.get_logger().info("Processing 50 frames, please wait...")

    def image_callback(self, msg):
        # Once the report has been generated, ignore all further frames
        if self.report_done:
            return

        if self.frame_count >= self.max_frames:
            if not self.report_done:
                self.report_results()
                self.visualize_detectors()
                self.report_done = True
            return

        # Convert to grayscale
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Store last frame for visualization
        if self.frame_count == 0:
            self.last_frame = cv_image.copy()

        # Test each detector
        for name, detector in self.detectors.items():
            start_time = time.time()

            # Detect keypoints
            keypoints = detector.detect(gray, None)

            elapsed = (time.time() - start_time) * 1000  # ms

            # Store results
            self.results[name]['counts'].append(len(keypoints))
            self.results[name]['times'].append(elapsed)

        self.frame_count += 1

        if self.frame_count % 10 == 0:
            self.get_logger().info(f"Processed {self.frame_count}/{self.max_frames} frames")

    def report_results(self):
        """Generate comparison report"""
        self.get_logger().info("\n" + "=" * 70)
        self.get_logger().info("FEATURE DETECTION COMPARISON RESULTS")
        self.get_logger().info("=" * 70)

        best_detector = None
        best_score = 0

        for name in self.detectors.keys():
            counts = self.results[name]['counts']
            times = self.results[name]['times']

            if len(counts) == 0:
                continue

            avg_count = np.mean(counts)
            std_count = np.std(counts)
            min_count = np.min(counts)
            max_count = np.max(counts)
            avg_time = np.mean(times)
            fps = 1000 / avg_time if avg_time > 0 else 0

            self.get_logger().info(f"\n{name}:")
            self.get_logger().info(f"  Features detected:")
            self.get_logger().info(f"    Average: {avg_count:.0f} ± {std_count:.0f}")
            self.get_logger().info(f"    Range: {min_count:.0f} to {max_count:.0f}")
            self.get_logger().info(f"  Performance:")
            self.get_logger().info(f"    Time: {avg_time:.2f} ms/frame")
            self.get_logger().info(f"    Speed: {fps:.1f} FPS")

            # Rating based on count and speed
            # Good VIO needs: >500 features and >20 FPS
            if avg_count > 800 and fps > 20:
                rating = "★★★ EXCELLENT for VIO"
                score = 3
            elif avg_count > 500 and fps > 15:
                rating = "★★☆ GOOD for VIO"
                score = 2
            elif avg_count > 300:
                rating = "★☆☆ ACCEPTABLE for VIO"
                score = 1
            else:
                rating = "☆☆☆ TOO FEW features for VIO"
                score = 0

            self.get_logger().info(f"  Rating: {rating}")

            if score > best_score:
                best_score = score
                best_detector = name

        self.get_logger().info("\n" + "=" * 70)
        self.get_logger().info("RECOMMENDATION:")
        if best_detector is not None:
            self.get_logger().info(f"  Use {best_detector} for your VIO system")
            self.get_logger().info(f"  Reason: Best balance of features and speed")
        else:
            self.get_logger().warn("  No detector scored above zero — environment may be textureless")
        self.get_logger().info("=" * 70 + "\n")

    def visualize_detectors(self):
        """Create visual comparison of detectors"""
        if self.last_frame is None:
            self.get_logger().warn("No frame captured for visualization")
            return

        gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)

        # Detect with each detector
        outputs = []

        for name, detector in self.detectors.items():
            # Detect keypoints
            keypoints = detector.detect(gray, None)

            # Draw keypoints
            output = cv2.drawKeypoints(
                self.last_frame.copy(), keypoints, None,
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

            # Add label
            label = f"{name}: {len(keypoints)} features"
            cv2.putText(output, label,
                       (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1.2, (0, 255, 0), 3)

            outputs.append(output)

        # Stack images vertically — all same width (640), no dimension mismatch
        comparison = np.vstack(outputs)

        # Save
        output_path = 'feature_comparison_module2.png'
        cv2.imwrite(output_path, comparison)
        self.get_logger().info(f"✓ Saved visualization: {output_path}")


def main(args=None):
    rclpy.init(args=args)
    node = FeatureDetectorComparison()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
