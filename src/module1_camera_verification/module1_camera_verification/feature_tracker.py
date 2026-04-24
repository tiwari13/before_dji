#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque

class FeatureTracker(Node):
    """
    Track features between consecutive frames
    This is the foundation for Visual Odometry
    """
    
    def __init__(self):
        super().__init__('feature_tracker')
        
        self.bridge = CvBridge()
        
        # Use ORB detector (good balance of speed and quality)
        self.detector = cv2.ORB_create(nfeatures=1000)
        
        # Brute-force matcher with Hamming distance (for ORB)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Previous frame data
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_image = None
        
        # Statistics
        self.match_counts = deque(maxlen=100)
        self.match_rates = deque(maxlen=100)
        
        # Subscribe to front camera
        base_path = '/world/default/model/x500_skydio_0/model'
        image_topic = f'{base_path}/camera_front/link/camera_link/sensor/IMX214/image'
        
        self.image_sub = self.create_subscription(
            Image, image_topic,
            self.track_features, 10
        )
        
        # For visualization
        self.show_visualization = True
        
        self.get_logger().info("Feature tracker ready!")
        self.get_logger().info("Tracking features between frames...")
        self.get_logger().info("Press Ctrl+C to stop and see statistics")
    
    def track_features(self, msg):
        # Convert to grayscale
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect features in current frame
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 10:
            self.get_logger().warn(f"Only {len(keypoints)} features detected - too few!")
            return
        
        # If we have a previous frame, match features
        if self.prev_descriptors is not None and len(self.prev_descriptors) > 0:
            try:
                # Match descriptors
                matches = self.matcher.match(descriptors, self.prev_descriptors)
                
                # Sort by distance (best matches first)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Filter good matches (distance < threshold)
                good_matches = [m for m in matches if m.distance < 50]
                
                # Calculate statistics
                match_count = len(good_matches)
                match_rate = (match_count / len(keypoints)) * 100
                
                self.match_counts.append(match_count)
                self.match_rates.append(match_rate)
                
                # Log results
                status = "✓ GOOD" if match_count > 100 else "⚠ LOW" if match_count > 50 else "✗ POOR"
                
                self.get_logger().info(
                    f"Detected: {len(keypoints):4d} | "
                    f"Matched: {match_count:4d} ({match_rate:5.1f}%) | "
                    f"Status: {status}"
                )
                
                # Visualize if enabled
                # query=current, train=previous — must match draw order in visualize_matches
                if self.show_visualization and len(good_matches) > 0:
                    self.visualize_matches(
                        cv_image, keypoints,
                        self.prev_image, self.prev_keypoints,
                        good_matches
                    )
                
                # Quality warnings
                if match_count < 50:
                    self.get_logger().warn(
                        "⚠ LOW TRACKING: < 50 matches (VIO will struggle!)"
                    )
                    self.get_logger().warn(
                        "  Try: Slow down drone motion or improve lighting"
                    )
                
            except cv2.error as e:
                self.get_logger().error(f"Matching error: {e}")
        
        # Store current frame for next iteration
        self.prev_gray = gray
        self.prev_image = cv_image
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
    
    def visualize_matches(self, current_img, current_kp, prev_img, prev_kp, matches):
        """Draw feature matches between frames.

        Match order: matcher.match(current_descriptors, prev_descriptors)
          queryIdx -> current frame keypoints
          trainIdx -> previous frame keypoints
        draw order must follow the same convention: current first, previous second.
        """
        # Draw matches (show top 50 for clarity)
        # current = query (first), previous = train (second)
        match_img = cv2.drawMatches(
            current_img, current_kp,
            prev_img, prev_kp,
            matches[:50],
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # Add info overlay
        info_text = f"Matches: {len(matches)}"
        cv2.putText(match_img, info_text,
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                   1.2, (0, 255, 0), 2)

        # Save to file every frame (headless-safe)
        cv2.imwrite('feature_tracking_latest.png', match_img)

        # Also show live window if display is available
        try:
            cv2.imshow('Feature Tracking', match_img)
            cv2.waitKey(1)
        except cv2.error:
            pass
    
    def print_statistics(self):
        """Print final statistics"""
        if len(self.match_counts) == 0:
            self.get_logger().warn("No tracking data collected")
            return
        
        avg_matches = np.mean(self.match_counts)
        std_matches = np.std(self.match_counts)
        min_matches = np.min(self.match_counts)
        max_matches = np.max(self.match_counts)
        avg_rate = np.mean(self.match_rates)
        
        print("\n" + "=" * 70)
        print("FEATURE TRACKING STATISTICS")
        print("=" * 70)
        print(f"Frames processed: {len(self.match_counts)}")
        print(f"\nMatches per frame:")
        print(f"  Average: {avg_matches:.1f} ± {std_matches:.1f}")
        print(f"  Range: {min_matches:.0f} to {max_matches:.0f}")
        print(f"  Match rate: {avg_rate:.1f}%")
        print("\nVIO Quality Assessment:")
        
        if avg_matches > 150:
            print("  ★★★ EXCELLENT: Perfect for VIO!")
            print("  Your camera motion and environment are ideal.")
        elif avg_matches > 100:
            print("  ★★☆ GOOD: Should work well for VIO")
            print("  Sufficient features for reliable tracking.")
        elif avg_matches > 50:
            print("  ★☆☆ ACCEPTABLE: May work for VIO")
            print("  Consider slowing drone motion or improving lighting.")
        else:
            print("  ☆☆☆ POOR: Not enough for reliable VIO")
            print("  Issues:")
            print("    - Camera motion too fast?")
            print("    - Textureless environment?")
            print("    - Poor lighting?")
        
        print("=" * 70 + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = FeatureTracker()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nStopping feature tracker...")
        node.print_statistics()
        cv2.destroyAllWindows()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
