#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraVerifier(Node):
    """Verify all 6 cameras are publishing correctly"""
    
    def __init__(self):
        super().__init__('camera_verifier')
        
        self.bridge = CvBridge()
        self.loop_test=0
        
        # Track status of all 6 cameras
        self.cameras = {
            'front': {'image_ok': False, 'info_ok': False, 'info': None},
            'right': {'image_ok': False, 'info_ok': False, 'info': None},
            'back': {'image_ok': False, 'info_ok': False, 'info': None},
            'left': {'image_ok': False, 'info_ok': False, 'info': None},
            'up': {'image_ok': False, 'info_ok': False, 'info': None},
            'down': {'image_ok': False, 'info_ok': False, 'info': None}
        }
        
        # Base topic path for your drone
        base_path = '/world/default/model/x500_skydio_0/model'
        
        # Subscribe to all cameras
        for cam_name in self.cameras.keys():
            print(f"loop test camera: {self.loop_test}")
            # Image topic
            image_topic = f'{base_path}/camera_{cam_name}/link/camera_link/sensor/IMX214/image'
            self.create_subscription(
                Image,
                image_topic,
                lambda msg, name=cam_name: self.image_callback(msg, name),
                10
            )
            
            # Camera info topic
            info_topic = f'{base_path}/camera_{cam_name}/link/camera_link/sensor/IMX214/camera_info'
            self.create_subscription(
                CameraInfo,
                info_topic,
                lambda msg, name=cam_name: self.info_callback(msg, name),
                10
            )
        
        # Status report timer (every 3 seconds)
        self.create_timer(3.0, self.report_status)
        self.loop_test += 1
        print(f"loop test timer: {self.loop_test}")
        
        self.get_logger().info("Camera Verifier Started for x500_skydio_0!")
        self.get_logger().info("Waiting for camera data...")
    
    def image_callback(self, msg, camera_name):
        """Mark that we received an image from this camera"""
        self.cameras[camera_name]['image_ok'] = True
    
    def info_callback(self, msg, camera_name):
        """Store camera info"""
        self.cameras[camera_name]['info_ok'] = True
        self.cameras[camera_name]['info'] = msg
    
    def report_status(self):
        """Print status of all cameras"""
        self.get_logger().info("=" * 70)
        self.get_logger().info("CAMERA STATUS REPORT - x500_skydio_0")
        self.get_logger().info("=" * 70)
        
        all_ok = True
        
        for cam_name, status in self.cameras.items():
            image_status = "✓" if status['image_ok'] else "✗"
            info_status = "✓" if status['info_ok'] else "✗"
            
            overall = "OK" if (status['image_ok'] and status['info_ok']) else "FAIL"
            
            if not (status['image_ok'] and status['info_ok']):
                all_ok = False
            
            self.get_logger().info(
                f"{cam_name:10s}: Image={image_status} Info={info_status} [{overall}]"
            )
            
            # If we have info, print parameters
            if status['info']:
                info = status['info']
                fx = info.k[0]
                fy = info.k[4]
                cx = info.k[2]
                cy = info.k[5]
                
                self.get_logger().info(
                    f"            Resolution: {info.width}x{info.height}"
                )
                self.get_logger().info(
                    f"            fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}"
                )
        
        self.get_logger().info("=" * 70)
        
        if all_ok:
            self.get_logger().info("✓ ALL CAMERAS WORKING!")
        else:
            self.get_logger().warn("⚠ Some cameras not responding")
        
        self.get_logger().info("=" * 70 + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = CameraVerifier()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
### Step 3: Now You Can Proceed with Module 1!

