#!/usr/bin/env python3

import rclpy
from ai_navigator.smart_navigator import SmartNavigator
import cv2

def main(args=None):
    rclpy.init(args=args)
    node = SmartNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Enhanced SmartNavigator...")
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()