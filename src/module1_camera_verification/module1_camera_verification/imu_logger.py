#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np
import csv
from collections import deque

class IMULogger(Node):
    """
    Log IMU data and compute statistics
    Goal: Understand IMU noise characteristics
    """
    
    def __init__(self):
        super().__init__('imu_logger')
        
        # Storage for statistics
        self.accel_data = {'x': deque(maxlen=1000), 
                          'y': deque(maxlen=1000), 
                          'z': deque(maxlen=1000)}
        self.gyro_data = {'x': deque(maxlen=1000), 
                         'y': deque(maxlen=1000), 
                         'z': deque(maxlen=1000)}
        
        # For CSV logging
        self.log_file = open('imu_data_module3.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow([
            'timestamp', 
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z'
        ])
        
        # Subscribe to IMU
        imu_topic = '/world/default/model/x500_skydio_0/link/base_link/sensor/imu_sensor/imu'
        
        self.imu_sub = self.create_subscription(
            Imu, imu_topic,
            self.imu_callback, 10
        )
        
        # Statistics timer
        self.create_timer(5.0, self.print_statistics)
        
        self.sample_count = 0
        
        self.get_logger().info("IMU Logger started!")
        self.get_logger().info("Keep drone STATIONARY for 30 seconds")
        self.get_logger().info("This measures noise characteristics")
    
    def imu_callback(self, msg):
        # Extract data
        accel = msg.linear_acceleration
        gyro = msg.angular_velocity
        
        # Store for statistics
        self.accel_data['x'].append(accel.x)
        self.accel_data['y'].append(accel.y)
        self.accel_data['z'].append(accel.z)
        
        self.gyro_data['x'].append(gyro.x)
        self.gyro_data['y'].append(gyro.y)
        self.gyro_data['z'].append(gyro.z)
        
        # Log to CSV
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.csv_writer.writerow([
            timestamp,
            accel.x, accel.y, accel.z,
            gyro.x, gyro.y, gyro.z
        ])
        
        self.sample_count += 1
    
    def print_statistics(self):
        if len(self.accel_data['x']) < 10:
            return
        
        self.get_logger().info("\n" + "=" * 70)
        self.get_logger().info(f"IMU STATISTICS (from {self.sample_count} samples)")
        self.get_logger().info("=" * 70)
        
        # Accelerometer statistics (drone stationary → should read ~9.81 in Z)
        self.get_logger().info("\nAccelerometer (m/s²):")
        for axis in ['x', 'y', 'z']:
            data = np.array(self.accel_data[axis])
            mean = np.mean(data)
            std = np.std(data)
            
            self.get_logger().info(
                f"  {axis.upper()}: mean={mean:7.4f}, std={std:7.4f}"
            )
            
            # Check Z axis (should be ~9.81 when stationary)
            if axis == 'z':
                expected_gravity = 9.81
                error = abs(mean - expected_gravity)
                self.get_logger().info(
                    f"       (Expected: {expected_gravity}, Error: {error:.4f})"
                )
        
        # Gyroscope statistics (stationary → should be ~0)
        self.get_logger().info("\nGyroscope (rad/s):")
        for axis in ['x', 'y', 'z']:
            data = np.array(self.gyro_data[axis])
            mean = np.mean(data)
            std = np.std(data)
            
            self.get_logger().info(
                f"  {axis.upper()}: mean={mean:7.4f}, std={std:7.4f}"
            )
            
            # Bias check
            if abs(mean) > 0.01:
                self.get_logger().warn(
                    f"       ⚠ Bias detected! (Should be ~0 when stationary)"
                )
        
        self.get_logger().info("=" * 70 + "\n")
    
    def shutdown(self):
        self.log_file.close()
        self.get_logger().info("✓ Saved IMU data to imu_data_module3.csv")


def main(args=None):
    rclpy.init(args=args)
    node = IMULogger()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nStopping IMU logger...")
        node.shutdown()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
