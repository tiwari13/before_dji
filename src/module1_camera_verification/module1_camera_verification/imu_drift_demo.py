#!/usr/bin/env python3

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
# Remove: from mpl_toolkits.mplot3d import Axes3D  
# It's auto-imported when you use projection='3d'
from scipy.spatial.transform import Rotation

class IMUDriftDemo(Node):
    """
    Demonstrate IMU drift by dead-reckoning position from IMU alone
    This shows WHY we need vision to constrain the estimate
    """
    
    def __init__(self):
        super().__init__('imu_drift_demo')
        
        # State vectors
        self.position = np.array([0.0, 0.0, 0.0])  # [x, y, z]
        self.velocity = np.array([0.0, 0.0, 0.0])  # [vx, vy, vz]
        self.orientation = Rotation.from_quat([0, 0, 0, 1])  # Identity
        
        # Storage for plotting
        self.positions = []
        self.timestamps = []
        
        # Previous timestamp for integration
        self.prev_time = None
        
        # Gravity vector (NED frame)
        self.gravity = np.array([0.0, 0.0, 9.81])
        
        # Subscribe to IMU
        imu_topic = '/world/default/model/x500_skydio_0/link/base_link/sensor/imu_sensor/imu'
        
        self.imu_sub = self.create_subscription(
            Imu, imu_topic,
            self.imu_callback, 10
        )
        
        # Status timer
        self.create_timer(2.0, self.print_status)
        
        self.get_logger().info("IMU Drift Demo started!")
        self.get_logger().info("Hover your drone for 60 seconds and watch the drift...")
        self.get_logger().info("Even though drone is stationary, position will drift!")
    
    def imu_callback(self, msg):
        # Get current time
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        if self.prev_time is None:
            self.prev_time = current_time
            return
        
        dt = current_time - self.prev_time
        self.prev_time = current_time
        
        # Skip if dt is too large (initial jump)
        if dt > 0.5 or dt <= 0:
            return
        
        # Extract IMU data
        accel_body = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        
        gyro_body = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        
        # Update orientation using gyroscope
        # Simple integration: R(t+dt) = R(t) * exp(ω * dt)
        rotation_delta = Rotation.from_rotvec(gyro_body * dt)
        self.orientation = self.orientation * rotation_delta
        
        # Transform acceleration from body to world frame
        accel_world = self.orientation.apply(accel_body)
        
        # Remove gravity (critical step!)
        accel_world = accel_world - self.gravity
        
        # Integrate acceleration → velocity
        self.velocity += accel_world * dt
        
        # Integrate velocity → position  
        self.position += self.velocity * dt
        
        # Store for plotting
        self.positions.append(self.position.copy())
        self.timestamps.append(current_time)
    
    def print_status(self):
        if len(self.positions) < 10:
            return
        
        elapsed = self.timestamps[-1] - self.timestamps[0]
        drift_distance = np.linalg.norm(self.position)
        
        self.get_logger().info(
            f"Time: {elapsed:5.1f}s | "
            f"Position: [{self.position[0]:6.2f}, {self.position[1]:6.2f}, {self.position[2]:6.2f}] | "
            f"Drift: {drift_distance:6.2f}m"
        )
        
        if drift_distance > 0.5:
            self.get_logger().warn(
                "⚠ Significant drift detected! This is normal for IMU-only navigation."
            )
    
    def plot_trajectory(self):
        if len(self.positions) < 2:
            self.get_logger().warn("Not enough data to plot")
            return
        
        positions_array = np.array(self.positions)
        times = np.array(self.timestamps) - self.timestamps[0]
        
        # Create figure with 2 subplots
        fig = plt.figure(figsize=(14, 6))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(positions_array[:, 0], 
                positions_array[:, 1], 
                positions_array[:, 2], 
                'b-', linewidth=2, label='IMU-only trajectory')
        ax1.scatter([0], [0], [0], color='g', s=100, marker='o', label='Start')
        ax1.scatter([positions_array[-1, 0]], 
                   [positions_array[-1, 1]], 
                   [positions_array[-1, 2]], 
                   color='r', s=100, marker='x', label='End')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('IMU-Only Dead Reckoning\n(Drone Hovering)')
        ax1.legend()
        ax1.grid(True)
        
        # Make axes equal scale
        max_range = np.array([
            positions_array[:, 0].max() - positions_array[:, 0].min(),
            positions_array[:, 1].max() - positions_array[:, 1].min(),
            positions_array[:, 2].max() - positions_array[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (positions_array[:, 0].max() + positions_array[:, 0].min()) * 0.5
        mid_y = (positions_array[:, 1].max() + positions_array[:, 1].min()) * 0.5
        mid_z = (positions_array[:, 2].max() + positions_array[:, 2].min()) * 0.5
        
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Drift over time
        ax2 = fig.add_subplot(122)
        drift = np.linalg.norm(positions_array, axis=1)
        ax2.plot(times, drift, 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Drift from Origin (m)')
        ax2.set_title('Position Drift Over Time')
        ax2.grid(True)
        
        # Add annotations
        final_drift = drift[-1]
        final_time = times[-1]
        ax2.annotate(f'Final drift: {final_drift:.2f}m\nafter {final_time:.1f}s',
                    xy=(final_time, final_drift),
                    xytext=(final_time*0.6, final_drift*0.8),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig('imu_drift_module3.png', dpi=150, bbox_inches='tight')
        self.get_logger().info("✓ Saved plot: imu_drift_module3.png")
        plt.show()
    
    def generate_report(self):
        """Generate drift analysis report"""
        if len(self.positions) < 10:
            return
        
        positions_array = np.array(self.positions)
        times = np.array(self.timestamps) - self.timestamps[0]
        drift = np.linalg.norm(positions_array, axis=1)
        
        print("\n" + "=" * 70)
        print("IMU DRIFT ANALYSIS REPORT")
        print("=" * 70)
        print(f"\nTest Duration: {times[-1]:.1f} seconds")
        print(f"Number of IMU samples: {len(self.positions)}")
        print(f"\nDrift Statistics:")
        print(f"  Initial drift (t=10s): {drift[int(len(drift)*0.17)]:.3f} m")
        print(f"  Mid-point drift (t={times[-1]/2:.0f}s): {drift[len(drift)//2]:.3f} m")
        print(f"  Final drift: {drift[-1]:.3f} m")
        print(f"  Drift rate: {drift[-1]/times[-1]:.4f} m/s")
        
        print(f"\nFinal Position Error:")
        print(f"  X: {positions_array[-1, 0]:7.3f} m")
        print(f"  Y: {positions_array[-1, 1]:7.3f} m")
        print(f"  Z: {positions_array[-1, 2]:7.3f} m")
        
        print(f"\nWHY THIS MATTERS:")
        print(f"  • Drone was HOVERING (should stay at [0, 0, 0])")
        print(f"  • But IMU-only estimates it moved {drift[-1]:.2f}m!")
        print(f"  • This is why VIO needs VISION to correct drift")
        print(f"  • Vision provides absolute position measurements")
        print(f"  • Vision also helps estimate IMU bias online")
        
        print("=" * 70 + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = IMUDriftDemo()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nStopping drift demo...")
        print("Generating plots...")
        node.plot_trajectory()
        node.generate_report()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
