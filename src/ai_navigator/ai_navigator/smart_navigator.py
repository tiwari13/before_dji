import numpy as np
import math
import time
import cv2
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
import threading
import queue

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand, 
                         VehicleLocalPosition, SensorGps, VehicleImu, 
                         VehicleAttitude, VehicleAngularVelocity, DistanceSensor)
from sensor_msgs.msg import Image as ROSImage, PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

# Computer vision and AI
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch

# Import our enhanced modules
from ai_navigator.navigation_params import (
    NavigationParams, FlightMode, TrackingMode
)

from ai_navigator.obstacle_data import ThreatLevel

from ai_navigator.obstacle_data import (
    ObstacleInfo, ObstacleCluster, ObstacleTracker, Vector3D, 
    BoundingBox3D, SensorType, ObstacleType
)
from ai_navigator.path_planner import PathPlanner, Trajectory, PlannerType
from ai_navigator.drone_state import DroneState

@dataclass
class SensorData:
    """Consolidated sensor data.

    Navigation frame convention (used throughout this node):
      LOCAL NED — matches PX4 VehicleLocalPosition output:
        +x = North  (meters)
        +y = East   (meters)
        +z = Down   (meters, negative = above ground)
      All positions, velocities, and obstacle coordinates MUST be in this frame
      before being stored here or passed to the planner.
      GPS and VIO measurements are converted to NED on ingestion.

    Body frame: x-forward, y-right, z-down  (PX4 convention).
    Camera frame: z-forward, x-right, y-down (standard pinhole).
    """
    # PX4 local NED position/velocity — primary nav source
    position: Vector3D = field(default_factory=Vector3D)
    velocity: Vector3D = field(default_factory=Vector3D)
    acceleration: Vector3D = field(default_factory=Vector3D)
    angular_velocity: Vector3D = field(default_factory=Vector3D)

    # Attitude stored as full quaternion (w, x, y, z) — PX4 VehicleAttitude.q order
    attitude_quaternion: Tuple[float, float, float, float] = field(
        default_factory=lambda: (1.0, 0.0, 0.0, 0.0))
    # Yaw derived from attitude_quaternion in attitude_callback; 0=North, +ve=clockwise (NED)
    yaw: float = 0.0

    # GPS data (converted to local NED before storage)
    gps_position: Vector3D = field(default_factory=Vector3D)
    gps_velocity: Vector3D = field(default_factory=Vector3D)
    gps_accuracy: float = 999.0
    satellites_used: int = 0
    gps_fix_type: int = 0

    # IMU data
    imu_acceleration: Vector3D = field(default_factory=Vector3D)
    imu_angular_velocity: Vector3D = field(default_factory=Vector3D)

    # VIO odometry — from step9_msckf or step10_multicam_msckf (local NED)
    vio_position: Vector3D = field(default_factory=Vector3D)
    vio_velocity: Vector3D = field(default_factory=Vector3D)
    vio_confidence: float = 0.0

    # Environmental sensors
    barometric_altitude: float = 0.0
    ground_distance: float = 0.0
    terrain_height: float = 0.0

    # Per-source timestamps for independent health checking
    timestamp: float = field(default_factory=time.time)       # generic last-update
    local_pose_timestamp: float = 0.0   # last VehicleLocalPosition msg
    vio_timestamp: float = 0.0          # last MSCKF pose msg
    gps_timestamp: float = 0.0          # last SensorGps msg
    perception_timestamp: float = 0.0  # last vision frame processed

    # Source validity flags — gated in safety_check, not just by confidence scalar
    local_pose_valid: bool = False
    gps_valid: bool = False
    vio_valid: bool = False

    sensor_health: Dict[str, float] = field(default_factory=dict)

class KalmanFilter:
    """Extended Kalman Filter for sensor fusion"""
    
    def __init__(self, state_dim: int = 9):  # [x, y, z, vx, vy, vz, ax, ay, az]
        self.state_dim = state_dim
        self.state = np.zeros(state_dim)  # State vector
        self.P = np.eye(state_dim) * 1.0  # Covariance matrix
        self.Q = np.eye(state_dim) * 0.01  # Process noise
        self.R_gps = np.eye(3) * 0.3  # GPS measurement noise
        self.R_vision = np.eye(3) * 0.05  # Vision measurement noise
        self.R_imu = np.eye(3) * 0.01  # IMU measurement noise
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(state_dim)
        self.dt = 0.01  # Will be updated with actual dt
        
    def predict(self, dt: float):
        """Prediction step"""
        self.dt = dt
        
        # Update state transition matrix for constant acceleration model
        self.F[0, 3] = dt  # x = x + vx*dt
        self.F[1, 4] = dt  # y = y + vy*dt
        self.F[2, 5] = dt  # z = z + vz*dt
        self.F[3, 6] = dt  # vx = vx + ax*dt
        self.F[4, 7] = dt  # vy = vy + ay*dt
        self.F[5, 8] = dt  # vz = vz + az*dt
        
        # Predict state
        self.state = self.F @ self.state
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update_gps(self, gps_position: Vector3D, accuracy: float):
        """Update with GPS measurement"""
        if accuracy > 5.0:  # Don't use inaccurate GPS
            return
            
        z = np.array([gps_position.x, gps_position.y, gps_position.z])
        H = np.zeros((3, self.state_dim))
        H[0, 0] = 1  # Measure x
        H[1, 1] = 1  # Measure y
        H[2, 2] = 1  # Measure z
        
        # Adjust measurement noise based on accuracy
        R = self.R_gps * (accuracy / 0.3)
        
        self._update(z, H, R)
    
    def update_vision(self, vision_position: Vector3D, confidence: float):
        """Update with vision measurement"""
        if confidence < 0.5:  # Don't use low confidence vision
            return
            
        z = np.array([vision_position.x, vision_position.y, vision_position.z])
        H = np.zeros((3, self.state_dim))
        H[0, 0] = 1  # Measure x
        H[1, 1] = 1  # Measure y
        H[2, 2] = 1  # Measure z
        
        # Adjust measurement noise based on confidence
        R = self.R_vision * (2.0 - confidence)
        
        self._update(z, H, R)
    
    def update_imu(self, imu_acceleration: Vector3D):
        """Update with IMU acceleration measurement"""
        z = np.array([imu_acceleration.x, imu_acceleration.y, imu_acceleration.z])
        H = np.zeros((3, self.state_dim))
        H[0, 6] = 1  # Measure ax
        H[1, 7] = 1  # Measure ay
        H[2, 8] = 1  # Measure az
        
        self._update(z, H, self.R_imu)
    
    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray):
        """Generic measurement update"""
        # Innovation
        y = z - H @ self.state
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P
    
    def get_position(self) -> Vector3D:
        """Get filtered position"""
        return Vector3D(self.state[0], self.state[1], self.state[2])
    
    def get_velocity(self) -> Vector3D:
        """Get filtered velocity"""
        return Vector3D(self.state[3], self.state[4], self.state[5])
    
    def get_acceleration(self) -> Vector3D:
        """Get filtered acceleration"""
        return Vector3D(self.state[6], self.state[7], self.state[8])

class PIDController:
    """Advanced PID controller with anti-windup and filtering"""
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 integral_limit: float = None, derivative_filter: float = 0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.derivative_filter = derivative_filter
        
        self.integral = 0.0
        self.last_error = 0.0
        self.last_derivative = 0.0
        self.last_time = time.time()
    
    def update(self, error: float, dt: float = None) -> float:
        """Update PID controller"""
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_time
        self.last_time = current_time
        
        if dt <= 0:
            return 0.0
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        if self.integral_limit:
            self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        integral = self.ki * self.integral
        
        # Derivative term with filtering
        derivative = (error - self.last_error) / dt
        self.last_derivative = (self.derivative_filter * derivative + 
                               (1 - self.derivative_filter) * self.last_derivative)
        derivative_term = self.kd * self.last_derivative
        
        self.last_error = error
        
        return proportional + integral + derivative_term
    
    def reset(self):
        """Reset PID controller"""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_derivative = 0.0

class PrecisionHoverController:
    """Precision hovering system with ±5cm accuracy"""
    
    def __init__(self, params: NavigationParams):
        self.params = params
        
        # Position PID controllers
        self.pid_x = PIDController(
            params.precision_hover.position_p_gain,
            params.precision_hover.position_i_gain,
            params.precision_hover.position_d_gain,
            params.precision_hover.position_i_max
        )
        self.pid_y = PIDController(
            params.precision_hover.position_p_gain,
            params.precision_hover.position_i_gain,
            params.precision_hover.position_d_gain,
            params.precision_hover.position_i_max
        )
        self.pid_z = PIDController(
            params.precision_hover.position_p_gain,
            params.precision_hover.position_i_gain,
            params.precision_hover.position_d_gain,
            params.precision_hover.position_i_max
        )
        
        # Velocity PID controllers
        self.pid_vx = PIDController(
            params.precision_hover.velocity_p_gain,
            params.precision_hover.velocity_i_gain,
            params.precision_hover.velocity_d_gain,
            params.precision_hover.velocity_i_max
        )
        self.pid_vy = PIDController(
            params.precision_hover.velocity_p_gain,
            params.precision_hover.velocity_i_gain,
            params.precision_hover.velocity_d_gain,
            params.precision_hover.velocity_i_max
        )
        self.pid_vz = PIDController(
            params.precision_hover.velocity_p_gain,
            params.precision_hover.velocity_i_gain,
            params.precision_hover.velocity_d_gain,
            params.precision_hover.velocity_i_max
        )
        
        # Attitude PID controllers
        self.pid_yaw = PIDController(
            params.precision_hover.attitude_p_gain,
            params.precision_hover.attitude_i_gain,
            params.precision_hover.attitude_d_gain
        )
        
        self.target_position = Vector3D()
        self.target_yaw = 0.0
        self.hover_start_time = None
        self.is_hovering = False
    
    def set_target(self, position: Vector3D, yaw: float = 0.0):
        """Set hover target"""
        self.target_position = position
        self.target_yaw = yaw
        self.hover_start_time = time.time()
        self.is_hovering = True
    
    def compute_control(self, current_pos: Vector3D, current_vel: Vector3D, 
                       current_yaw: float, dt: float) -> Tuple[Vector3D, float]:
        """Compute precision hover control commands"""
        
        # Position errors
        error_x = self.target_position.x - current_pos.x
        error_y = self.target_position.y - current_pos.y
        error_z = self.target_position.z - current_pos.z
        error_yaw = self._normalize_angle(self.target_yaw - current_yaw)
        
        # Position control (outer loop) - outputs desired velocities
        desired_vx = self.pid_x.update(error_x, dt)
        desired_vy = self.pid_y.update(error_y, dt)
        desired_vz = self.pid_z.update(error_z, dt)
        
        # Limit desired velocities for precision
        max_hover_vel = self.params.precision_hover.max_hover_velocity
        desired_vx = max(-max_hover_vel, min(max_hover_vel, desired_vx))
        desired_vy = max(-max_hover_vel, min(max_hover_vel, desired_vy))
        desired_vz = max(-max_hover_vel, min(max_hover_vel, desired_vz))
        
        # Velocity control (inner loop) - outputs accelerations/attitudes
        vel_error_x = desired_vx - current_vel.x
        vel_error_y = desired_vy - current_vel.y
        vel_error_z = desired_vz - current_vel.z
        
        accel_x = self.pid_vx.update(vel_error_x, dt)
        accel_y = self.pid_vy.update(vel_error_y, dt)
        accel_z = self.pid_vz.update(vel_error_z, dt)
        
        # Limit accelerations
        max_hover_accel = self.params.precision_hover.max_hover_acceleration
        accel_x = max(-max_hover_accel, min(max_hover_accel, accel_x))
        accel_y = max(-max_hover_accel, min(max_hover_accel, accel_y))
        accel_z = max(-max_hover_accel, min(max_hover_accel, accel_z))
        
        # Yaw control
        yaw_rate = self.pid_yaw.update(error_yaw, dt)
        
        # Check if we're achieving precision hover
        position_error = math.sqrt(error_x**2 + error_y**2 + error_z**2)
        velocity_magnitude = math.sqrt(current_vel.x**2 + current_vel.y**2 + current_vel.z**2)
        
        precision_achieved = (
            position_error < self.params.precision_hover.max_horizontal_error and
            abs(error_z) < self.params.precision_hover.max_vertical_error and
            velocity_magnitude < self.params.precision_hover.max_hover_velocity and
            abs(error_yaw) < self.params.precision_hover.max_yaw_error
        )
        
        return Vector3D(accel_x, accel_y, accel_z), yaw_rate, precision_achieved
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

class ActiveTrackController:
    """ActiveTrack subject following system"""
    
    def __init__(self, params: NavigationParams):
        self.params = params
        self.tracking_mode = TrackingMode.TRACE
        self.target_detected = False
        self.target_position = Vector3D()
        self.target_velocity = Vector3D()
        self.target_confidence = 0.0
        self.target_bbox = None
        self.target_lost_time = None
        
        # Tracking history for prediction
        self.position_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=5)
        
        # Tracking model (YOLO or custom tracker)
        self.tracker_id = None
        self.last_detection_time = 0.0
    
    def set_tracking_mode(self, mode: TrackingMode):
        """Set ActiveTrack mode"""
        self.tracking_mode = mode
    
    def update_target(self, detections: List[Dict], frame: np.ndarray) -> bool:
        """Update target with new detections"""
        current_time = time.time()
        
        if not detections:
            if self.target_detected:
                if self.target_lost_time is None:
                    self.target_lost_time = current_time
                elif current_time - self.target_lost_time > self.params.active_track.subject_lost_timeout:
                    self.target_detected = False
                    self.target_lost_time = None
            return False
        
        # For simplicity, track highest confidence detection
        # In production, use proper multi-object tracking
        best_detection = max(detections, key=lambda x: x.get('conf', 0.0))
        
        if best_detection['conf'] >= self.params.active_track.detection_confidence:
            self.target_detected = True
            self.target_lost_time = None
            self.target_confidence = best_detection['conf']
            
            # Extract 3D position (assuming depth available)
            if 'world_x' in best_detection and 'world_y' in best_detection:
                new_position = Vector3D(
                    best_detection['world_x'],
                    best_detection['world_y'],
                    best_detection.get('world_z', 0.0)
                )
                
                # Update position history
                self.position_history.append((current_time, new_position))
                
                # Estimate velocity
                if len(self.position_history) >= 2:
                    dt = self.position_history[-1][0] - self.position_history[-2][0]
                    if dt > 0:
                        dp = self.position_history[-1][1] - self.position_history[-2][1]
                        velocity = Vector3D(dp.x / dt, dp.y / dt, dp.z / dt)
                        
                        # Smooth velocity estimate
                        if self.velocity_history:
                            prev_vel = self.velocity_history[-1]
                            smoothing = self.params.active_track.velocity_smoothing
                            velocity = Vector3D(
                                smoothing * prev_vel.x + (1 - smoothing) * velocity.x,
                                smoothing * prev_vel.y + (1 - smoothing) * velocity.y,
                                smoothing * prev_vel.z + (1 - smoothing) * velocity.z
                            )
                        
                        self.velocity_history.append(velocity)
                        self.target_velocity = velocity
                
                self.target_position = new_position
                self.last_detection_time = current_time
                return True
        
        return False
    
    def compute_follow_position(self, drone_position: Vector3D) -> Vector3D:
        """Compute desired drone position for current tracking mode"""
        if not self.target_detected:
            return drone_position
        
        tracking_params = self.params.get_tracking_params()
        
        # Predict target future position
        prediction_time = self.params.active_track.prediction_horizon
        predicted_position = self.target_position
        
        if self.velocity_history:
            predicted_position = Vector3D(
                self.target_position.x + self.target_velocity.x * prediction_time,
                self.target_position.y + self.target_velocity.y * prediction_time,
                self.target_position.z + self.target_velocity.z * prediction_time
            )
        
        if self.tracking_mode == TrackingMode.SPOTLIGHT:
            # Stay in place, just point at target
            return drone_position
        
        elif self.tracking_mode == TrackingMode.PROFILE:
            # Follow alongside at specified distance and height
            follow_distance = tracking_params['follow_distance']
            height_offset = tracking_params['follow_height_offset']
            side_offset = tracking_params['side_offset']
            
            # Calculate position to the side of target movement direction
            if self.target_velocity.magnitude() > 0.5:
                # Move to the side of target's movement
                movement_dir = self.target_velocity.normalize()
                side_dir = Vector3D(-movement_dir.y, movement_dir.x, 0).normalize()
                
                follow_pos = Vector3D(
                    predicted_position.x + side_dir.x * side_offset,
                    predicted_position.y + side_dir.y * side_offset,
                    predicted_position.z + height_offset
                )
            else:
                # Maintain current relative position
                follow_pos = Vector3D(
                    predicted_position.x + side_offset,
                    predicted_position.y,
                    predicted_position.z + height_offset
                )
            
            return follow_pos
        
        elif self.tracking_mode == TrackingMode.TRACE:
            # Follow behind at specified distance
            follow_distance = tracking_params['follow_distance']
            height_offset = tracking_params['follow_height_offset']
            
            if self.target_velocity.magnitude() > 0.5:
                # Follow behind target's movement direction
                movement_dir = self.target_velocity.normalize()
                follow_pos = Vector3D(
                    predicted_position.x - movement_dir.x * follow_distance,
                    predicted_position.y - movement_dir.y * follow_distance,
                    predicted_position.z + height_offset
                )
            else:
                # Maintain distance if target is stationary
                follow_pos = Vector3D(
                    predicted_position.x - follow_distance,
                    predicted_position.y,
                    predicted_position.z + height_offset
                )
            
            return follow_pos
        
        elif self.tracking_mode == TrackingMode.CIRCLE:
            # Circle around target
            radius = tracking_params['circle_radius']
            height = tracking_params['circle_height']
            speed = tracking_params['circle_speed']
            
            # Calculate angle based on time
            angle = (time.time() * speed) % (2 * math.pi)
            
            circle_pos = Vector3D(
                predicted_position.x + radius * math.cos(angle),
                predicted_position.y + radius * math.sin(angle),
                predicted_position.z + height
            )
            
            return circle_pos
        
        elif self.tracking_mode == TrackingMode.HELIX:
            # Spiral around target with altitude change
            radius = tracking_params['helix_radius']
            pitch = tracking_params['helix_pitch']
            speed = tracking_params['helix_speed']
            
            # Calculate angle and height based on time
            angle = (time.time() * speed) % (2 * math.pi)
            height_change = (angle / (2 * math.pi)) * pitch
            
            helix_pos = Vector3D(
                predicted_position.x + radius * math.cos(angle),
                predicted_position.y + radius * math.sin(angle),
                predicted_position.z + height_change
            )
            
            return helix_pos
        
        # Default: maintain current position
        return drone_position
    
    def get_gimbal_target(self) -> Optional[Vector3D]:
        """Get target position for gimbal pointing"""
        if self.target_detected:
            return self.target_position
        return None

class TerrainFollowController:
    """Terrain following system"""
    
    def __init__(self, params: NavigationParams):
        self.params = params
        self.terrain_data = {}  # Grid of terrain heights
        self.ground_distance = 0.0
        self.terrain_slope = 0.0
        self.follow_height = 3.0  # Desired height above ground
        
    def update_terrain_data(self, lidar_points: np.ndarray):
        """Update terrain map from LIDAR data"""
        # Process LIDAR points to create terrain map
        # This is a simplified version - in production use proper SLAM
        if len(lidar_points) == 0:
            return
        
        # Extract ground points (assuming ground is lowest points in each grid cell)
        grid_size = 1.0  # 1 meter grid
        grid_data = defaultdict(list)
        
        for point in lidar_points:
            if len(point) >= 3:
                x, y, z = point[:3]
                grid_x = int(x / grid_size)
                grid_y = int(y / grid_size)
                grid_data[(grid_x, grid_y)].append(z)
        
        # Update terrain heights (use lowest point in each grid cell)
        for (gx, gy), heights in grid_data.items():
            if heights:
                terrain_height = min(heights)  # Ground is lowest point
                self.terrain_data[(gx, gy)] = terrain_height
    
    def get_terrain_height(self, position: Vector3D) -> float:
        """Get terrain height at position"""
        grid_size = 1.0
        grid_x = int(position.x / grid_size)
        grid_y = int(position.y / grid_size)
        
        # Bilinear interpolation from surrounding grid points
        heights = []
        for dx in [0, 1]:
            for dy in [0, 1]:
                key = (grid_x + dx, grid_y + dy)
                if key in self.terrain_data:
                    heights.append(self.terrain_data[key])
        
        if heights:
            return sum(heights) / len(heights)
        return 0.0  # Unknown terrain
    
    def compute_follow_height(self, position: Vector3D, velocity: Vector3D) -> float:
        """Compute desired altitude for terrain following"""
        # Get current terrain height
        terrain_height = self.get_terrain_height(position)
        
        # Look ahead based on velocity
        lookahead_distance = self.params.terrain_follow.terrain_prediction_distance
        if velocity.magnitude() > 0:
            lookahead_time = lookahead_distance / velocity.magnitude()
            future_pos = Vector3D(
                position.x + velocity.x * lookahead_time,
                position.y + velocity.y * lookahead_time,
                position.z
            )
            future_terrain_height = self.get_terrain_height(future_pos)
        else:
            future_terrain_height = terrain_height
        
        # Calculate desired altitude (relative to terrain)
        target_terrain_height = max(terrain_height, future_terrain_height)
        desired_altitude = target_terrain_height + self.follow_height
        
        # Apply safety margins
        min_clearance = self.params.terrain_follow.min_terrain_clearance
        max_clearance = self.params.terrain_follow.max_terrain_clearance
        
        desired_altitude = max(desired_altitude, target_terrain_height + min_clearance)
        desired_altitude = min(desired_altitude, target_terrain_height + max_clearance)
        
        return desired_altitude
    
    def is_terrain_suitable(self, position: Vector3D) -> bool:
        """Check if terrain is suitable for following"""
        # Check slope
        grid_size = 1.0
        surrounding_heights = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_pos = Vector3D(position.x + dx * grid_size, 
                                   position.y + dy * grid_size, position.z)
                height = self.get_terrain_height(check_pos)
                surrounding_heights.append(height)
        
        if len(surrounding_heights) < 4:
            return False  # Insufficient data
        
        # Calculate maximum slope
        min_height = min(surrounding_heights)
        max_height = max(surrounding_heights)
        max_slope = (max_height - min_height) / (grid_size * math.sqrt(2))
        
        return max_slope <= self.params.terrain_follow.terrain_slope_limit

class SmartNavigator(Node):
    """Enhanced autonomous navigation node."""
    
    def __init__(self):
        super().__init__('smart_navigator')
        
        # Initialize parameters and subsystems
        self.nav_params = NavigationParams()
        self.path_planner = PathPlanner(self.nav_params)
        
        # Sensor fusion and state estimation
        self.kalman_filter = KalmanFilter()
        self.sensor_data = SensorData()
        self.fused_position = Vector3D()
        self.fused_velocity = Vector3D()
        
        # Advanced controllers
        self.precision_hover = PrecisionHoverController(self.nav_params)
        self.active_track = ActiveTrackController(self.nav_params)
        self.terrain_follow = TerrainFollowController(self.nav_params)
        
        # Obstacle detection and tracking
        self.obstacle_tracker = ObstacleTracker()
        self.obstacle_info = ObstacleInfo()
        
        # Communication setup
        self._setup_qos_profiles()
        self._setup_publishers()
        self._setup_subscribers()
        print("#########################################################")
        
        # Initialize state
        self._initialize_enhanced_state()
        self._setup_enhanced_computer_vision()
        
        # Multi-threaded processing
        self._setup_processing_threads()
        
        # Main control timer (100 Hz for precision)
        self.control_timer = self.create_timer(0.01, self.control_loop)
        
        self.get_logger().info("Enhanced SmartNavigator initialized")
    
    def _setup_qos_profiles(self):
        """Setup QoS profiles for different data types"""
        # High-frequency control data
        self.control_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Sensor data
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        # Navigation data
        self.nav_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
    
    def _setup_publishers(self):
        """Setup enhanced publishers"""
        # Control outputs
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', self.control_qos)
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', self.control_qos)
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', self.control_qos)
        
        # Enhanced telemetry
        self.nav_status_pub = self.create_publisher(
            Float32MultiArray, '/nav/status', self.nav_qos)
        self.sensor_fusion_pub = self.create_publisher(
            Float32MultiArray, '/nav/sensor_fusion', self.nav_qos)
    
    def _setup_subscribers(self):
        """Setup enhanced subscribers"""
        # Vehicle state
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',
            self.position_callback, self.sensor_qos)
        self.attitude_sub = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude',
            self.attitude_callback, self.sensor_qos)
        self.angular_velocity_sub = self.create_subscription(
            VehicleAngularVelocity, '/fmu/out/vehicle_angular_velocity',
            self.angular_velocity_callback, self.sensor_qos)
        
        # Sensors
        self.gps_sub = self.create_subscription(
            SensorGps, '/fmu/out/vehicle_gps_position',
            self.gps_callback, self.sensor_qos)
        self.imu_sub = self.create_subscription(
            VehicleImu, '/fmu/out/vehicle_imu',
            self.imu_callback, self.sensor_qos)
        self.distance_sensor_sub = self.create_subscription(
            DistanceSensor, '/fmu/out/distance_sensor',
            self.distance_sensor_callback, self.sensor_qos)
        
        # Vision and perception
        # RGB camera from x500_depth model (bridged by PX4 automatically)
        self.image_sub = self.create_subscription(
            ROSImage, '/camera/image',
            self.image_callback, self.sensor_qos)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info',
            self.camera_info_callback, self.sensor_qos)

        # Depth camera (requires manual bridge - see launch instructions)
        self.depth_sub = self.create_subscription(
            ROSImage, '/depth_camera', self.depth_callback, self.sensor_qos)

        # LIDAR (requires manual bridge - see launch instructions)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/lidar', self.lidar_callback, self.sensor_qos)

        # VIO odometry — step9_msckf publishes nav_msgs/Odometry on /msckf_vio/odometry
        # step10_multicam_msckf publishes on /multicam_msckf/odometry
        # The ROS parameter 'vio_topic' selects which to use (default: msckf mono)
        vio_topic = self.declare_parameter('vio_topic', '/msckf_vio/odometry').value
        self.vio_sub = self.create_subscription(
            Odometry, vio_topic,
            self.vio_callback, self.sensor_qos)
        self.get_logger().info(f"VIO subscriber: {vio_topic}")

        # Primary pose source — set via ROS parameter at launch time:
        #   px4_local  : use PX4 EKF2 VehicleLocalPosition (safe default)
        #   msckf      : use step9 MSCKF odometry on vio_topic
        # Example: ros2 run ai_navigator smart_navigator --ros-args -p primary_pose_source:=msckf
        self.primary_pose_source = self.declare_parameter(
            'primary_pose_source', 'px4_local').value
        self.get_logger().info(f"Primary pose source: {self.primary_pose_source}")

        # VIO freshness timeout — separate from GPS timeout because VIO runs at ~30 Hz
        # and should be declared stale much faster than a 1–5 Hz GPS signal.
        self.vio_timeout = self.declare_parameter('vio_timeout', 1.0).value
        self.local_pose_timeout = self.declare_parameter(
            'local_pose_timeout',
            self.nav_params.safety.gps_timeout
        ).value
    
    def _initialize_enhanced_state(self):
        """Initialize enhanced state variables"""
        # Flight state
        self.state = DroneState.INIT
        self.flight_mode = FlightMode.POSITION
        self.tracking_mode = TrackingMode.TRACE
        
        # Mission parameters
        self.takeoff_point = Vector3D(0.0, 0.0, self.nav_params.takeoff_altitude)
        self.destination_point = Vector3D(0.0, self.nav_params.destination_offset, 
                                        self.nav_params.takeoff_altitude)
        self.home_position = Vector3D()
        self.home_position_initialized = False  # set True once first local pose arrives

        # Current trajectory
        self.current_trajectory: Optional[Trajectory] = None
        self.trajectory_index = 0
        self.trajectory_start_time = 0.0
        self.planning_request_in_flight = False
        self.last_planning_request_time = 0.0
        self.planning_request_interval = 0.25
        self.camera_intrinsics: Optional[Dict[str, float]] = None
        
        # Safety and monitoring
        self.safety_violations = []
        self.performance_metrics = {}
        self.last_control_update = time.time()
        
        # Threading
        self.processing_queues = {
            'vision': queue.Queue(maxsize=10),
            'lidar': queue.Queue(maxsize=10),
            'planning': queue.Queue(maxsize=5)
        }
        
        # Counters and timers
        self.offboard_setpoint_counter = 0
        self.control_loop_count = 0
        self.emergency_stop = False
        
        self.get_logger().info("Enhanced state initialized successfully")
    
    def _setup_enhanced_computer_vision(self):
        """Setup enhanced computer vision system"""
        self.bridge = CvBridge()
        self.depth_image: Optional[np.ndarray] = None
        
        # YOLO models for different detection tasks
        try:
            # Main object detection model
            self.detection_model = YOLO("yolo11s.pt")
            
            # Specialized models (would be loaded in production)
            # self.person_model = YOLO("yolo11s-person.pt")
            # self.vehicle_model = YOLO("yolo11s-vehicle.pt")
            
            self.get_logger().info("Enhanced computer vision models loaded successfully")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load vision models: {e}")
            self.state = DroneState.EMERGENCY
    
    def _setup_processing_threads(self):
        """Setup multi-threaded processing"""
        # Vision processing thread
        self.vision_thread = threading.Thread(target=self._vision_processing_loop)
        self.vision_thread.daemon = True
        self.vision_thread.start()
        
        # LIDAR processing thread
        self.lidar_thread = threading.Thread(target=self._lidar_processing_loop)
        self.lidar_thread.daemon = True
        self.lidar_thread.start()
        
        # Path planning thread
        self.planning_thread = threading.Thread(target=self._planning_processing_loop)
        self.planning_thread.daemon = True
        self.planning_thread.start()
        
        self.get_logger().info("Processing threads started successfully")
    
    # Callback methods for sensor data
    def position_callback(self, msg):
        """Enhanced position callback with sensor fusion"""
        self.sensor_data.position = Vector3D(msg.x, msg.y, msg.z)
        self.sensor_data.velocity = Vector3D(msg.vx, msg.vy, msg.vz)
        now = time.time()
        self.sensor_data.timestamp = now
        self.sensor_data.local_pose_timestamp = now
        self.sensor_data.local_pose_valid = True

        # Record home position once from first valid local NED reading
        if not self.home_position_initialized:
            self.home_position = Vector3D(msg.x, msg.y, msg.z)
            self.home_position_initialized = True
            self.get_logger().info(
                f"Home position set: ({msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f}) NED"
            )
        
        # Update fused position based on authoritative source
        if self.primary_pose_source == 'msckf' and self.sensor_data.vio_valid:
            # VIO is primary — both position and velocity come from vio_callback
            # Nothing to do here; vio_callback already updated fused_position/velocity
            pass
        else:
            # px4_local is primary — use PX4 EKF2 position directly
            self.fused_position = self.sensor_data.position
            self.fused_velocity = self.sensor_data.velocity
            # Also feed Kalman filter as secondary smoother
            dt = time.time() - self.last_control_update
            if dt > 0:
                self.kalman_filter.predict(dt)
            if self.sensor_data.vio_valid:
                self.kalman_filter.update_vision(
                    self.sensor_data.vio_position,
                    self.sensor_data.vio_confidence
                )
    
    def gps_callback(self, msg):
        """GPS data callback"""
        # Update GPS sensor data using local NED position from VehicleLocalPosition,
        # not raw GPS lat/lon. Home position is set once from the first local position
        # reading in position_callback so it is already in metric NED frame.

        # Convert GPS to local NED for Kalman filter (rough approximation only —
        # primary position source is VehicleLocalPosition, not this).
        if hasattr(self, '_home_gps_lat'):
            lat0, lon0, alt0 = self._home_gps_lat, self._home_gps_lon, self._home_gps_alt
        else:
            lat0 = msg.latitude_deg
            lon0 = msg.longitude_deg
            alt0 = msg.altitude_msl_m
            self._home_gps_lat = lat0
            self._home_gps_lon = lon0
            self._home_gps_alt = alt0

        north = (msg.latitude_deg - lat0) * 111320.0
        east  = (msg.longitude_deg - lon0) * 111320.0 * math.cos(math.radians(lat0))
        down  = -(msg.altitude_msl_m - alt0)

        self.sensor_data.gps_position = Vector3D(north, east, down
        )
        self.sensor_data.gps_accuracy = msg.eph
        self.sensor_data.satellites_used = msg.satellites_used
        self.sensor_data.gps_timestamp = time.time()
        self.sensor_data.gps_valid = (
            msg.satellites_used >= self.nav_params.sensor_fusion.gps_min_satellites
            and msg.eph < self.nav_params.sensor_fusion.gps_hdop_threshold
        )

        # Feed Kalman filter secondary smoother with GPS
        if self.sensor_data.gps_valid:
            self.kalman_filter.update_gps(self.sensor_data.gps_position, msg.eph)
    
    def imu_callback(self, msg):
        """IMU data callback"""
        self.sensor_data.imu_acceleration = Vector3D(
            msg.delta_vel_dt[0], msg.delta_vel_dt[1], msg.delta_vel_dt[2])
        self.sensor_data.imu_angular_velocity = Vector3D(
            msg.delta_angle_dt[0], msg.delta_angle_dt[1], msg.delta_angle_dt[2])
        
        # Update Kalman filter with IMU
        self.kalman_filter.update_imu(self.sensor_data.imu_acceleration)
    
    def attitude_callback(self, msg):
        """Store full attitude quaternion and derive yaw.

        PX4 VehicleAttitude.q = [w, x, y, z].
        """
        from scipy.spatial.transform import Rotation as R
        w, x, y, z = msg.q[0], msg.q[1], msg.q[2], msg.q[3]
        self.sensor_data.attitude_quaternion = (w, x, y, z)
        # scipy expects [x, y, z, w]
        rot = R.from_quat([x, y, z, w])
        # ZYX Euler in NED: yaw is rotation about z-down axis
        _, _, yaw = rot.as_euler('xyz', degrees=False)
        self.sensor_data.yaw = yaw
    
    def angular_velocity_callback(self, msg):
        """Angular velocity callback"""
        self.sensor_data.angular_velocity = Vector3D(msg.xyz[0], msg.xyz[1], msg.xyz[2])
    
    def vio_callback(self, msg: Odometry):
        """MSCKF VIO odometry callback — reads pose AND twist from nav_msgs/Odometry.

        step9_msckf publishes on /msckf_vio/odometry (Odometry, frame: world/imu).
        Position/velocity are in the same local frame as VehicleLocalPosition
        (gravity-bug fixed in commit a4a62f2 — NED-aligned at rest).

        Confidence is estimated from position covariance trace: low trace → high conf.
        """
        now = time.time()

        # Pose
        p = msg.pose.pose.position
        self.sensor_data.vio_position = Vector3D(p.x, p.y, p.z)

        # Twist (linear velocity in the odometry child frame)
        v = msg.twist.twist.linear
        self.sensor_data.vio_velocity = Vector3D(v.x, v.y, v.z)

        # Confidence from position covariance trace (lower = more confident)
        cov_trace = msg.pose.covariance[0] + msg.pose.covariance[7] + msg.pose.covariance[14]
        # Map: trace 0→conf 1.0, trace 10→conf 0.0, clamp to [0,1]
        self.sensor_data.vio_confidence = max(0.0, min(1.0, 1.0 - cov_trace / 10.0))

        self.sensor_data.vio_timestamp = now
        self.sensor_data.vio_valid = self.sensor_data.vio_confidence > 0.3

        # If VIO is the primary source, update fused pose + velocity immediately
        if self.primary_pose_source == 'msckf' and self.sensor_data.vio_valid:
            self.fused_position = self.sensor_data.vio_position
            self.fused_velocity = self.sensor_data.vio_velocity

    def distance_sensor_callback(self, msg):
        """Distance sensor callback"""
        if msg.orientation == 8:  # Downward facing
            self.sensor_data.ground_distance = msg.current_distance
            self.terrain_follow.ground_distance = msg.current_distance
    
    def image_callback(self, msg):
        """Enhanced image processing callback"""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Queue for asynchronous processing
            if not self.processing_queues['vision'].full():
                self.processing_queues['vision'].put(frame)
                
        except Exception as e:
            self.get_logger().error(f"Image processing failed: {e}")

    def camera_info_callback(self, msg: CameraInfo):
        """Store live camera intrinsics for pixel-to-world conversion."""
        if len(msg.k) < 9:
            return

        self.camera_intrinsics = {
            'fx': float(msg.k[0]),
            'fy': float(msg.k[4]),
            'cx': float(msg.k[2]),
            'cy': float(msg.k[5]),
            'width': int(msg.width),
            'height': int(msg.height),
        }
    
    def depth_callback(self, msg):
        """Depth camera callback"""
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_image = np.array(depth_image, dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"Depth callback failed: {e}")
    
    def lidar_callback(self, msg):
        """LIDAR point cloud callback"""
        try:
            # Convert PointCloud2 to numpy array (simplified)
            # In production, use proper point cloud libraries
            if not self.processing_queues['lidar'].full():
                self.processing_queues['lidar'].put(msg)
        except Exception as e:
            self.get_logger().warn(f"LIDAR callback failed: {e}")

    def lidar_down_callback(self, msg):
        """Downward LIDAR callback for ground distance measurement"""
        try:
            # Extract ground distance from downward LIDAR
            # The downward LIDAR provides a single range measurement
            # We'll extract the minimum distance as ground distance
            import struct

            # PointCloud2 structure: fields contain x, y, z data
            # For a single-point LIDAR, we just need to extract the distance
            if msg.width > 0 and msg.height > 0:
                # Simple extraction - assumes first point contains distance in z
                point_step = msg.point_step
                if len(msg.data) >= point_step:
                    # Extract x, y, z from first point (assuming float32 format)
                    x = struct.unpack('f', msg.data[0:4])[0]
                    y = struct.unpack('f', msg.data[4:8])[0]
                    z = struct.unpack('f', msg.data[8:12])[0]

                    # Calculate distance (Euclidean)
                    distance = math.sqrt(x*x + y*y + z*z)

                    # Update sensor data
                    self.sensor_data.ground_distance = distance
                    self.terrain_follow.ground_distance = distance

                    # Also update terrain height estimate
                    self.sensor_data.terrain_height = self.fused_position.z + distance

        except Exception as e:
            self.get_logger().debug(f"Downward LIDAR callback failed: {e}")
    
    def _vision_processing_loop(self):
        """Asynchronous vision processing thread"""
        while True:
            try:
                frame = self.processing_queues['vision'].get(timeout=1.0)
                
                # Process frame for obstacles and tracking
                obstacle_info = self._process_frame_enhanced(frame)

                # Update obstacle tracker and stamp perception health
                self.obstacle_info = obstacle_info
                self.sensor_data.perception_timestamp = time.time()
                
                # ActiveTrack processing
                if self.flight_mode == FlightMode.ACTIVETRACK:
                    detections = self._extract_detections_for_tracking(obstacle_info)
                    self.active_track.update_target(detections, frame)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Vision processing error: {e}")
    
    def _lidar_processing_loop(self):
        """Asynchronous LIDAR processing thread"""
        while True:
            try:
                pointcloud_msg = self.processing_queues['lidar'].get(timeout=1.0)

                points = np.asarray(
                    list(pc2.read_points(
                        pointcloud_msg,
                        field_names=('x', 'y', 'z'),
                        skip_nans=True
                    )),
                    dtype=np.float32
                )
                self.terrain_follow.update_terrain_data(points)
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"LIDAR processing error: {e}")
            finally:
                if 'pointcloud_msg' in locals():
                    self.processing_queues['lidar'].task_done()
                    del pointcloud_msg
    
    def _planning_processing_loop(self):
        """Asynchronous path planning thread"""
        while True:
            try:
                planning_request = self.processing_queues['planning'].get(timeout=1.0)
                
                # Process path planning request
                start_pos = planning_request['start']
                goal_pos = planning_request['goal']
                obstacle_info = planning_request['obstacles']
                
                # Plan path
                path = self.path_planner.plan_avoidance_path(
                    [start_pos.x, start_pos.y, start_pos.z],
                    [goal_pos.x, goal_pos.y, goal_pos.z],
                    obstacle_info
                )
                
                # Generate trajectory
                if path:
                    waypoints = [Vector3D(wp[0], wp[1], wp[2]) for wp in path]
                    trajectory = self.path_planner.generate_trajectory(
                        waypoints, 
                        self.nav_params.max_speed,
                        self.nav_params.max_acceleration
                    )
                    
                    # Update current trajectory
                    self.current_trajectory = trajectory
                    self.trajectory_index = 0
                    self.trajectory_start_time = time.time()
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Planning processing error: {e}")
            finally:
                if 'planning_request' in locals():
                    self.planning_request_in_flight = False
                    self.processing_queues['planning'].task_done()
                    del planning_request
    
    def control_loop(self):
        """Main control loop at 100Hz"""
        try:
            current_time = time.time()
            dt = current_time - self.last_control_update
            self.last_control_update = current_time
            self.control_loop_count += 1
            
            # Safety checks
            if not self._enhanced_safety_check():
                return
            
            # Publish offboard control mode
            self.publish_offboard_control_mode()
            
            # State machine handling
            self._handle_enhanced_state_machine(dt)
            
            # Increment counter for initial setup
            if self.offboard_setpoint_counter < 100:
                self.offboard_setpoint_counter += 1
            
            # Performance monitoring
            if self.control_loop_count % 100 == 0:  # Every second
                self._update_performance_metrics()
                
        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
            self.state = DroneState.EMERGENCY
    
    # States in which a GPS loss should trigger GPS_DENIED navigation
    _GPS_DENIED_ELIGIBLE = frozenset({
        DroneState.MOVE, DroneState.AVOIDING, DroneState.RETREATING,
        DroneState.CIRCLING, DroneState.HOLD,
    })

    def _enhanced_safety_check(self) -> bool:
        """Per-source health checks with graduated degradation."""
        now = time.time()

        # --- Local pose timeout → emergency (no position at all)
        if (self.sensor_data.local_pose_valid and
                now - self.sensor_data.local_pose_timestamp > self.local_pose_timeout):
            self.sensor_data.local_pose_valid = False
            self.get_logger().error("Local pose timeout — EMERGENCY")
            self.state = DroneState.EMERGENCY
            return False

        # --- GPS timeout → mark invalid; transition to GPS_DENIED if actively flying
        if (self.sensor_data.gps_valid and
                now - self.sensor_data.gps_timestamp > self.nav_params.safety.gps_timeout):
            self.sensor_data.gps_valid = False
            self.get_logger().warn("GPS timeout — GPS source marked invalid")
            if self.state in self._GPS_DENIED_ELIGIBLE:
                self.get_logger().warn("Entering GPS_DENIED state — navigating on VIO/EKF")
                self.state = DroneState.GPS_DENIED

        # --- VIO timeout → mark invalid (planning continues on px4_local)
        if (self.sensor_data.vio_valid and
                now - self.sensor_data.vio_timestamp > self.vio_timeout):
            self.sensor_data.vio_valid = False
            if self.primary_pose_source == 'msckf':
                self.primary_pose_source = 'px4_local'
                self.get_logger().warn("VIO timeout — falling back to px4_local")

        # --- Perception timeout → clear stale obstacles
        if (self.sensor_data.perception_timestamp > 0 and
                now - self.sensor_data.perception_timestamp > 2.0):
            self.obstacle_info = ObstacleInfo()
            self.get_logger().warn("Perception timeout — obstacle map cleared")

        # --- Emergency collision threat
        if (self.obstacle_info.detected and
                self.obstacle_info.max_threat_level >= ThreatLevel.EMERGENCY):
            self.emergency_stop = True
            self.get_logger().warn("EMERGENCY COLLISION THREAT DETECTED!")

        # --- Geofence: distance from home (only when home is known)
        if self.home_position_initialized:
            home_dist = self.fused_position.distance_to(self.home_position)
            if home_dist > self.nav_params.safety.max_distance:
                self.get_logger().warn(f"Geofence breach ({home_dist:.0f}m) — RTH")
                self.state = DroneState.RTH

        # --- Altitude limit (NED: z negative is up, so check abs)
        if abs(self.fused_position.z) > self.nav_params.safety.max_altitude:
            self.get_logger().warn("Altitude limit exceeded — stopping climb")
            self.emergency_stop = True

        return True
    
    def _handle_enhanced_state_machine(self, dt: float):
        """Enhanced state machine with advanced flight modes"""
        
        if self.state == DroneState.INIT:
            self._handle_init_state()
        elif self.state == DroneState.TAKEOFF:
            self._handle_takeoff_state()
        elif self.state == DroneState.MOVE:
            self._handle_enhanced_move_state(dt)
        elif self.state == DroneState.AVOIDING:
            self._handle_enhanced_avoiding_state(dt)
        elif self.state == DroneState.HOLD:
            self._handle_precision_hover_state(dt)
        elif self.state == DroneState.LANDING:
            self._handle_precision_landing_state(dt)
        elif self.state == DroneState.EMERGENCY:
            self._handle_emergency_state()
        elif self.state == DroneState.RTH:
            self._handle_rth_state(dt)
        elif self.state == DroneState.GPS_DENIED:
            self._handle_gps_denied_state(dt)
        # ACTIVETRACK and TERRAIN_FOLLOW are handled inside _handle_enhanced_move_state
        # via self.flight_mode — they are not lifecycle states.
    
    def _handle_enhanced_move_state(self, dt: float):
        """Enhanced movement with trajectory following"""
        if self.emergency_stop:
            self._execute_precision_hover(self.fused_position, 0.0, dt)
            return
        
        # Flight mode specific behavior
        if self.flight_mode == FlightMode.ACTIVETRACK:
            self._handle_activetrack_state(dt)
            return
        elif self.flight_mode == FlightMode.TERRAIN_FOLLOW:
            self._handle_terrain_follow_state(dt)
            return
        
        # Standard trajectory following
        if self.current_trajectory:
            self._follow_trajectory(dt)
        else:
            # Request new path planning
            self._request_path_planning()
            
            # Fly direct to destination in the meantime
            self._fly_to_position(self.destination_point, dt)
    
    def _handle_precision_hover_state(self, dt: float):
        """Precision hovering state"""
        target_pos = self.destination_point
        target_yaw = 0.0
        
        # Set hover target if not already set
        if not self.precision_hover.is_hovering:
            self.precision_hover.set_target(target_pos, target_yaw)
        
        # Execute precision hover
        self._execute_precision_hover(target_pos, target_yaw, dt)
    
    def _handle_activetrack_state(self, dt: float):
        """ActiveTrack state"""
        if self.active_track.target_detected:
            # Compute follow position
            follow_pos = self.active_track.compute_follow_position(self.fused_position)
            
            # Fly to follow position
            self._fly_to_position(follow_pos, dt)
            
            # Update gimbal pointing (if available)
            gimbal_target = self.active_track.get_gimbal_target()
            if gimbal_target:
                self._point_gimbal_at_target(gimbal_target)
        else:
            # Search for target or hover
            self._execute_precision_hover(self.fused_position, 0.0, dt)
    
    def _handle_terrain_follow_state(self, dt: float):
        """Terrain following state"""
        # Get desired altitude from terrain follower
        desired_altitude = self.terrain_follow.compute_follow_height(
            self.fused_position, self.fused_velocity)
        
        # Check if terrain is suitable
        if self.terrain_follow.is_terrain_suitable(self.fused_position):
            target_pos = Vector3D(
                self.destination_point.x,
                self.destination_point.y,
                desired_altitude
            )
            self._fly_to_position(target_pos, dt)
        else:
            # Terrain not suitable - climb to safe altitude
            safe_altitude = self.fused_position.z - 5.0  # 5 meters higher
            target_pos = Vector3D(
                self.fused_position.x,
                self.fused_position.y,
                safe_altitude
            )
            self._fly_to_position(target_pos, dt)
    
    def _handle_rth_state(self, dt: float):
        """Return to Home state"""
        home_distance = self.fused_position.distance_to(self.home_position)
        
        if home_distance < self.nav_params.arrival_threshold:
            self.state = DroneState.LANDING
        else:
            # Fly home at safe altitude
            rth_altitude = max(self.home_position.z - 10.0, self.fused_position.z)
            target_pos = Vector3D(
                self.home_position.x,
                self.home_position.y,
                rth_altitude
            )
            self._fly_to_position(target_pos, dt)
    
    def _execute_precision_hover(self, target_pos: Vector3D, target_yaw: float, dt: float):
        """Execute precision hovering control"""
        current_yaw = self.sensor_data.yaw
        
        # Compute precision control
        control_acc, yaw_rate, precision_achieved = self.precision_hover.compute_control(
            self.fused_position, self.fused_velocity, current_yaw, dt)
        
        # Convert acceleration to trajectory setpoint
        # In a real system, this would be converted to attitude commands
        next_pos = Vector3D(
            self.fused_position.x + self.fused_velocity.x * dt + 0.5 * control_acc.x * dt * dt,
            self.fused_position.y + self.fused_velocity.y * dt + 0.5 * control_acc.y * dt * dt,
            self.fused_position.z + self.fused_velocity.z * dt + 0.5 * control_acc.z * dt * dt
        )
        
        self.publish_setpoint(next_pos.x, next_pos.y, next_pos.z, target_yaw)
        
        if precision_achieved:
            self.get_logger().info("Precision hover achieved (±5cm accuracy)")
    
    def _fly_to_position(self, target_pos: Vector3D, dt: float):
        """Fly to target position with current flight mode constraints"""
        # Get performance limits for current flight mode
        limits = self.nav_params.get_current_performance_limits()
        
        # Calculate direction and distance
        direction = target_pos - self.fused_position
        distance = direction.magnitude()
        
        if distance < self.nav_params.arrival_threshold:
            self.state = DroneState.HOLD
            return
        
        # Normalize direction
        if distance > 0:
            direction = direction * (1.0 / distance)
        
        # Calculate desired velocity based on distance and flight mode
        max_speed = limits['max_speed']
        if distance < 5.0:  # Slow down when approaching
            max_speed *= (distance / 5.0)
        
        desired_velocity = direction * max_speed
        
        # Apply smoothing based on flight mode
        smoothing = limits['smoothing_factor']
        if hasattr(self, 'last_desired_velocity'):
            desired_velocity = Vector3D(
                smoothing * self.last_desired_velocity.x + (1 - smoothing) * desired_velocity.x,
                smoothing * self.last_desired_velocity.y + (1 - smoothing) * desired_velocity.y,
                smoothing * self.last_desired_velocity.z + (1 - smoothing) * desired_velocity.z
            )
        self.last_desired_velocity = desired_velocity
        
        # Calculate next position and publish with velocity feedforward
        next_pos = self.fused_position + desired_velocity * dt

        self.publish_setpoint(
            next_pos.x, next_pos.y, next_pos.z,
            self.sensor_data.yaw,            # use actual yaw, not hardcoded 0
            vx=desired_velocity.x,
            vy=desired_velocity.y,
            vz=desired_velocity.z
        )
    
    def _follow_trajectory(self, dt: float):
        """Follow current trajectory"""
        if not self.current_trajectory:
            return
        
        current_time = time.time() - self.trajectory_start_time
        
        # Get position from trajectory
        target_pos = self.current_trajectory.get_position_at_time(current_time)
        
        if target_pos:
            # Derive velocity feedforward index from elapsed time so it stays
            # in sync with get_position_at_time (trajectory_index is never
            # incremented, so using it directly always returns velocities[0]).
            traj = self.current_trajectory
            vel = Vector3D()
            if traj.velocities and traj.timestamps and len(traj.timestamps) > 1:
                # Find which segment current_time falls in (same logic as
                # get_position_at_time) and read the corresponding velocity.
                for i in range(len(traj.timestamps) - 1):
                    if traj.timestamps[i] <= current_time <= traj.timestamps[i + 1]:
                        if i < len(traj.velocities):
                            vel = traj.velocities[i]
                        break
                else:
                    # Past the last timestamp — use the final velocity
                    if traj.velocities:
                        vel = traj.velocities[-1]

            self.publish_setpoint(
                target_pos.x, target_pos.y, target_pos.z,
                self.sensor_data.yaw,
                vx=vel.x, vy=vel.y, vz=vel.z
            )
        else:
            # Trajectory finished
            self.current_trajectory = None
            self.state = DroneState.HOLD
    
    def _request_path_planning(self):
        """Request asynchronous path planning"""
        now = time.time()
        if self.current_trajectory is not None:
            return
        if self.planning_request_in_flight:
            return
        if now - self.last_planning_request_time < self.planning_request_interval:
            return
        if not self.processing_queues['planning'].full():
            planning_request = {
                'start': self.fused_position,
                'goal': self.destination_point,
                'obstacles': self.obstacle_info
            }
            self.processing_queues['planning'].put(planning_request)
            self.planning_request_in_flight = True
            self.last_planning_request_time = now
    
    def _point_gimbal_at_target(self, target: Vector3D):
        """Point gimbal at target (placeholder)"""
        # Calculate gimbal angles to point at target
        relative_pos = target - self.fused_position
        
        # Calculate pitch and yaw for gimbal
        pitch = math.atan2(-relative_pos.z, 
                          math.sqrt(relative_pos.x**2 + relative_pos.y**2))
        yaw = math.atan2(relative_pos.y, relative_pos.x)
        
        # Send gimbal commands (would be implemented with gimbal controller)
        self.get_logger().debug(f"Pointing gimbal: pitch={pitch:.2f}, yaw={yaw:.2f}")
    
    def _update_performance_metrics(self):
        """Update performance monitoring metrics"""
        self.performance_metrics.update({
            'control_frequency': self.control_loop_count,
            'position_error': self.fused_position.distance_to(self.destination_point),
            'sensor_fusion_confidence': self.obstacle_info.fusion_confidence,
            'active_sensors': len(self.obstacle_info.active_sensors),
            'obstacle_count': self.obstacle_info.total_obstacles,
            'threat_level': self.obstacle_info.max_threat_level.value
        })
        
        # Publish enhanced telemetry
        self._publish_enhanced_telemetry()
    
    def _publish_enhanced_telemetry(self):
        """Publish enhanced navigation telemetry"""
        # Navigation status
        nav_status = Float32MultiArray()
        nav_status.data = [
            float(self.state.value == 'MOVE'),  # Is flying
            self.fused_position.x,
            self.fused_position.y,
            self.fused_position.z,
            self.fused_velocity.magnitude(),
            float(self.flight_mode.value == 'PRECISION'),
            self.obstacle_info.fusion_confidence,
            float(self.emergency_stop)
        ]
        self.nav_status_pub.publish(nav_status)
        
        # Sensor fusion status
        fusion_status = Float32MultiArray()
        fusion_status.data = [
            self.sensor_data.gps_accuracy,
            float(self.sensor_data.satellites_used),
            self.sensor_data.vio_confidence,
            self.sensor_data.ground_distance,
            self.kalman_filter.P[0, 0],  # Position uncertainty X
            self.kalman_filter.P[1, 1],  # Position uncertainty Y
            self.kalman_filter.P[2, 2],  # Position uncertainty Z
        ]
        self.sensor_fusion_pub.publish(fusion_status)
    
    # Legacy compatibility methods
    def _handle_init_state(self):
        """Initialize drone for flight"""
        self.publish_setpoint(self.takeoff_point.x, self.takeoff_point.y, 
                            self.takeoff_point.z, 0.0)
        
        if self.offboard_setpoint_counter == 10:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.arm()
            self.get_logger().info("Switched to Offboard mode and armed the drone")

        if self.offboard_setpoint_counter >= 20:
            self.state = DroneState.TAKEOFF
            self.get_logger().info("Entering takeoff phase...")
    
    def _handle_takeoff_state(self):
        """Handle takeoff phase"""
        self.publish_setpoint(self.takeoff_point.x, self.takeoff_point.y, 
                            self.takeoff_point.z, 0.0)
        
        altitude_error = abs(self.fused_position.z - self.takeoff_point.z)
        if altitude_error < 0.3:
            self.state = DroneState.MOVE
            self.get_logger().info("Takeoff complete, entering autonomous navigation...")
    
    def _handle_enhanced_avoiding_state(self, dt: float):
        """Enhanced obstacle avoidance"""
        # Use advanced path planning for avoidance
        self._request_path_planning()
        
        # Emergency avoidance if immediate threat
        if self.obstacle_info.max_threat_level >= ThreatLevel.CRITICAL:
            # Execute immediate avoidance maneuver
            escape_pos = self._calculate_emergency_escape_position()
            self._fly_to_position(escape_pos, dt)
        elif self.current_trajectory:
            self._follow_trajectory(dt)
        else:
            # Hold position until new path is available
            self._execute_precision_hover(self.fused_position, 0.0, dt)
    
    def _calculate_emergency_escape_position(self) -> Vector3D:
        """Calculate immediate escape position"""
        # Find direction away from immediate threats
        escape_direction = Vector3D()
        
        for cluster in self.obstacle_info.clusters:
            if cluster.threat_level >= ThreatLevel.CRITICAL:
                away_vector = self.fused_position - cluster.center
                if away_vector.magnitude() > 0:
                    away_vector = away_vector * (1.0 / away_vector.magnitude())
                    escape_direction = escape_direction + away_vector
        
        if escape_direction.magnitude() > 0:
            escape_direction = escape_direction * (1.0 / escape_direction.magnitude())
            escape_distance = 5.0  # 5 meters away
            return self.fused_position + escape_direction * escape_distance
        
        # If no clear direction, go up
        return Vector3D(self.fused_position.x, self.fused_position.y, 
                       self.fused_position.z - 3.0)
    
    def _handle_precision_landing_state(self, dt: float):
        """Landing state: blind descent to home position.
        TODO: replace with vision-guided precision landing (ArUco / optical flow)."""
        # Placeholder — descends at fixed rate toward home_position with no visual feedback
        landing_target = self.home_position
        
        # Descend slowly with precision control
        descent_rate = 0.5  # m/s
        target_pos = Vector3D(
            landing_target.x,
            landing_target.y,
            self.fused_position.z + descent_rate * dt
        )
        
        self._execute_precision_hover(target_pos, 0.0, dt)
        
        # Check if landed
        if (self.sensor_data.ground_distance < 0.2 or 
            self.fused_position.z > self.nav_params.landing_altitude):
            self.get_logger().info("Landing complete - disarming...")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
            self.state = DroneState.DISARMED
    
    def _handle_emergency_state(self):
        """Enhanced emergency handling"""
        self.get_logger().error("EMERGENCY STATE - Executing emergency landing")

        # Emergency landing with maximum safety
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

        # Log emergency details
        self.get_logger().error(f"Emergency triggered by: {self.safety_violations}")

    def _handle_gps_denied_state(self, dt: float):
        """GPS-denied navigation: rely on VIO; loiter if VIO also lost."""
        now = time.time()
        vio_fresh = (
            self.sensor_data.vio_valid and
            now - self.sensor_data.vio_timestamp < self.vio_timeout
        )
        if vio_fresh:
            # VIO healthy — continue at reduced speed (cap at half normal max)
            self.get_logger().warn("GPS denied — navigating on VIO only",
                                   throttle_duration_sec=5.0)
            self._fly_to_position(self.destination_point, dt)
        else:
            # Both GPS and VIO unavailable — hold current position
            self.get_logger().error("GPS and VIO both unavailable — loitering",
                                    throttle_duration_sec=2.0)
            self.publish_setpoint(
                self.fused_position.x, self.fused_position.y, self.fused_position.z,
                self.sensor_data.yaw
            )
    
    # Enhanced vision processing
    def _process_frame_enhanced(self, frame: np.ndarray) -> ObstacleInfo:
        """Enhanced frame processing with multi-sensor fusion"""
        if self.detection_model is None:
            return ObstacleInfo()
        
        results = self.detection_model.predict(
            source=frame, 
            conf=self.nav_params.yolo_confidence, 
            stream=False
        )

        detections = []
        frame_height, frame_width = frame.shape[:2]
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    depth = self._get_depth_at_point(cx, cy)
                    
                    if depth and depth < self.nav_params.obstacle_threshold:
                        # Convert to world coordinates
                        world_pos = self._pixel_to_world_coordinates(cx, cy, depth)
                        
                        detection = {
                            'bbox': (x1, y1, x2, y2),
                            'center': (cx, cy),
                            'depth': depth,
                            'world_x': world_pos.x,
                            'world_y': world_pos.y,
                            'world_z': world_pos.z,
                            'conf': float(box.conf[0]),
                            'cls': int(box.cls[0])
                        }
                        detections.append(detection)

        # Create obstacle clusters
        clusters = self._create_enhanced_obstacle_clusters(detections, frame.shape)
        
        # Update clusters with tracking
        tracked_clusters = self.obstacle_tracker.update_tracks(clusters)
        
        # Assess collision risks
        for cluster in tracked_clusters:
            cluster.assess_collision_risk(self.fused_position, self.fused_velocity)
            cluster.calculate_priority_score(self.fused_position)
        
        # Sort by priority
        tracked_clusters.sort(key=lambda c: c.priority_score, reverse=True)
        
        # Create enhanced obstacle info
        obstacle_info = ObstacleInfo(
            detected=len(tracked_clusters) > 0,
            clusters=tracked_clusters,
            total_obstacles=len(tracked_clusters)
        )
        obstacle_info.update_derived_properties()
        obstacle_info.calculate_fusion_confidence()
        
        # Visualize results
        self._draw_enhanced_visualization(frame, tracked_clusters, obstacle_info)
        
        return obstacle_info
    
    def _pixel_to_world_coordinates(self, px: int, py: int, depth: float) -> Vector3D:
        """Convert pixel coordinates to world coordinates.

        Pipeline:
          pixel → normalised camera ray → rotate by drone attitude → scale by depth → add position
        """
        intrinsics = self.camera_intrinsics
        if intrinsics is not None:
            fx = intrinsics['fx']
            fy = intrinsics['fy']
            cx_cam = intrinsics['cx']
            cy_cam = intrinsics['cy']
        elif self.depth_image is not None:
            img_h, img_w = self.depth_image.shape[:2]
            fx = fy = float(max(img_w, img_h))
            cx_cam = img_w / 2.0
            cy_cam = img_h / 2.0
        else:
            fx = 465.74
            fy = 465.74
            cx_cam = 320.0
            cy_cam = 240.0

        # Back-project pixel to unit ray in camera frame (x-right, y-down, z-forward)
        ray_cam = np.array([
            (px - cx_cam) / fx,
            (py - cy_cam) / fy,
            1.0
        ])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)

        # Read attitude quaternion set by attitude_callback — (w, x, y, z)
        from scipy.spatial.transform import Rotation as R
        w, x, y, z = self.sensor_data.attitude_quaternion
        rot = R.from_quat([x, y, z, w])  # scipy expects [x, y, z, w]
        R_body_world = rot.as_matrix()  # 3×3 rotation: body → NED world

        # Camera is assumed forward-facing, body-aligned (no separate extrinsic rotation).
        # In NED: camera z-forward maps to body x-forward.
        # Simple remapping: cam_x→body_y, cam_y→body_z, cam_z→body_x
        ray_body = np.array([ray_cam[2], ray_cam[0], ray_cam[1]])

        # Rotate to world frame
        ray_world = R_body_world @ ray_body

        # Scale by metric depth and add drone position
        world_x = self.fused_position.x + ray_world[0] * depth
        world_y = self.fused_position.y + ray_world[1] * depth
        world_z = self.fused_position.z + ray_world[2] * depth

        return Vector3D(world_x, world_y, world_z)
    
    def _create_enhanced_obstacle_clusters(self, detections: List[dict], 
                                         frame_shape: Tuple[int, int]) -> List[ObstacleCluster]:
        """Create enhanced obstacle clusters with 3D information"""
        if not detections:
            return []
        
        clusters = []
        used = [False] * len(detections)
        
        for i, det in enumerate(detections):
            if used[i]:
                continue
            
            # Start new cluster
            cluster_detections = [det]
            used[i] = True
            
            # Find nearby detections to merge
            for j, other_det in enumerate(detections):
                if used[j]:
                    continue
                
                # Check 3D distance for merging
                dx = det['world_x'] - other_det['world_x']
                dy = det['world_y'] - other_det['world_y']
                dz = det.get('world_z', 0) - other_det.get('world_z', 0)
                distance_3d = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                if distance_3d < 2.0:  # 2 meter merge threshold
                    cluster_detections.append(other_det)
                    used[j] = True
            
            # Create enhanced cluster
            if cluster_detections:
                cluster = self._create_cluster_from_detections_3d(cluster_detections)
                clusters.append(cluster)
        
        return clusters
    
    def _create_cluster_from_detections_3d(self, detections: List[dict]) -> ObstacleCluster:
        """Create 3D obstacle cluster from detections"""
        # Calculate 3D center
        world_x = sum(det['world_x'] for det in detections) / len(detections)
        world_y = sum(det['world_y'] for det in detections) / len(detections)
        world_z = sum(det.get('world_z', 0) for det in detections) / len(detections)
        
        center_3d = Vector3D(world_x, world_y, world_z)
        
        # Calculate bounding box
        min_x = min(det['world_x'] for det in detections)
        max_x = max(det['world_x'] for det in detections)
        min_y = min(det['world_y'] for det in detections)
        max_y = max(det['world_y'] for det in detections)
        min_z = min(det.get('world_z', 0) for det in detections)
        max_z = max(det.get('world_z', 0) for det in detections)
        
        dimensions = Vector3D(max_x - min_x, max_y - min_y, max_z - min_z)
        bbox = BoundingBox3D(center=center_3d, dimensions=dimensions)
        
        # Calculate distance and threat level
        min_distance = min(det['depth'] for det in detections)
        
        if min_distance < self.nav_params.emergency_threshold:
            threat_level = ThreatLevel.EMERGENCY
        elif min_distance < self.nav_params.critical_threshold:
            threat_level = ThreatLevel.CRITICAL
        elif min_distance < self.nav_params.obstacle_threshold / 2:
            threat_level = ThreatLevel.HIGH
        else:
            threat_level = ThreatLevel.MEDIUM
        
        # Use the most common YOLO class across merged detections
        class_ids = [det['cls'] for det in detections if 'cls' in det]
        dominant_class = max(set(class_ids), key=class_ids.count) if class_ids else -1

        # Create cluster
        cluster = ObstacleCluster(
            cluster_id=0,  # Will be assigned by tracker
            center=center_3d,
            bounding_box=bbox,
            confidence=sum(det['conf'] for det in detections) / len(detections),
            size_estimate=dimensions.magnitude(),
            threat_level=threat_level,
            object_class=dominant_class,

            # Legacy compatibility
            center_x=world_x,
            center_y=world_y,
            min_distance=min_distance,
            width=dimensions.x,
            height=dimensions.y,
            pixel_count=len(detections),
            danger_level=min(3, max(0, threat_level.value - 1)),
            world_x=world_x,
            world_y=world_y
        )
        
        # Add sensor source information
        cluster.sensor_sources.add(SensorType.STEREO_VISION)
        cluster.sensor_confidence[SensorType.STEREO_VISION] = cluster.confidence
        
        return cluster
    
    def _extract_detections_for_tracking(self, obstacle_info: ObstacleInfo) -> List[Dict]:
        """Extract detections suitable for ActiveTrack"""
        tracking_detections = []
        
        for cluster in obstacle_info.clusters:
            # Filter for person-like objects (class 0 in COCO is person)
            if cluster.object_class == 0:
                detection = {
                    'world_x': cluster.world_x,
                    'world_y': cluster.world_y,
                    'world_z': 0.0,  # Assume ground level
                    'conf': cluster.confidence,
                    'bbox': (cluster.center_x - cluster.width/2,
                            cluster.center_y - cluster.height/2,
                            cluster.center_x + cluster.width/2,
                            cluster.center_y + cluster.height/2)
                }
                tracking_detections.append(detection)
        
        return tracking_detections
    
    def _get_depth_at_point(self, x: int, y: int) -> Optional[float]:
        """Enhanced depth extraction with noise filtering"""
        if self.depth_image is None:
            return None
        
        try:
            if (0 <= x < self.depth_image.shape[1]) and (0 <= y < self.depth_image.shape[0]):
                # Sample neighborhood for robust depth estimate
                sample_size = 5
                depths = []
                
                for dx in range(-sample_size, sample_size + 1):
                    for dy in range(-sample_size, sample_size + 1):
                        sx, sy = x + dx, y + dy
                        if (0 <= sx < self.depth_image.shape[1]) and (0 <= sy < self.depth_image.shape[0]):
                            depth = float(self.depth_image[sy, sx])
                            # Filter reasonable depth values
                            if (self.nav_params.min_obstacle_distance < depth < 
                                self.nav_params.max_obstacle_distance):
                                depths.append(depth)
                
                if depths:
                    # Use median for robust estimation
                    return np.median(depths)
        
        except Exception as e:
            self.get_logger().debug(f"Depth reading error: {e}")
        
        return None
    
    def _draw_enhanced_visualization(self, frame: np.ndarray, 
                                   clusters: List[ObstacleCluster],
                                   obstacle_info: ObstacleInfo):
        """Enhanced visualization with threat levels and tracking info"""
        
        for cluster in clusters:
            # Color based on threat level
            if cluster.threat_level == ThreatLevel.EMERGENCY:
                color = (0, 0, 255)  # Red
            elif cluster.threat_level == ThreatLevel.CRITICAL:
                color = (0, 100, 255)  # Orange
            elif cluster.threat_level == ThreatLevel.HIGH:
                color = (0, 165, 255)  # Yellow
            elif cluster.threat_level == ThreatLevel.MEDIUM:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green
            
            # Draw bounding box
            x1 = int(cluster.center_x - cluster.width/2)
            y1 = int(cluster.center_y - cluster.height/2)
            x2 = int(cluster.center_x + cluster.width/2)
            y2 = int(cluster.center_y + cluster.height/2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Enhanced label with tracking info
            track_id = cluster.track_id if cluster.track_id else "NEW"
            stable = "STABLE" if cluster.is_stable else "TRACK"
            moving = "MOVING" if cluster.is_moving else "STATIC"
            
            label = (f"ID:{track_id} {stable} {moving}\n"
                    f"D:{cluster.min_distance:.1f}m T:{cluster.threat_level.name}\n"
                    f"Conf:{cluster.confidence:.2f} P:{cluster.priority_score:.2f}")
            
            # Multi-line text
            lines = label.split('\n')
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (x1, y1 - 30 + i * 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw velocity vector if moving
            if cluster.is_moving and cluster.velocity_estimate:
                vel = cluster.velocity_estimate.velocity
                if vel.magnitude() > 0.1:
                    end_x = int(cluster.center_x + vel.x * 20)  # Scale for visibility
                    end_y = int(cluster.center_y + vel.y * 20)
                    cv2.arrowedLine(frame, (int(cluster.center_x), int(cluster.center_y)),
                                   (end_x, end_y), (255, 0, 255), 2)
        
        # Draw flight mode and status
        h, w = frame.shape[:2]
        status_text = (f"Mode: {self.flight_mode.name} | State: {self.state.name} | "
                      f"Obstacles: {obstacle_info.total_obstacles} | "
                      f"Max Threat: {obstacle_info.max_threat_level.name}")
        
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw ActiveTrack info
        if self.flight_mode == FlightMode.ACTIVETRACK:
            track_status = f"ActiveTrack: {self.tracking_mode.name} | Target: {'DETECTED' if self.active_track.target_detected else 'LOST'}"
            cv2.putText(frame, track_status, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Enhanced Smart Navigator - DJI Mavic 4 Pro Level", frame)
        cv2.waitKey(1)
    
    # Publishing methods (enhanced)
    def publish_setpoint(self, x: float, y: float, z: float, yaw: float,
                         vx: float = float('nan'), vy: float = float('nan'),
                         vz: float = float('nan')):
        """Publish position setpoint with optional velocity feedforward.

        Passing vx/vy/vz lets PX4 EKF2 use feedforward for tighter tracking at speed.
        If not supplied, PX4 will derive velocity from successive position setpoints.
        """
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.yaw = float(yaw)

        # Use caller-supplied feedforward if provided; otherwise NaN (PX4 ignores NaN)
        msg.velocity = [float(vx), float(vy), float(vz)]

        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(msg)
    
    def publish_offboard_control_mode(self):
        """Publish offboard control mode"""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_pub.publish(msg)
    
    def publish_vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0):
        """Publish vehicle command"""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(msg)
    
    def arm(self):
        """Arm the drone"""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("Enhanced Smart Navigator armed")
    
    def __del__(self):
        """Cleanup"""
        cv2.destroyAllWindows()
