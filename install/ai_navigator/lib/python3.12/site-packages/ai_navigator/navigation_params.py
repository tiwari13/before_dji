from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import math
from enum import Enum

class FlightMode(Enum):
    """Enhanced flight modes matching DJI standards"""
    POSITION = "POSITION"          # GPS position hold
    ALTITUDE = "ALTITUDE"          # Altitude hold only
    MANUAL = "MANUAL"              # Manual control
    SPORT = "SPORT"                # High performance mode
    CINEMATIC = "CINEMATIC"        # Smooth cinematic mode
    TRIPOD = "TRIPOD"              # Ultra-precise mode
    ACTIVETRACK = "ACTIVETRACK"    # Subject tracking
    TERRAIN_FOLLOW = "TERRAIN_FOLLOW"  # Terrain following
    RTH = "RTH"                    # Return to home
    LANDING = "LANDING"            # Precision landing
    WAYPOINT = "WAYPOINT"          # Waypoint navigation

class TrackingMode(Enum):
    """ActiveTrack modes"""
    SPOTLIGHT = "SPOTLIGHT"        # Keep subject centered while manual control
    PROFILE = "PROFILE"            # Follow alongside subject
    TRACE = "TRACE"                # Follow behind subject at distance
    PARALLEL = "PARALLEL"          # Track parallel to subject movement
    CIRCLE = "CIRCLE"              # Circle around subject
    HELIX = "HELIX"                # Spiral around subject

@dataclass
class SensorFusionParams:
    """Parameters for sensor fusion and state estimation"""
    # IMU parameters
    imu_accel_noise: float = 0.01          # m/s² noise
    imu_gyro_noise: float = 0.001          # rad/s noise
    imu_mag_noise: float = 0.1             # magnetic field noise
    
    # GPS parameters  
    gps_position_noise: float = 0.3        # meters
    gps_velocity_noise: float = 0.1        # m/s
    gps_min_satellites: int = 8            # minimum satellites for GPS lock
    gps_hdop_threshold: float = 2.0        # horizontal dilution of precision
    
    # Barometer parameters
    baro_noise: float = 0.1                # meters altitude noise
    baro_drift_rate: float = 0.01          # m/s drift rate
    
    # Vision parameters
    vision_position_noise: float = 0.05    # meters
    vision_velocity_noise: float = 0.02    # m/s
    vision_max_distance: float = 15.0      # max reliable distance
    
    # Kalman filter parameters
    process_noise_position: float = 0.01
    process_noise_velocity: float = 0.1
    process_noise_acceleration: float = 1.0
    measurement_timeout: float = 0.5       # seconds before switching sensors
    
    # Sensor fusion weights (normalized)
    gps_weight: float = 0.4
    vision_weight: float = 0.4
    imu_weight: float = 0.2

@dataclass
class PrecisionHoverParams:
    """Parameters for precision hovering and station keeping"""
    # Position control
    max_horizontal_error: float = 0.05     # meters
    max_vertical_error: float = 0.03       # meters
    max_yaw_error: float = 0.02           # radians (1.15 degrees)
    
    # PID gains for position control
    position_p_gain: float = 5.0
    position_i_gain: float = 0.5
    position_d_gain: float = 0.8
    position_i_max: float = 2.0           # integral windup limit
    
    # PID gains for velocity control
    velocity_p_gain: float = 8.0
    velocity_i_gain: float = 2.0
    velocity_d_gain: float = 0.1
    velocity_i_max: float = 5.0
    
    # PID gains for attitude control
    attitude_p_gain: float = 6.0
    attitude_i_gain: float = 0.3
    attitude_d_gain: float = 0.05
    
    # Hover performance limits
    max_hover_velocity: float = 0.02       # m/s
    max_hover_acceleration: float = 0.1    # m/s²
    hover_timeout: float = 300.0           # seconds before auto-land
    wind_compensation: bool = True
    max_wind_speed: float = 12.0          # m/s operational limit

@dataclass
class TerrainFollowParams:
    """Parameters for terrain following capabilities"""
    # Terrain sensing
    terrain_radar_range: float = 30.0      # meters
    terrain_radar_resolution: float = 0.1  # meters
    min_terrain_clearance: float = 2.0     # meters
    max_terrain_clearance: float = 20.0    # meters
    
    # Terrain following behavior
    terrain_follow_speed: float = 3.0      # m/s max speed
    terrain_slope_limit: float = 0.4       # radians (23 degrees)
    terrain_roughness_limit: float = 2.0   # meters vertical variation
    
    # Adaptive parameters
    adaptive_height: bool = True
    height_adaptation_rate: float = 0.5    # m/s climb/descent rate
    terrain_prediction_distance: float = 10.0  # meters lookahead
    
    # Safety margins
    obstacle_clearance: float = 3.0        # meters above obstacles
    emergency_climb_rate: float = 2.0      # m/s when obstacle detected
    terrain_mode_ceiling: float = 100.0    # meters AGL limit

@dataclass
class ActiveTrackParams:
    """Parameters for subject tracking and following"""
    # Detection and tracking
    detection_confidence: float = 0.7      # minimum confidence for tracking
    tracking_fps: int = 30                 # tracking update rate
    max_tracking_distance: float = 50.0    # meters
    min_tracking_distance: float = 2.0     # meters
    
    # Subject prediction
    prediction_horizon: float = 2.0        # seconds ahead prediction
    velocity_smoothing: float = 0.8        # velocity filter factor
    acceleration_limit: float = 5.0        # m/s² max subject acceleration
    
    # Tracking behavior per mode
    tracking_modes: Dict[TrackingMode, Dict] = field(default_factory=lambda: {
        TrackingMode.SPOTLIGHT: {
            'follow_distance': 0.0,           # stay in place, just point at subject
            'follow_height_offset': 0.0,
            'max_speed': 5.0,
            'rotation_speed': 0.5             # rad/s
        },
        TrackingMode.PROFILE: {
            'follow_distance': 8.0,           # meters behind/beside
            'follow_height_offset': 2.0,      # meters above subject
            'max_speed': 8.0,
            'side_offset': 5.0                # meters to the side
        },
        TrackingMode.TRACE: {
            'follow_distance': 10.0,          # meters behind
            'follow_height_offset': 3.0,
            'max_speed': 10.0,
            'lag_distance': 2.0               # smoothing lag
        },
        TrackingMode.PARALLEL: {
            'follow_distance': 0.0,
            'follow_height_offset': 0.0,
            'max_speed': 12.0,
            'parallel_offset': 8.0            # meters parallel
        },
        TrackingMode.CIRCLE: {
            'circle_radius': 8.0,             # meters
            'circle_height': 5.0,             # meters above subject
            'circle_speed': 0.2,              # rad/s
            'max_speed': 6.0
        },
        TrackingMode.HELIX: {
            'helix_radius': 10.0,             # meters
            'helix_pitch': 2.0,               # meters per revolution
            'helix_speed': 0.15,              # rad/s
            'max_speed': 4.0
        }
    })
    
    # Loss recovery
    subject_lost_timeout: float = 3.0       # seconds before search mode
    search_pattern_radius: float = 20.0     # meters search radius
    reacquisition_confidence: float = 0.8   # confidence to resume tracking

@dataclass
class AdvancedPathPlannerParams:
    """Parameters for advanced path planning algorithms"""
    # A* pathfinding
    grid_resolution: float = 0.5            # meters per grid cell
    heuristic_weight: float = 1.2           # A* heuristic multiplier
    max_planning_time: float = 0.1          # seconds per planning cycle
    
    # RRT* parameters
    rrt_max_iterations: int = 1000
    rrt_step_size: float = 1.0             # meters
    rrt_goal_bias: float = 0.1             # probability of sampling goal
    rrt_rewiring_radius: float = 2.0       # meters
    
    # Dynamic window approach
    dwa_velocity_samples: int = 50
    dwa_angular_samples: int = 20
    dwa_prediction_time: float = 2.0       # seconds
    dwa_obstacle_weight: float = 0.4
    dwa_goal_weight: float = 0.3
    dwa_velocity_weight: float = 0.3
    
    # Trajectory optimization
    trajectory_smoothing: bool = True
    smoothing_iterations: int = 10
    max_curvature: float = 0.5             # rad/m
    comfort_acceleration: float = 2.0       # m/s²
    
    # Multi-level planning
    global_replanning_interval: float = 5.0  # seconds
    local_replanning_interval: float = 0.2   # seconds
    corridor_width: float = 4.0            # meters safe corridor

@dataclass
class SafetyParams:
    """Enhanced safety and fail-safe parameters"""
    # Geofencing
    max_altitude: float = 120.0            # meters AGL legal limit
    max_distance: float = 500.0            # meters from home
    geofence_margin: float = 10.0          # meters warning zone
    
    # Battery management
    battery_rtl_threshold: float = 30.0    # % battery for RTH
    battery_land_threshold: float = 15.0   # % battery for immediate landing
    battery_critical_threshold: float = 10.0  # % for emergency landing
    
    # Environmental limits
    max_wind_speed: float = 12.0           # m/s operational limit
    max_temperature: float = 40.0          # °C
    min_temperature: float = -10.0         # °C
    max_precipitation: float = 0.1         # mm/h light drizzle limit
    
    # Collision avoidance
    emergency_stop_distance: float = 1.0   # meters
    collision_time_horizon: float = 3.0    # seconds
    obstacle_detection_range: float = 20.0 # meters
    
    # Communication
    rc_signal_timeout: float = 3.0         # seconds
    telemetry_timeout: float = 10.0        # seconds
    gps_timeout: float = 5.0               # seconds

@dataclass
class PerformanceParams:
    """Performance tuning and optimization parameters"""
    # Flight performance modes
    performance_modes: Dict[FlightMode, Dict] = field(default_factory=lambda: {
        FlightMode.POSITION: {
            'max_speed': 5.0,                # m/s
            'max_acceleration': 2.0,         # m/s²
            'max_angular_velocity': 0.5,     # rad/s
            'smoothing_factor': 0.8
        },
        FlightMode.SPORT: {
            'max_speed': 15.0,
            'max_acceleration': 5.0,
            'max_angular_velocity': 1.5,
            'smoothing_factor': 0.3
        },
        FlightMode.CINEMATIC: {
            'max_speed': 3.0,
            'max_acceleration': 1.0,
            'max_angular_velocity': 0.2,
            'smoothing_factor': 0.95
        },
        FlightMode.TRIPOD: {
            'max_speed': 1.0,
            'max_acceleration': 0.5,
            'max_angular_velocity': 0.1,
            'smoothing_factor': 0.98
        }
    })
    
    # Real-time optimization
    control_frequency: float = 100.0        # Hz main control loop
    estimation_frequency: float = 200.0     # Hz state estimation
    planning_frequency: float = 20.0        # Hz path planning
    vision_frequency: float = 30.0          # Hz computer vision
    
    # Resource management
    cpu_usage_limit: float = 80.0           # % max CPU usage
    memory_usage_limit: float = 80.0        # % max memory usage
    thermal_throttle_temp: float = 75.0     # °C CPU temperature limit

@dataclass
class NavigationParams:
    """Enhanced navigation parameters for DJI Mavic 4 Pro level performance"""
    
    # Basic flight parameters
    takeoff_altitude: float = -3.0          # meters (negative in NED)
    landing_altitude: float = -0.2          # meters
    destination_offset: float = 25.0        # meters forward travel
    arrival_threshold: float = 0.5          # meters
    
    # Speed and acceleration limits
    max_speed: float = 8.0                  # m/s (sport mode)
    max_acceleration: float = 3.0           # m/s²
    max_angular_velocity: float = 1.0       # rad/s
    max_climb_rate: float = 3.0            # m/s
    max_descent_rate: float = 2.0          # m/s
    
    # Obstacle avoidance
    obstacle_threshold: float = 8.0         # meters detection range
    critical_threshold: float = 3.0         # meters critical distance
    emergency_threshold: float = 1.5        # meters emergency stop
    obstacle_buffer_zone: float = 2.0       # meters safety margin
    min_obstacle_distance: float = 0.2      # meters minimum detection
    max_obstacle_distance: float = 30.0     # meters maximum detection
    
    # Computer vision
    yolo_confidence: float = 0.6            # higher confidence for better detection
    camera_fov_horizontal: float = 1.396    # radians (80°)
    camera_fov_vertical: float = 1.047      # radians (60°)
    vision_update_rate: float = 30.0        # Hz
    
    # Path planning
    avoidance_offset: float = 4.0           # meters lateral avoidance
    vertical_avoidance: float = 3.0         # meters vertical avoidance
    retreat_distance: float = 5.0           # meters retreat distance
    circle_radius: float = 8.0              # meters circling radius
    lookahead_distance: float = 15.0        # meters planning horizon
    
    # Clustering and grouping
    cluster_merge_threshold: float = 100.0  # pixels
    min_safe_gap: float = 150.0            # pixels minimum gap
    
    # Timeouts and intervals
    max_avoidance_time: float = 30.0        # seconds
    position_timeout: float = 2.0           # seconds
    planning_interval: float = 0.1          # seconds
    
    # Advanced subsystem parameters
    sensor_fusion: SensorFusionParams = field(default_factory=SensorFusionParams)
    precision_hover: PrecisionHoverParams = field(default_factory=PrecisionHoverParams)
    terrain_follow: TerrainFollowParams = field(default_factory=TerrainFollowParams)
    active_track: ActiveTrackParams = field(default_factory=ActiveTrackParams)
    path_planner: AdvancedPathPlannerParams = field(default_factory=AdvancedPathPlannerParams)
    safety: SafetyParams = field(default_factory=SafetyParams)
    performance: PerformanceParams = field(default_factory=PerformanceParams)
    
    # Current flight mode
    flight_mode: FlightMode = FlightMode.POSITION
    tracking_mode: TrackingMode = TrackingMode.TRACE
    
    def get_current_performance_limits(self) -> Dict:
        """Get performance limits for current flight mode"""
        return self.performance.performance_modes.get(
            self.flight_mode, 
            self.performance.performance_modes[FlightMode.POSITION]
        )
    
    def get_tracking_params(self) -> Dict:
        """Get parameters for current tracking mode"""
        return self.active_track.tracking_modes.get(
            self.tracking_mode,
            self.active_track.tracking_modes[TrackingMode.TRACE]
        )
    
    def update_flight_mode(self, mode: FlightMode):
        """Update flight mode and associated parameters"""
        self.flight_mode = mode
        limits = self.get_current_performance_limits()
        
        # Update main parameters based on flight mode
        self.max_speed = limits['max_speed']
        self.max_acceleration = limits['max_acceleration']
        self.max_angular_velocity = limits['max_angular_velocity']
    
    def validate_parameters(self) -> List[str]:
        """Validate parameter consistency and return warnings"""
        warnings = []
        
        if self.obstacle_threshold <= self.critical_threshold:
            warnings.append("Obstacle threshold should be greater than critical threshold")
        
        if self.critical_threshold <= self.emergency_threshold:
            warnings.append("Critical threshold should be greater than emergency threshold")
        
        if self.max_speed > self.safety.max_wind_speed * 1.5:
            warnings.append("Max speed too high for wind resistance")
        
        if self.takeoff_altitude > -1.0:
            warnings.append("Takeoff altitude should be at least 1 meter")
        
        return warnings