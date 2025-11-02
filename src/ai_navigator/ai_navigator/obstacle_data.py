import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
from enum import Enum
import time
from collections import deque, defaultdict
import cv2

class ObstacleType(Enum):
    """Types of obstacles that can be detected"""
    STATIC = "STATIC"              # Buildings, trees, poles
    DYNAMIC = "DYNAMIC"            # Vehicles, people, animals
    AIRCRAFT = "AIRCRAFT"          # Other drones, aircraft
    BIRD = "BIRD"                  # Birds and flying objects
    GROUND = "GROUND"              # Ground/terrain
    WIRE = "WIRE"                  # Power lines, cables
    UNKNOWN = "UNKNOWN"            # Unclassified obstacles

class SensorType(Enum):
    """Types of sensors providing obstacle data"""
    STEREO_VISION = "STEREO_VISION"
    MONOCULAR_VISION = "MONOCULAR_VISION"
    LIDAR = "LIDAR"
    TOF = "TOF"                    # Time of Flight
    ULTRASONIC = "ULTRASONIC"
    RADAR = "RADAR"
    THERMAL = "THERMAL"

class ThreatLevel(Enum):
    """Threat assessment levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

@dataclass
class Vector3D:
    """3D vector for positions and velocities"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag > 0:
            return Vector3D(self.x/mag, self.y/mag, self.z/mag)
        return Vector3D()
    
    def distance_to(self, other: 'Vector3D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

@dataclass
class BoundingBox3D:
    """3D bounding box for obstacles"""
    center: Vector3D
    dimensions: Vector3D  # width, height, depth
    orientation: float = 0.0  # yaw angle in radians
    
    def contains_point(self, point: Vector3D) -> bool:
        """Check if point is inside bounding box"""
        # Simplified check - assumes axis-aligned box
        half_dims = self.dimensions * 0.5
        return (abs(point.x - self.center.x) <= half_dims.x and
                abs(point.y - self.center.y) <= half_dims.y and
                abs(point.z - self.center.z) <= half_dims.z)
    
    def volume(self) -> float:
        """Calculate volume of bounding box"""
        return self.dimensions.x * self.dimensions.y * self.dimensions.z
    
    def intersects(self, other: 'BoundingBox3D') -> bool:
        """Check if this box intersects with another"""
        # Simplified AABB intersection
        half_dims_self = self.dimensions * 0.5
        half_dims_other = other.dimensions * 0.5
        
        return (abs(self.center.x - other.center.x) <= (half_dims_self.x + half_dims_other.x) and
                abs(self.center.y - other.center.y) <= (half_dims_self.y + half_dims_other.y) and
                abs(self.center.z - other.center.z) <= (half_dims_self.z + half_dims_other.z))

@dataclass
class ObstaclePoint:
    """Individual obstacle point with sensor data"""
    position: Vector3D
    sensor_type: SensorType
    confidence: float
    timestamp: float
    intensity: float = 0.0  # For LIDAR/radar reflectivity
    color: Tuple[int, int, int] = (0, 0, 0)  # RGB for vision
    temperature: float = 0.0  # For thermal sensors

@dataclass
class VelocityEstimate:
    """Velocity estimation for dynamic obstacles"""
    velocity: Vector3D
    acceleration: Vector3D
    confidence: float
    timestamp: float
    prediction_horizon: float = 3.0  # seconds
    
    def predict_position(self, current_pos: Vector3D, time_delta: float) -> Vector3D:
        """Predict future position using kinematic model"""
        if time_delta > self.prediction_horizon:
            time_delta = self.prediction_horizon
            
        # Use kinematic equation: s = ut + 0.5*at²
        displacement = (self.velocity * time_delta + 
                       self.acceleration * (0.5 * time_delta * time_delta))
        return current_pos + displacement

@dataclass
class CollisionRisk:
    """Collision risk assessment"""
    time_to_collision: float  # seconds
    collision_probability: float  # 0.0 to 1.0
    closest_approach_distance: float  # meters
    threat_level: ThreatLevel
    avoidance_direction: Optional[Vector3D] = None
    
    def is_critical(self) -> bool:
        return self.threat_level >= ThreatLevel.HIGH

@dataclass
class ObstacleCluster:
    """Enhanced cluster of obstacle points with advanced tracking"""
    # Basic properties
    cluster_id: int
    points: List[ObstaclePoint] = field(default_factory=list)
    center: Vector3D = field(default_factory=Vector3D)
    bounding_box: Optional[BoundingBox3D] = None
    
    # Classification
    obstacle_type: ObstacleType = ObstacleType.UNKNOWN
    confidence: float = 0.0
    size_estimate: float = 0.0  # meters
    
    # Temporal tracking
    first_detected: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    track_id: Optional[int] = None
    is_stable: bool = False  # stable for multiple frames
    
    # Motion analysis
    velocity_estimate: Optional[VelocityEstimate] = None
    position_history: deque = field(default_factory=lambda: deque(maxlen=20))
    is_moving: bool = False
    movement_confidence: float = 0.0
    
    # Multi-sensor fusion
    sensor_sources: Set[SensorType] = field(default_factory=set)
    sensor_confidence: Dict[SensorType, float] = field(default_factory=dict)
    fusion_weight: float = 1.0
    
    # Risk assessment
    collision_risk: Optional[CollisionRisk] = None
    threat_level: ThreatLevel = ThreatLevel.NONE
    priority_score: float = 0.0  # higher = more important
    
    # Geometric properties (legacy compatibility)
    center_x: float = 0.0
    center_y: float = 0.0
    min_distance: float = float('inf')
    width: float = 0.0
    height: float = 0.0
    pixel_count: int = 0
    danger_level: int = 0
    world_x: float = 0.0
    world_y: float = 0.0
    
    def __post_init__(self):
        """Initialize derived properties"""
        if self.points:
            self.update_properties()
    
    def update_properties(self):
        """Update all derived properties from points"""
        if not self.points:
            return
            
        # Calculate center
        positions = [p.position for p in self.points]
        self.center = Vector3D(
            sum(p.x for p in positions) / len(positions),
            sum(p.y for p in positions) / len(positions),
            sum(p.z for p in positions) / len(positions)
        )
        
        # Update legacy properties for compatibility
        self.center_x = self.center.x
        self.center_y = self.center.y
        self.world_x = self.center.x
        self.world_y = self.center.y
        self.pixel_count = len(self.points)
        
        # Calculate bounding box
        if len(positions) > 1:
            min_x = min(p.x for p in positions)
            max_x = max(p.x for p in positions)
            min_y = min(p.y for p in positions)
            max_y = max(p.y for p in positions)
            min_z = min(p.z for p in positions)
            max_z = max(p.z for p in positions)
            
            self.width = max_x - min_x
            self.height = max_y - min_y
            
            self.bounding_box = BoundingBox3D(
                center=self.center,
                dimensions=Vector3D(max_x - min_x, max_y - min_y, max_z - min_z)
            )
            
            self.size_estimate = self.bounding_box.dimensions.magnitude()
        
        # Calculate distance (legacy)
        self.min_distance = min(p.position.magnitude() for p in self.points)
        
        # Update sensor sources
        self.sensor_sources = set(p.sensor_type for p in self.points)
        
        # Calculate confidence
        if self.points:
            self.confidence = sum(p.confidence for p in self.points) / len(self.points)
        
        # Update timestamps
        self.last_updated = time.time()
        
        # Add to position history
        self.position_history.append((time.time(), self.center))
    
    def add_points(self, new_points: List[ObstaclePoint]):
        """Add new points to cluster"""
        self.points.extend(new_points)
        self.update_properties()
    
    def merge_with(self, other: 'ObstacleCluster'):
        """Merge another cluster into this one"""
        self.points.extend(other.points)
        self.sensor_sources.update(other.sensor_sources)
        self.update_properties()
        
        # Keep the earlier detection time
        self.first_detected = min(self.first_detected, other.first_detected)
        
        # Use higher confidence classification
        if other.confidence > self.confidence:
            self.obstacle_type = other.obstacle_type
            self.confidence = other.confidence
    
    def estimate_velocity(self) -> Optional[VelocityEstimate]:
        """Estimate velocity from position history"""
        if len(self.position_history) < 3:
            return None
            
        # Use last few positions for velocity estimation
        recent_positions = list(self.position_history)[-5:]
        if len(recent_positions) < 2:
            return None
            
        # Linear regression for velocity estimation
        times = [pos[0] for pos in recent_positions]
        x_positions = [pos[1].x for pos in recent_positions]
        y_positions = [pos[1].y for pos in recent_positions]
        z_positions = [pos[1].z for pos in recent_positions]
        
        # Simple finite difference for velocity
        dt = times[-1] - times[0]
        if dt < 0.1:  # Too short time interval
            return None
            
        dx = x_positions[-1] - x_positions[0]
        dy = y_positions[-1] - y_positions[0]
        dz = z_positions[-1] - z_positions[0]
        
        velocity = Vector3D(dx/dt, dy/dt, dz/dt)
        
        # Estimate acceleration if we have enough data
        acceleration = Vector3D()
        if len(recent_positions) >= 3:
            mid_idx = len(recent_positions) // 2
            dt1 = times[mid_idx] - times[0]
            dt2 = times[-1] - times[mid_idx]
            
            if dt1 > 0 and dt2 > 0:
                v1_x = (x_positions[mid_idx] - x_positions[0]) / dt1
                v2_x = (x_positions[-1] - x_positions[mid_idx]) / dt2
                acceleration.x = (v2_x - v1_x) / dt2
                
                v1_y = (y_positions[mid_idx] - y_positions[0]) / dt1
                v2_y = (y_positions[-1] - y_positions[mid_idx]) / dt2
                acceleration.y = (v2_y - v1_y) / dt2
                
                v1_z = (z_positions[mid_idx] - z_positions[0]) / dt1
                v2_z = (z_positions[-1] - z_positions[mid_idx]) / dt2
                acceleration.z = (v2_z - v1_z) / dt2
        
        # Velocity confidence based on consistency
        velocity_magnitude = velocity.magnitude()
        confidence = 0.5  # Base confidence
        
        if velocity_magnitude > 0.1:  # Moving object
            self.is_moving = True
            confidence = min(1.0, len(recent_positions) / 5.0)
        else:
            self.is_moving = False
            
        self.movement_confidence = confidence
        
        self.velocity_estimate = VelocityEstimate(
            velocity=velocity,
            acceleration=acceleration,
            confidence=confidence,
            timestamp=time.time()
        )
        
        return self.velocity_estimate
    
    def assess_collision_risk(self, drone_position: Vector3D, drone_velocity: Vector3D) -> CollisionRisk:
        """Assess collision risk with drone"""
        if not self.velocity_estimate:
            self.estimate_velocity()
        
        # Relative velocity
        rel_velocity = Vector3D()
        if self.velocity_estimate:
            rel_velocity = self.velocity_estimate.velocity - drone_velocity
        else:
            rel_velocity = Vector3D() - drone_velocity  # Assume obstacle is stationary
        
        # Relative position
        rel_position = self.center - drone_position
        
        # Time to closest approach
        rel_speed = rel_velocity.magnitude()
        if rel_speed < 0.1:  # Nearly stationary relative motion
            time_to_collision = float('inf')
            closest_distance = rel_position.magnitude()
        else:
            # Project relative position onto relative velocity
            t_closest = -(rel_position.x * rel_velocity.x + 
                         rel_position.y * rel_velocity.y + 
                         rel_position.z * rel_velocity.z) / (rel_speed * rel_speed)
            
            if t_closest < 0:
                t_closest = 0  # Already past closest approach
            
            # Position at closest approach
            closest_pos = rel_position + rel_velocity * t_closest
            closest_distance = closest_pos.magnitude()
            
            # Account for obstacle size
            safety_margin = max(2.0, self.size_estimate + 1.0)  # meters
            if closest_distance > safety_margin:
                time_to_collision = float('inf')
            else:
                time_to_collision = t_closest
        
        # Collision probability based on uncertainty
        collision_prob = 0.0
        if time_to_collision < 10.0:  # Only consider near-term collisions
            uncertainty_factor = 1.0 - self.confidence
            distance_factor = max(0.0, 1.0 - closest_distance / 5.0)
            time_factor = max(0.0, 1.0 - time_to_collision / 10.0)
            collision_prob = distance_factor * time_factor * (1.0 - uncertainty_factor)
        
        # Determine threat level
        threat_level = ThreatLevel.NONE
        if time_to_collision < 1.0 and closest_distance < 2.0:
            threat_level = ThreatLevel.EMERGENCY
        elif time_to_collision < 2.0 and closest_distance < 3.0:
            threat_level = ThreatLevel.CRITICAL
        elif time_to_collision < 5.0 and closest_distance < 5.0:
            threat_level = ThreatLevel.HIGH
        elif time_to_collision < 10.0 and closest_distance < 8.0:
            threat_level = ThreatLevel.MEDIUM
        elif closest_distance < 15.0:
            threat_level = ThreatLevel.LOW
        
        # Calculate avoidance direction (perpendicular to relative velocity)
        avoidance_dir = None
        if rel_velocity.magnitude() > 0.1:
            # Choose direction perpendicular to relative motion
            perp = Vector3D(-rel_velocity.y, rel_velocity.x, 0).normalize()
            if perp.magnitude() > 0:
                avoidance_dir = perp
        
        self.collision_risk = CollisionRisk(
            time_to_collision=time_to_collision,
            collision_probability=collision_prob,
            closest_approach_distance=closest_distance,
            threat_level=threat_level,
            avoidance_direction=avoidance_dir
        )
        
        self.threat_level = threat_level
        
        # Update legacy danger level for compatibility
        self.danger_level = min(3, max(0, int(threat_level.value) - 1))
        
        return self.collision_risk
    
    def calculate_priority_score(self, drone_position: Vector3D) -> float:
        """Calculate priority score for processing order"""
        distance_score = 1.0 / max(0.1, self.min_distance)  # Closer = higher priority
        threat_score = self.threat_level.value / 5.0  # Threat level contribution
        size_score = min(1.0, self.size_estimate / 5.0)  # Larger obstacles more important
        confidence_score = self.confidence  # Higher confidence more important
        
        # Moving objects get higher priority
        motion_score = 0.0
        if self.is_moving and self.velocity_estimate:
            motion_score = min(1.0, self.velocity_estimate.velocity.magnitude() / 5.0)
        
        self.priority_score = (distance_score * 0.3 + 
                              threat_score * 0.4 + 
                              size_score * 0.1 + 
                              confidence_score * 0.1 + 
                              motion_score * 0.1)
        
        return self.priority_score
    
    def is_expired(self, max_age: float = 2.0) -> bool:
        """Check if cluster is too old to be reliable"""
        return (time.time() - self.last_updated) > max_age
    
    def get_predicted_position(self, time_ahead: float) -> Vector3D:
        """Get predicted position at future time"""
        if self.velocity_estimate:
            return self.velocity_estimate.predict_position(self.center, time_ahead)
        return self.center

@dataclass
class ObstacleInfo:
    """Enhanced obstacle information with multi-sensor fusion"""
    # Detection status
    detected: bool = False
    total_obstacles: int = 0
    
    # Obstacle clusters
    clusters: List[ObstacleCluster] = field(default_factory=list)
    
    # Risk assessment
    min_distance: float = float('inf')
    max_threat_level: ThreatLevel = ThreatLevel.NONE
    immediate_threats: List[ObstacleCluster] = field(default_factory=list)
    
    # Spatial analysis
    safe_directions: List[str] = field(default_factory=list)
    blocked_directions: List[str] = field(default_factory=list)
    escape_angle: Optional[float] = None
    recommended_action: str = "CONTINUE"
    
    # Sensor fusion status
    active_sensors: Set[SensorType] = field(default_factory=set)
    sensor_health: Dict[SensorType, float] = field(default_factory=dict)
    fusion_confidence: float = 0.0
    
    # Environmental context
    visibility_conditions: float = 1.0  # 0=poor, 1=excellent
    weather_factor: float = 1.0
    lighting_conditions: str = "GOOD"
    
    # Processing metrics
    processing_time: float = 0.0
    frame_rate: float = 0.0
    cpu_usage: float = 0.0
    
    # Tracking statistics
    new_detections: int = 0
    lost_tracks: int = 0
    stable_tracks: int = 0
    
    def __post_init__(self):
        """Initialize derived properties"""
        self.update_derived_properties()
    
    def update_derived_properties(self):
        """Update all derived properties from clusters"""
        if not self.clusters:
            self.detected = False
            self.total_obstacles = 0
            self.min_distance = float('inf')
            self.max_threat_level = ThreatLevel.NONE
            self.immediate_threats = []
            return
        
        self.detected = True
        self.total_obstacles = len(self.clusters)
        
        # Calculate minimum distance
        self.min_distance = min(c.min_distance for c in self.clusters)
        
        # Find maximum threat level
        self.max_threat_level = max(c.threat_level for c in self.clusters)
        
        # Identify immediate threats
        self.immediate_threats = [c for c in self.clusters 
                                 if c.threat_level >= ThreatLevel.HIGH]
        
        # Update active sensors
        self.active_sensors = set()
        for cluster in self.clusters:
            self.active_sensors.update(cluster.sensor_sources)
    
    def sort_by_priority(self):
        """Sort clusters by priority score"""
        self.clusters.sort(key=lambda c: c.priority_score, reverse=True)
    
    def get_clusters_by_threat(self, min_threat: ThreatLevel) -> List[ObstacleCluster]:
        """Get clusters above minimum threat level"""
        return [c for c in self.clusters if c.threat_level >= min_threat]
    
    def get_moving_obstacles(self) -> List[ObstacleCluster]:
        """Get all moving obstacles"""
        return [c for c in self.clusters if c.is_moving]
    
    def get_static_obstacles(self) -> List[ObstacleCluster]:
        """Get all static obstacles"""
        return [c for c in self.clusters if not c.is_moving]
    
    def cleanup_expired_clusters(self, max_age: float = 2.0):
        """Remove expired clusters"""
        active_clusters = [c for c in self.clusters if not c.is_expired(max_age)]
        lost_count = len(self.clusters) - len(active_clusters)
        self.lost_tracks = lost_count
        self.clusters = active_clusters
        self.update_derived_properties()
    
    def calculate_fusion_confidence(self):
        """Calculate overall sensor fusion confidence"""
        if not self.clusters:
            self.fusion_confidence = 0.0
            return
        
        # Weighted average of cluster confidences
        total_weight = sum(c.fusion_weight for c in self.clusters)
        if total_weight > 0:
            weighted_confidence = sum(c.confidence * c.fusion_weight for c in self.clusters)
            self.fusion_confidence = weighted_confidence / total_weight
        else:
            self.fusion_confidence = 0.0
        
        # Adjust for sensor diversity
        sensor_diversity = len(self.active_sensors) / len(SensorType)
        self.fusion_confidence *= (0.5 + 0.5 * sensor_diversity)
    
    def get_emergency_action(self) -> str:
        """Get recommended emergency action"""
        if self.max_threat_level >= ThreatLevel.EMERGENCY:
            return "EMERGENCY_STOP"
        elif self.max_threat_level >= ThreatLevel.CRITICAL:
            return "IMMEDIATE_AVOIDANCE"
        elif self.max_threat_level >= ThreatLevel.HIGH:
            return "PLAN_AVOIDANCE"
        elif self.max_threat_level >= ThreatLevel.MEDIUM:
            return "ADJUST_PATH"
        else:
            return "CONTINUE"

class ObstacleTracker:
    """Advanced multi-target tracking for obstacles"""
    
    def __init__(self, max_track_age: float = 3.0):
        self.tracks: Dict[int, ObstacleCluster] = {}
        self.next_track_id = 1
        self.max_track_age = max_track_age
        self.track_association_threshold = 2.0  # meters
        
    def update_tracks(self, detections: List[ObstacleCluster]) -> List[ObstacleCluster]:
        """Update tracks with new detections using Hungarian algorithm"""
        # Simple nearest neighbor association for now
        # In production, use Hungarian algorithm or Joint Probabilistic Data Association
        
        updated_tracks = []
        unmatched_detections = detections.copy()
        
        # Try to match detections to existing tracks
        for track_id, track in self.tracks.items():
            best_match = None
            best_distance = float('inf')
            
            for detection in unmatched_detections:
                distance = track.center.distance_to(detection.center)
                if distance < self.track_association_threshold and distance < best_distance:
                    best_match = detection
                    best_distance = distance
            
            if best_match:
                # Update existing track
                track.add_points(best_match.points)
                track.estimate_velocity()
                track.is_stable = (time.time() - track.first_detected) > 1.0
                updated_tracks.append(track)
                unmatched_detections.remove(best_match)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            detection.track_id = self.next_track_id
            detection.cluster_id = self.next_track_id
            self.tracks[self.next_track_id] = detection
            updated_tracks.append(detection)
            self.next_track_id += 1
        
        # Remove expired tracks
        current_time = time.time()
        expired_track_ids = [
            track_id for track_id, track in self.tracks.items()
            if (current_time - track.last_updated) > self.max_track_age
        ]
        
        for track_id in expired_track_ids:
            del self.tracks[track_id]
        
        # Update tracks dictionary
        self.tracks = {track.track_id: track for track in updated_tracks if track.track_id}
        
        return updated_tracks