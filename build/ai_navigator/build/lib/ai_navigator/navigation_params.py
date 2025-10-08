from dataclasses import dataclass
from typing import List, Optional
import math
from collections import deque


@dataclass
class NavigationParams:
    """Enhanced navigation parameters"""
    takeoff_altitude: float = -1.0
    destination_offset: float = 25.0
    obstacle_threshold: float = 5.0
    critical_threshold: float = 2.5
    emergency_threshold: float = 1.0
    min_obstacle_distance: float = 0.1
    max_obstacle_distance: float = 10.0
    avoidance_offset: float = 4.0
    vertical_avoidance: float = 1.5
    arrival_threshold: float = 0.5
    landing_altitude: float = -0.2
    yolo_confidence: float = 0.25
    max_avoidance_time: float = 25.0
    retreat_distance: float = 4.0
    circle_radius: float = 5.0
    max_speed: float = 2.5
    safe_corridor_width: float = 2.5
    vertical_climb_height: float = 2.5
    max_climb_altitude: float = -10.0
    lateral_scan_width: float = 8.0
    forward_scan_depth: float = 10.0
    multi_level_avoidance: bool = True
    obstacle_buffer_zone: float = 2.0
    adaptive_speed_factor: float = 0.25
    cluster_merge_threshold: float = 80.0
    min_safe_gap: float = 100.0
    lookahead_distance: float = 15.0
    camera_fov_horizontal: float = 1.396  # 80 degrees in radians (approximate for typical drone camera)
    camera_fov_vertical: float = 0.872  # 50 degrees in radians