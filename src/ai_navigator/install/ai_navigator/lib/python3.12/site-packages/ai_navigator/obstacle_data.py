from dataclasses import dataclass
from typing import List, Optional
from math import sqrt
# ai_navigator/obstacle_data.py

"""Obstacle data structures for AI navigation system"""

@dataclass
class ObstacleCluster:
    """Represents a cluster of obstacles"""
    center_x: float
    center_y: float
    min_distance: float
    width: float
    height: float
    pixel_count: int
    danger_level: int  # 1=low, 2=medium, 3=high
    world_x: float = 0.0  # Real-world X coordinate
    world_y: float = 0.0  # Real-world Y coordinate

@dataclass
class ObstacleInfo:
    """Enhanced obstacle information"""
    detected: bool = False
    clusters: List[ObstacleCluster] = None
    min_distance: float = float('inf')
    safe_directions: List[str] = None
    blocked_directions: List[str] = None
    escape_angle: Optional[float] = None
    
    def __post_init__(self):
        if self.clusters is None:
            self.clusters = []
        if self.safe_directions is None:
            self.safe_directions = []
        if self.blocked_directions is None:
            self.blocked_directions = []