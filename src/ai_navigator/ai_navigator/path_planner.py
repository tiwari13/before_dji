import numpy as np
import math
import heapq
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
from collections import deque
import time
from enum import Enum

# Import our enhanced obstacle data structures
from ai_navigator.obstacle_data import (
    ObstacleInfo, ObstacleCluster, Vector3D, CollisionRisk, 
    ThreatLevel, BoundingBox3D
)
from ai_navigator.navigation_params import NavigationParams

class PlannerType(Enum):
    """Types of path planning algorithms"""
    DIRECT = "DIRECT"                   # Straight line to goal
    A_STAR = "A_STAR"                  # A* grid-based planning
    RRT_STAR = "RRT_STAR"              # RRT* sampling-based planning
    DWA = "DWA"                        # Dynamic Window Approach
    HYBRID = "HYBRID"                   # Multi-algorithm hybrid
    EMERGENCY = "EMERGENCY"             # Emergency avoidance
    FOLLOW_TERRAIN = "FOLLOW_TERRAIN"   # Terrain following
    ACTIVETRACK = "ACTIVETRACK"         # Subject tracking paths

class PathType(Enum):
    """Types of paths that can be generated"""
    GLOBAL = "GLOBAL"                   # Long-term strategic path
    LOCAL = "LOCAL"                     # Short-term tactical path
    EMERGENCY = "EMERGENCY"             # Immediate collision avoidance
    SMOOTH = "SMOOTH"                   # Smoothed/optimized path
    WAYPOINT = "WAYPOINT"               # Waypoint-based path

@dataclass
class PathNode:
    """Node in a path planning graph"""
    position: Vector3D
    parent: Optional['PathNode'] = None
    g_cost: float = 0.0                 # Cost from start
    h_cost: float = 0.0                 # Heuristic cost to goal
    f_cost: float = 0.0                 # Total cost (g + h)
    timestamp: float = field(default_factory=time.time)
    velocity: Vector3D = field(default_factory=Vector3D)
    is_safe: bool = True
    clearance: float = float('inf')     # Distance to nearest obstacle
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        if not isinstance(other, PathNode):
            return False
        return (abs(self.position.x - other.position.x) < 0.1 and
                abs(self.position.y - other.position.y) < 0.1 and
                abs(self.position.z - other.position.z) < 0.1)

@dataclass
class Trajectory:
    """A complete trajectory with timing information"""
    waypoints: List[Vector3D]
    velocities: List[Vector3D] = field(default_factory=list)
    accelerations: List[Vector3D] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    total_time: float = 0.0
    total_distance: float = 0.0
    max_velocity: float = 0.0
    max_acceleration: float = 0.0
    smoothness_score: float = 0.0       # 0=rough, 1=smooth
    safety_margin: float = 0.0          # Minimum clearance along path
    
    def __post_init__(self):
        if self.waypoints:
            self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate trajectory metrics"""
        if len(self.waypoints) < 2:
            return
        
        # Calculate distances and total distance
        distances = []
        for i in range(1, len(self.waypoints)):
            dist = self.waypoints[i].distance_to(self.waypoints[i-1])
            distances.append(dist)
            self.total_distance += dist
        
        # Calculate velocities if not provided
        if not self.velocities and self.timestamps:
            self.velocities = []
            for i in range(1, len(self.waypoints)):
                dt = self.timestamps[i] - self.timestamps[i-1]
                if dt > 0:
                    displacement = self.waypoints[i] - self.waypoints[i-1]
                    velocity = displacement * (1.0 / dt)
                    self.velocities.append(velocity)
                    self.max_velocity = max(self.max_velocity, velocity.magnitude())
        
        # Calculate accelerations if not provided
        if not self.accelerations and len(self.velocities) > 1:
            self.accelerations = []
            for i in range(1, len(self.velocities)):
                if self.timestamps:
                    dt = self.timestamps[i] - self.timestamps[i-1]
                    if dt > 0:
                        dv = self.velocities[i] - self.velocities[i-1]
                        acceleration = dv * (1.0 / dt)
                        self.accelerations.append(acceleration)
                        self.max_acceleration = max(self.max_acceleration, acceleration.magnitude())
        
        # Calculate smoothness (based on acceleration changes)
        if len(self.accelerations) > 1:
            jerk_sum = 0.0
            for i in range(1, len(self.accelerations)):
                jerk = (self.accelerations[i] - self.accelerations[i-1]).magnitude()
                jerk_sum += jerk
            
            # Normalize smoothness score (lower jerk = higher smoothness)
            self.smoothness_score = max(0.0, 1.0 - jerk_sum / (len(self.accelerations) * 10.0))
    
    def get_position_at_time(self, t: float) -> Optional[Vector3D]:
        """Get interpolated position at specific time"""
        if not self.timestamps or t < self.timestamps[0] or t > self.timestamps[-1]:
            return None
        
        # Find surrounding waypoints
        for i in range(len(self.timestamps) - 1):
            if self.timestamps[i] <= t <= self.timestamps[i + 1]:
                # Linear interpolation
                dt = self.timestamps[i + 1] - self.timestamps[i]
                if dt == 0:
                    return self.waypoints[i]
                
                alpha = (t - self.timestamps[i]) / dt
                pos1 = self.waypoints[i]
                pos2 = self.waypoints[i + 1]
                
                return Vector3D(
                    pos1.x + alpha * (pos2.x - pos1.x),
                    pos1.y + alpha * (pos2.y - pos1.y),
                    pos1.z + alpha * (pos2.z - pos1.z)
                )
        
        return None

class OccupancyGrid3D:
    """3D occupancy grid for path planning"""
    
    def __init__(self, bounds: Tuple[Vector3D, Vector3D], resolution: float = 0.5):
        self.min_bounds = bounds[0]
        self.max_bounds = bounds[1]
        self.resolution = resolution
        
        # Calculate grid dimensions
        self.size_x = int((self.max_bounds.x - self.min_bounds.x) / resolution) + 1
        self.size_y = int((self.max_bounds.y - self.min_bounds.y) / resolution) + 1
        self.size_z = int((self.max_bounds.z - self.min_bounds.z) / resolution) + 1
        
        # Initialize grid (0=free, 1=occupied, 0.5=unknown)
        self.grid = np.zeros((self.size_x, self.size_y, self.size_z), dtype=np.float32)
        
        # Distance field for clearance calculations
        self.distance_field = np.full((self.size_x, self.size_y, self.size_z), float('inf'))
    
    def world_to_grid(self, pos: Vector3D) -> Tuple[int, int, int]:
        """Convert world coordinates to grid indices"""
        x = int((pos.x - self.min_bounds.x) / self.resolution)
        y = int((pos.y - self.min_bounds.y) / self.resolution)
        z = int((pos.z - self.min_bounds.z) / self.resolution)
        
        x = max(0, min(self.size_x - 1, x))
        y = max(0, min(self.size_y - 1, y))
        z = max(0, min(self.size_z - 1, z))
        
        return x, y, z
    
    def grid_to_world(self, x: int, y: int, z: int) -> Vector3D:
        """Convert grid indices to world coordinates"""
        world_x = self.min_bounds.x + x * self.resolution
        world_y = self.min_bounds.y + y * self.resolution
        world_z = self.min_bounds.z + z * self.resolution
        return Vector3D(world_x, world_y, world_z)
    
    def is_occupied(self, pos: Vector3D, safety_margin: float = 1.0) -> bool:
        """Check if position is occupied considering safety margin"""
        x, y, z = self.world_to_grid(pos)
        
        # Check neighborhood for safety margin
        margin_cells = int(safety_margin / self.resolution)
        
        for dx in range(-margin_cells, margin_cells + 1):
            for dy in range(-margin_cells, margin_cells + 1):
                for dz in range(-margin_cells, margin_cells + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (0 <= nx < self.size_x and 0 <= ny < self.size_y and 0 <= nz < self.size_z):
                        if self.grid[nx, ny, nz] > 0.5:  # Occupied threshold
                            return True
        return False
    
    def get_clearance(self, pos: Vector3D) -> float:
        """Get clearance (distance to nearest obstacle) at position"""
        x, y, z = self.world_to_grid(pos)
        if 0 <= x < self.size_x and 0 <= y < self.size_y and 0 <= z < self.size_z:
            return self.distance_field[x, y, z] * self.resolution
        return 0.0
    
    def update_obstacles(self, obstacles: List[ObstacleCluster]):
        """Update grid with current obstacles"""
        # Reset grid
        self.grid.fill(0.0)
        
        # Add obstacles
        for obstacle in obstacles:
            if obstacle.bounding_box:
                self._add_bounding_box(obstacle.bounding_box, obstacle.threat_level)
        
        # Compute distance field
        self._compute_distance_field()
    
    def _add_bounding_box(self, bbox: BoundingBox3D, threat_level: ThreatLevel):
        """Add bounding box to occupancy grid"""
        # Get grid bounds of bounding box
        half_dims = bbox.dimensions * 0.5
        min_corner = bbox.center - half_dims
        max_corner = bbox.center + half_dims
        
        min_x, min_y, min_z = self.world_to_grid(min_corner)
        max_x, max_y, max_z = self.world_to_grid(max_corner)
        
        # Set occupancy value based on threat level
        occupancy_value = min(1.0, threat_level.value / 5.0)
        
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                for z in range(min_z, max_z + 1):
                    if 0 <= x < self.size_x and 0 <= y < self.size_y and 0 <= z < self.size_z:
                        self.grid[x, y, z] = max(self.grid[x, y, z], occupancy_value)
    
    def _compute_distance_field(self):
        """Compute distance field using Euclidean distance transform"""
        # Simple distance field computation (in production, use optimized algorithms)
        self.distance_field.fill(float('inf'))
        
        # Find all occupied cells
        occupied_cells = np.where(self.grid > 0.5)
        
        # For each free cell, compute distance to nearest occupied cell
        for x in range(self.size_x):
            for y in range(self.size_y):
                for z in range(self.size_z):
                    if self.grid[x, y, z] <= 0.5:  # Free cell
                        min_dist = float('inf')
                        
                        # Find nearest occupied cell
                        for i in range(len(occupied_cells[0])):
                            ox, oy, oz = occupied_cells[0][i], occupied_cells[1][i], occupied_cells[2][i]
                            dist = math.sqrt((x - ox)**2 + (y - oy)**2 + (z - oz)**2)
                            min_dist = min(min_dist, dist)
                        
                        self.distance_field[x, y, z] = min_dist

class AStarPlanner:
    """A* path planning algorithm"""
    
    def __init__(self, grid: OccupancyGrid3D):
        self.grid = grid
        self.directions = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
            (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
            (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
        ]
    
    def plan(self, start: Vector3D, goal: Vector3D, max_iterations: int = 10000) -> List[Vector3D]:
        """Plan path using A* algorithm"""
        start_node = PathNode(position=start)
        goal_node = PathNode(position=goal)
        
        open_set = []
        closed_set = set()
        
        heapq.heappush(open_set, start_node)
        
        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            
            current = heapq.heappop(open_set)
            
            # Check if we reached the goal
            if current.position.distance_to(goal) < self.grid.resolution:
                return self._reconstruct_path(current)
            
            closed_set.add(self._position_hash(current.position))
            
            # Explore neighbors
            for direction in self.directions:
                neighbor_pos = Vector3D(
                    current.position.x + direction[0] * self.grid.resolution,
                    current.position.y + direction[1] * self.grid.resolution,
                    current.position.z + direction[2] * self.grid.resolution
                )
                
                neighbor_hash = self._position_hash(neighbor_pos)
                if neighbor_hash in closed_set:
                    continue
                
                # Check if neighbor is valid
                if self.grid.is_occupied(neighbor_pos, safety_margin=1.0):
                    continue
                
                # Calculate costs
                move_cost = math.sqrt(sum(d*d for d in direction)) * self.grid.resolution
                tentative_g = current.g_cost + move_cost
                
                neighbor = PathNode(
                    position=neighbor_pos,
                    parent=current,
                    g_cost=tentative_g,
                    h_cost=self._heuristic(neighbor_pos, goal),
                    clearance=self.grid.get_clearance(neighbor_pos)
                )
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                
                # Add clearance bonus to f_cost (prefer paths with more clearance)
                clearance_bonus = -min(2.0, neighbor.clearance) * 0.1
                neighbor.f_cost += clearance_bonus
                
                # Check if this path to neighbor is better
                existing_neighbor = None
                for node in open_set:
                    if self._position_hash(node.position) == neighbor_hash:
                        existing_neighbor = node
                        break
                
                if existing_neighbor is None or tentative_g < existing_neighbor.g_cost:
                    if existing_neighbor:
                        open_set.remove(existing_neighbor)
                        heapq.heapify(open_set)
                    
                    heapq.heappush(open_set, neighbor)
        
        return []  # No path found
    
    def _heuristic(self, pos1: Vector3D, pos2: Vector3D) -> float:
        """Heuristic function for A* (Euclidean distance)"""
        return pos1.distance_to(pos2)
    
    def _position_hash(self, pos: Vector3D) -> str:
        """Create hash for position"""
        x, y, z = self.grid.world_to_grid(pos)
        return f"{x},{y},{z}"
    
    def _reconstruct_path(self, node: PathNode) -> List[Vector3D]:
        """Reconstruct path from goal to start"""
        path = []
        current = node
        while current:
            path.append(current.position)
            current = current.parent
        path.reverse()
        return path

class RRTStarPlanner:
    """RRT* sampling-based path planning"""
    
    def __init__(self, grid: OccupancyGrid3D, params: NavigationParams):
        self.grid = grid
        self.params = params
        self.nodes: List[PathNode] = []
        self.max_step_size = 2.0
        self.rewiring_radius = 3.0
        self.goal_bias = 0.1
    
    def plan(self, start: Vector3D, goal: Vector3D, max_iterations: int = 1000) -> List[Vector3D]:
        """Plan path using RRT* algorithm"""
        start_node = PathNode(position=start)
        self.nodes = [start_node]
        
        goal_node = None
        best_goal_cost = float('inf')
        
        for iteration in range(max_iterations):
            # Sample random point (with goal bias)
            if random.random() < self.goal_bias:
                sample_point = goal
            else:
                sample_point = self._sample_free_space()
            
            # Find nearest node
            nearest_node = self._find_nearest(sample_point)
            
            # Steer towards sample
            new_pos = self._steer(nearest_node.position, sample_point)
            
            # Check if path is collision-free
            if not self._is_path_valid(nearest_node.position, new_pos):
                continue
            
            # Create new node
            new_node = PathNode(
                position=new_pos,
                parent=nearest_node,
                g_cost=nearest_node.g_cost + nearest_node.position.distance_to(new_pos),
                clearance=self.grid.get_clearance(new_pos)
            )
            
            # Find nearby nodes for rewiring
            nearby_nodes = self._find_nearby_nodes(new_node, self.rewiring_radius)
            
            # Choose best parent
            best_parent = nearest_node
            best_cost = new_node.g_cost
            
            for nearby_node in nearby_nodes:
                cost = nearby_node.g_cost + nearby_node.position.distance_to(new_pos)
                if (cost < best_cost and 
                    self._is_path_valid(nearby_node.position, new_pos)):
                    best_parent = nearby_node
                    best_cost = cost
            
            new_node.parent = best_parent
            new_node.g_cost = best_cost
            self.nodes.append(new_node)
            
            # Rewire nearby nodes
            for nearby_node in nearby_nodes:
                cost = new_node.g_cost + new_node.position.distance_to(nearby_node.position)
                if (cost < nearby_node.g_cost and 
                    self._is_path_valid(new_node.position, nearby_node.position)):
                    nearby_node.parent = new_node
                    nearby_node.g_cost = cost
            
            # Check if we can connect to goal
            if new_node.position.distance_to(goal) < self.max_step_size:
                if self._is_path_valid(new_node.position, goal):
                    goal_cost = new_node.g_cost + new_node.position.distance_to(goal)
                    if goal_cost < best_goal_cost:
                        goal_node = PathNode(position=goal, parent=new_node, g_cost=goal_cost)
                        best_goal_cost = goal_cost
        
        if goal_node:
            return self._reconstruct_path(goal_node)
        return []
    
    def _sample_free_space(self) -> Vector3D:
        """Sample random point in free space"""
        while True:
            x = random.uniform(self.grid.min_bounds.x, self.grid.max_bounds.x)
            y = random.uniform(self.grid.min_bounds.y, self.grid.max_bounds.y)
            z = random.uniform(self.grid.min_bounds.z, self.grid.max_bounds.z)
            
            pos = Vector3D(x, y, z)
            if not self.grid.is_occupied(pos):
                return pos
    
    def _find_nearest(self, pos: Vector3D) -> PathNode:
        """Find nearest node to given position"""
        min_dist = float('inf')
        nearest = self.nodes[0]
        
        for node in self.nodes:
            dist = node.position.distance_to(pos)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _steer(self, from_pos: Vector3D, to_pos: Vector3D) -> Vector3D:
        """Steer from one position towards another"""
        direction = to_pos - from_pos
        distance = direction.magnitude()
        
        if distance <= self.max_step_size:
            return to_pos
        
        normalized = direction.normalize()
        return from_pos + normalized * self.max_step_size
    
    def _is_path_valid(self, start: Vector3D, end: Vector3D) -> bool:
        """Check if path between two points is valid"""
        steps = int(start.distance_to(end) / (self.grid.resolution * 0.5)) + 1
        
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            pos = Vector3D(
                start.x + t * (end.x - start.x),
                start.y + t * (end.y - start.y),
                start.z + t * (end.z - start.z)
            )
            
            if self.grid.is_occupied(pos):
                return False
        
        return True
    
    def _find_nearby_nodes(self, node: PathNode, radius: float) -> List[PathNode]:
        """Find all nodes within radius"""
        nearby = []
        for other_node in self.nodes:
            if other_node != node and node.position.distance_to(other_node.position) <= radius:
                nearby.append(other_node)
        return nearby
    
    def _reconstruct_path(self, node: PathNode) -> List[Vector3D]:
        """Reconstruct path from goal to start"""
        path = []
        current = node
        while current:
            path.append(current.position)
            current = current.parent
        path.reverse()
        return path

class DynamicWindowApproach:
    """Dynamic Window Approach for local path planning"""
    
    def __init__(self, params: NavigationParams):
        self.params = params
        self.velocity_samples = 50
        self.angular_samples = 20
        self.prediction_time = 2.0
        
    def plan(self, current_pos: Vector3D, current_vel: Vector3D, goal: Vector3D,
             obstacles: List[ObstacleCluster]) -> Tuple[Vector3D, Vector3D]:
        """Plan next velocity command using DWA"""
        
        best_velocity = current_vel
        best_score = float('-inf')
        
        # Generate velocity samples
        max_vel = self.params.max_speed
        max_angular_vel = self.params.max_angular_velocity
        
        for v_x in np.linspace(-max_vel, max_vel, self.velocity_samples):
            for v_y in np.linspace(-max_vel, max_vel, self.velocity_samples):
                for omega in np.linspace(-max_angular_vel, max_angular_vel, self.angular_samples):
                    
                    velocity = Vector3D(v_x, v_y, 0)
                    
                    # Check velocity constraints
                    if velocity.magnitude() > max_vel:
                        continue
                    
                    # Predict trajectory
                    trajectory = self._predict_trajectory(current_pos, velocity, omega)
                    
                    # Evaluate trajectory
                    score = self._evaluate_trajectory(trajectory, goal, obstacles)
                    
                    if score > best_score:
                        best_score = score
                        best_velocity = velocity
        
        return best_velocity, goal  # Return velocity and next waypoint
    
    def _predict_trajectory(self, start_pos: Vector3D, velocity: Vector3D, 
                          angular_vel: float) -> List[Vector3D]:
        """Predict trajectory given velocity commands"""
        trajectory = []
        pos = start_pos
        dt = 0.1  # Time step
        steps = int(self.prediction_time / dt)
        
        for _ in range(steps):
            pos = pos + velocity * dt
            trajectory.append(Vector3D(pos.x, pos.y, pos.z))
        
        return trajectory
    
    def _evaluate_trajectory(self, trajectory: List[Vector3D], goal: Vector3D,
                           obstacles: List[ObstacleCluster]) -> float:
        """Evaluate trajectory quality"""
        if not trajectory:
            return float('-inf')
        
        # Distance to goal (closer is better)
        goal_distance = trajectory[-1].distance_to(goal)
        goal_score = 1.0 / (1.0 + goal_distance)
        
        # Obstacle avoidance (farther from obstacles is better)
        min_obstacle_distance = float('inf')
        for pos in trajectory:
            for obstacle in obstacles:
                dist = pos.distance_to(obstacle.center)
                min_obstacle_distance = min(min_obstacle_distance, dist)
        
        if min_obstacle_distance < 1.0:  # Collision
            return float('-inf')
        
        obstacle_score = min(1.0, min_obstacle_distance / 5.0)
        
        # Velocity preference (maintain reasonable speed)
        velocity_score = 0.5  # Neutral score for velocity
        
        # Combine scores
        total_score = (0.4 * goal_score + 
                      0.4 * obstacle_score + 
                      0.2 * velocity_score)
        
        return total_score

class PathSmoother:
    """Path smoothing and optimization"""
    
    def __init__(self, grid: OccupancyGrid3D):
        self.grid = grid
    
    def smooth_path(self, path: List[Vector3D], iterations: int = 10) -> List[Vector3D]:
        """Smooth path using iterative optimization"""
        if len(path) < 3:
            return path
        
        smoothed = path.copy()
        
        for _ in range(iterations):
            for i in range(1, len(smoothed) - 1):
                # Average with neighbors
                prev_pos = smoothed[i - 1]
                next_pos = smoothed[i + 1]
                
                new_pos = Vector3D(
                    (prev_pos.x + next_pos.x) / 2,
                    (prev_pos.y + next_pos.y) / 2,
                    (prev_pos.z + next_pos.z) / 2
                )
                
                # Check if new position is valid
                if not self.grid.is_occupied(new_pos):
                    # Check if path segments are valid
                    if (self._is_path_valid(prev_pos, new_pos) and
                        self._is_path_valid(new_pos, next_pos)):
                        smoothed[i] = new_pos
        
        return smoothed
    
    def _is_path_valid(self, start: Vector3D, end: Vector3D) -> bool:
        """Check if path segment is collision-free"""
        steps = int(start.distance_to(end) / (self.grid.resolution * 0.5)) + 1
        
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            pos = Vector3D(
                start.x + t * (end.x - start.x),
                start.y + t * (end.y - start.y),
                start.z + t * (end.z - start.z)
            )
            
            if self.grid.is_occupied(pos):
                return False
        
        return True

class PathPlanner:
    """Enhanced path planning system with multiple algorithms"""
    
    def __init__(self, params: NavigationParams):
        self.params = params
        
        # Initialize grid bounds (adjust based on your environment)
        min_bounds = Vector3D(-50, -50, -20)
        max_bounds = Vector3D(50, 50, 5)
        self.grid = OccupancyGrid3D((min_bounds, max_bounds), params.path_planner.grid_resolution)
        
        # Initialize planners
        self.astar = AStarPlanner(self.grid)
        self.rrt_star = RRTStarPlanner(self.grid, params)
        self.dwa = DynamicWindowApproach(params)
        self.smoother = PathSmoother(self.grid)
        
        # Planning state
        self.current_planner = PlannerType.HYBRID
        self.last_global_plan_time = 0.0
        self.global_path: List[Vector3D] = []
        self.local_path: List[Vector3D] = []
        
        # Legacy compatibility
        self.waypoints = deque()
        self.current_strategy = "hybrid"
        self.previous_obstacle_info = None
    
    def plan_avoidance_path(self, current_pos: List[float], 
                          destination: List[float], 
                          obstacle_info: ObstacleInfo) -> List[List[float]]:
        """Main path planning interface (legacy compatibility)"""
        
        # Convert to Vector3D
        start = Vector3D(current_pos[0], current_pos[1], current_pos[2])
        goal = Vector3D(destination[0], destination[1], destination[2])
        
        # Update occupancy grid
        self.grid.update_obstacles(obstacle_info.clusters)
        
        # Choose planning strategy
        strategy = self._select_planning_strategy(obstacle_info)
        
        # Plan path
        path = self._plan_path(start, goal, obstacle_info, strategy)
        
        # Convert back to legacy format
        legacy_path = []
        for waypoint in path:
            legacy_path.append([waypoint.x, waypoint.y, waypoint.z])
        
        return legacy_path
    
    def _select_planning_strategy(self, obstacle_info: ObstacleInfo) -> PlannerType:
        """Select best planning algorithm based on situation"""
        
        # Emergency situations
        if obstacle_info.max_threat_level >= ThreatLevel.CRITICAL:
            return PlannerType.EMERGENCY
        
        # High threat situations - use fast local planning
        if obstacle_info.max_threat_level >= ThreatLevel.HIGH:
            return PlannerType.DWA
        
        # Complex environments - use RRT*
        if len(obstacle_info.clusters) > 5:
            return PlannerType.RRT_STAR
        
        # Simple environments - use A*
        if len(obstacle_info.clusters) <= 2:
            return PlannerType.A_STAR
        
        # Default to hybrid approach
        return PlannerType.HYBRID
    
    def _plan_path(self, start: Vector3D, goal: Vector3D, 
                  obstacle_info: ObstacleInfo, strategy: PlannerType) -> List[Vector3D]:
        """Plan path using selected strategy"""
        
        current_time = time.time()
        
        if strategy == PlannerType.EMERGENCY:
            return self._plan_emergency_path(start, goal, obstacle_info)
        
        elif strategy == PlannerType.A_STAR:
            path = self.astar.plan(start, goal)
            if path:
                return self.smoother.smooth_path(path)
            return [goal]
        
        elif strategy == PlannerType.RRT_STAR:
            path = self.rrt_star.plan(start, goal)
            if path:
                return self.smoother.smooth_path(path)
            return [goal]
        
        elif strategy == PlannerType.DWA:
            # DWA returns next velocity, convert to waypoint
            next_vel, next_waypoint = self.dwa.plan(
                start, Vector3D(), goal, obstacle_info.clusters)
            
            # Create short path in direction of next velocity
            next_pos = start + next_vel * 0.5  # 0.5 second ahead
            return [next_pos, goal]
        
        elif strategy == PlannerType.HYBRID:
            # Use global A* planning with local DWA refinement
            need_global_replan = (
                current_time - self.last_global_plan_time > 
                self.params.path_planner.global_replanning_interval or
                not self.global_path
            )
            
            if need_global_replan:
                self.global_path = self.astar.plan(start, goal)
                if self.global_path:
                    self.global_path = self.smoother.smooth_path(self.global_path)
                self.last_global_plan_time = current_time
            
            # Use DWA for local planning along global path
            if self.global_path:
                # Find next waypoint on global path
                next_global_waypoint = self._find_next_global_waypoint(start)
                if next_global_waypoint:
                    next_vel, local_waypoint = self.dwa.plan(
                        start, Vector3D(), next_global_waypoint, obstacle_info.clusters)
                    
                    # Create local path
                    local_pos = start + next_vel * 0.2
                    return [local_pos, next_global_waypoint]
            
            return [goal]
        
        # Fallback - direct path
        return [goal]
    
    def _plan_emergency_path(self, start: Vector3D, goal: Vector3D, 
                           obstacle_info: ObstacleInfo) -> List[Vector3D]:
        """Plan emergency avoidance path"""
        
        # Find immediate threats
        immediate_threats = [c for c in obstacle_info.clusters 
                           if c.threat_level >= ThreatLevel.CRITICAL]
        
        if not immediate_threats:
            return [goal]
        
        # Find escape direction
        escape_direction = Vector3D()
        
        for threat in immediate_threats:
            # Vector away from threat
            away_vector = start - threat.center
            if away_vector.magnitude() > 0:
                away_vector = away_vector.normalize()
                escape_direction = escape_direction + away_vector
        
        if escape_direction.magnitude() > 0:
            escape_direction = escape_direction.normalize()
            
            # Move away from threats
            escape_distance = max(5.0, self.params.retreat_distance)
            escape_position = start + escape_direction * escape_distance
            
            return [escape_position]
        
        # If no clear escape direction, move up
        return [Vector3D(start.x, start.y, start.z - 3.0)]
    
    def _find_next_global_waypoint(self, current_pos: Vector3D) -> Optional[Vector3D]:
        """Find next waypoint on global path"""
        if not self.global_path:
            return None
        
        # Find closest point on global path
        min_dist = float('inf')
        closest_idx = 0
        
        for i, waypoint in enumerate(self.global_path):
            dist = current_pos.distance_to(waypoint)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Return next waypoint (or goal if at end)
        lookahead_idx = min(closest_idx + 2, len(self.global_path) - 1)
        return self.global_path[lookahead_idx]
    
    def generate_trajectory(self, path: List[Vector3D], max_velocity: float = 5.0,
                          max_acceleration: float = 2.0) -> Trajectory:
        """Generate time-optimal trajectory from path"""
        if len(path) < 2:
            return Trajectory(waypoints=path)
        
        # Calculate timing for each segment
        timestamps = [0.0]
        velocities = []
        
        current_time = 0.0
        current_velocity = 0.0
        
        for i in range(1, len(path)):
            segment_distance = path[i].distance_to(path[i-1])
            
            # Calculate maximum velocity for this segment
            segment_max_vel = max_velocity
            
            # Reduce velocity for sharp turns
            if i < len(path) - 1:
                # Calculate turn angle
                v1 = path[i] - path[i-1]
                v2 = path[i+1] - path[i]
                if v1.magnitude() > 0 and v2.magnitude() > 0:
                    dot_product = (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z) / (v1.magnitude() * v2.magnitude())
                    angle = math.acos(max(-1, min(1, dot_product)))
                    
                    # Reduce velocity for sharp turns
                    if angle > math.pi / 4:  # 45 degrees
                        segment_max_vel *= 0.5
            
            # Time to complete segment with acceleration constraints
            if current_velocity < segment_max_vel:
                # Acceleration phase
                accel_time = min((segment_max_vel - current_velocity) / max_acceleration,
                               segment_distance / (current_velocity + 0.5 * max_acceleration))
                accel_distance = current_velocity * accel_time + 0.5 * max_acceleration * accel_time**2
                
                if accel_distance >= segment_distance:
                    # Pure acceleration
                    segment_time = accel_time
                    final_velocity = current_velocity + max_acceleration * accel_time
                else:
                    # Acceleration + constant velocity
                    const_distance = segment_distance - accel_distance
                    const_time = const_distance / segment_max_vel
                    segment_time = accel_time + const_time
                    final_velocity = segment_max_vel
            else:
                # Constant or deceleration
                segment_time = segment_distance / min(current_velocity, segment_max_vel)
                final_velocity = segment_max_vel
            
            current_time += segment_time
            current_velocity = final_velocity
            
            timestamps.append(current_time)
            
            # Calculate velocity vector
            direction = path[i] - path[i-1]
            if direction.magnitude() > 0:
                direction = direction.normalize()
                velocity_vector = direction * final_velocity
                velocities.append(velocity_vector)
        
        return Trajectory(
            waypoints=path,
            velocities=velocities,
            timestamps=timestamps,
            total_time=current_time,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration
        )