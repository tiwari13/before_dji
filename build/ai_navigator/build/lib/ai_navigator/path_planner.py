from dataclasses import dataclass
from typing import List, Optional
import math
from collections import deque
import time
from ai_navigator.obstacle_data import ObstacleInfo, ObstacleCluster
from ai_navigator.navigation_params import NavigationParams
@dataclass
class PathPlanner:
    """Enhanced path planning for obstacle avoidance"""
    
    def __init__(self, params: NavigationParams):
        self.params = params
        self.waypoints = deque()
        self.current_strategy = "direct"
        self.previous_obstacle_info = None
        
    def plan_avoidance_path(self, current_pos: List[float], 
                          destination: List[float], 
                          obstacle_info: ObstacleInfo) -> List[List[float]]:
        """Plan a path around obstacles with multiple strategies"""
        if not obstacle_info.detected:
            self.current_strategy = "direct"
            return [destination]
            
        if self._obstacles_changed_significantly(obstacle_info):
            self.current_strategy = self._select_strategy(obstacle_info)
            
        self.previous_obstacle_info = obstacle_info
        
        if self.current_strategy == "vertical":
            return self._plan_vertical_avoidance(current_pos, destination)
        elif self.current_strategy == "retreat":
            return self._plan_retreat_path(current_pos, obstacle_info)
        elif self.current_strategy == "circle":
            return self._plan_circling_path(current_pos, destination)
        else:
            return self._plan_lateral_avoidance(current_pos, destination, obstacle_info)
    
    def _obstacles_changed_significantly(self, new_obstacle_info: ObstacleInfo) -> bool:
        """Check if obstacle situation has changed significantly"""
        if not self.previous_obstacle_info:
            return True
            
        old_clusters = self.previous_obstacle_info.clusters
        new_clusters = new_obstacle_info.clusters
        
        if len(old_clusters) != len(new_clusters):
            return True
            
        for old_c, new_c in zip(old_clusters, new_clusters):
            if (abs(old_c.center_x - new_c.center_x) > 50 or
                abs(old_c.center_y - new_c.center_y) > 50 or
                abs(old_c.min_distance - new_c.min_distance) > 0.5):
                return True
                
        return False
    
    def _select_strategy(self, obstacle_info: ObstacleInfo) -> str:
        """Select avoidance strategy based on obstacle characteristics"""
        cluster_count = len(obstacle_info.clusters)
        min_distance = obstacle_info.min_distance
        
        if min_distance < self.params.emergency_threshold:
            return "retreat"
        elif cluster_count > 3 or min_distance < self.params.critical_threshold:
            return "vertical"
        elif len(obstacle_info.safe_directions) <= 1:
            return "circle"
        else:
            return "lateral"
    
    def _plan_vertical_avoidance(self, current_pos: List[float], 
                               destination: List[float]) -> List[List[float]]:
        """Plan vertical avoidance with altitude limits"""
        target_altitude = max(self.params.max_climb_altitude, 
                            current_pos[2] - self.params.vertical_climb_height)
        
        waypoints = []
        waypoints.append([current_pos[0], current_pos[1], target_altitude])
        waypoints.append([destination[0], destination[1], target_altitude])
        waypoints.append([destination[0], destination[1], destination[2]])
        
        return waypoints
    
    def _plan_retreat_path(self, current_pos: List[float], 
                         obstacle_info: ObstacleInfo) -> List[List[float]]:
        """Plan retreat path with adaptive distance"""
        retreat_dist = self.params.retreat_distance * (self.params.critical_threshold / 
                                                     max(obstacle_info.min_distance, 0.1))
        retreat_pos = [
            current_pos[0] - retreat_dist,
            current_pos[1],
            current_pos[2]
        ]
        return [retreat_pos]
    
    def _plan_circling_path(self, current_pos: List[float], 
                          destination: List[float]) -> List[List[float]]:
        """Plan circling path to find clear route"""
        angle = (time.time() * 0.3) % (2 * math.pi)
        circle_x = current_pos[0] + self.params.circle_radius * math.cos(angle)
        circle_y = current_pos[1] + self.params.circle_radius * math.sin(angle)
        
        return [[circle_x, circle_y, current_pos[2]]]
    
    def _plan_lateral_avoidance(self, current_pos: List[float], 
                              destination: List[float], 
                              obstacle_info: ObstacleInfo) -> List[List[float]]:
        """Plan lateral avoidance with continuous path adjustment"""
        waypoints = []
        current_point = current_pos.copy()
        
        dx = destination[0] - current_pos[0]
        dy = destination[1] - current_pos[1]
        distance_to_destination = math.sqrt(dx*dx + dy*dy)
        
        step_size = min(self.params.avoidance_offset / 2, distance_to_destination)
        steps = max(1, int(distance_to_destination / step_size))
        
        attempts = 0
        max_attempts = 5
        
        while distance_to_destination > step_size and attempts < max_attempts:
            # Calculate next step towards destination
            next_x = current_point[0] + (dx / distance_to_destination) * step_size
            next_y = current_point[1] + (dy / distance_to_destination) * step_size
            next_point = [next_x, next_y, current_point[2]]
            
            if self._is_safe_path(current_point, next_point, obstacle_info):
                waypoints.append(next_point)
                current_point = next_point
            else:
                # Try detours in multiple directions
                detour_found = False
                best_detour = None
                best_score = float('inf')
                angles = [math.pi/2, -math.pi/2, math.pi/4, -math.pi/4, 3*math.pi/4, -3*math.pi/4]
                
                for angle in angles:
                    offset_x = self.params.avoidance_offset * math.cos(angle)
                    offset_y = self.params.avoidance_offset * math.sin(angle)
                    detour_point = [
                        current_point[0] + offset_x,
                        current_point[1] + offset_y,
                        current_point[2]
                    ]
                    
                    if self._is_safe_path(current_point, detour_point, obstacle_info):
                        detour_dx = destination[0] - detour_point[0]
                        detour_dy = destination[1] - detour_point[1]
                        detour_distance = math.sqrt(detour_dx*detour_dx + detour_dy*detour_dy)
                        score = detour_distance
                        
                        if score < best_score:
                            best_score = score
                            best_detour = detour_point
                            detour_found = True
                
                if detour_found:
                    waypoints.append(best_detour)
                    current_point = best_detour
                    self.current_strategy = "lateral"
                else:
                    attempts += 1
                    self.current_strategy = "vertical"
                    return self._plan_vertical_avoidance(current_pos, destination)
            
            # Recalculate distance to destination
            dx = destination[0] - current_point[0]
            dy = destination[1] - current_point[1]
            distance_to_destination = math.sqrt(dx*dx + dy*dy)
        
        if distance_to_destination <= step_size:
            waypoints.append(destination)
        
        return waypoints
    
    def _is_safe_path(self, start: List[float], end: List[float], 
                     obstacle_info: ObstacleInfo) -> bool:
        """Check if the path between start and end is safe using real-world coordinates"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        path_length = math.sqrt(dx*dx + dy*dy)
        
        if path_length == 0:
            return True
        
        steps = max(1, int(path_length / 0.2))  # Check every 0.2 meters
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            x = start[0] + t * dx
            y = start[1] + t * dy
            point = [x, y, start[2]]
            
            for cluster in obstacle_info.clusters:
                distance_to_obstacle = math.sqrt(
                    (point[0] - cluster.world_x)**2 + 
                    (point[1] - cluster.world_y)**2
                )
                if distance_to_obstacle < self.params.obstacle_buffer_zone:
                    return False
        
        return True