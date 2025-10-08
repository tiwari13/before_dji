from ai_navigator.navigation_params import NavigationParams
from ai_navigator.path_planner import PathPlanner
from ai_navigator.obstacle_data import ObstacleInfo, ObstacleCluster
from ai_navigator.drone_state import DroneState
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition
from sensor_msgs.msg import Image as ROSImage
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import math
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List
import time
from collections import deque
from typing import Dict, Any


class SmartNavigator(Node):
    def __init__(self):
        super().__init__('smart_navigator')
        
        self.nav_params = NavigationParams()
        self.path_planner = PathPlanner(self.nav_params)
        
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self._setup_publishers()
        self._setup_subscribers()
        self._initialize_state()
        self._setup_computer_vision()

        self.timer = self.create_timer(0.05, self.timer_callback)
        
        self.get_logger().info("Enhanced SmartNavigator initialized successfully")

    def _setup_publishers(self):
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', self.qos_profile)
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', self.qos_profile)
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', self.qos_profile)

    def _setup_subscribers(self):
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', 
            self.position_callback, self.qos_profile)
        self.image_sub = self.create_subscription(
            Image, '/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image', 
            self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            ROSImage, '/depth_camera', self.depth_callback, 10)

    def _initialize_state(self):
        self.current_position: Optional[List[float]] = None
        self.state = DroneState.INIT
        self.obstacle_info = ObstacleInfo()
        self.depth_image: Optional[np.ndarray] = None
        
        self.takeoff_point = [0.0, 0.0, self.nav_params.takeoff_altitude]
        self.destination_point = [0.0, self.nav_params.destination_offset, self.nav_params.takeoff_altitude]
        self.current_waypoint = None
        self.waypoint_queue = deque()
        
        self.avoidance_start_time: Optional[float] = None
        self.yaw = 0.0
        self.offboard_setpoint_counter = 0
        self.stuck_counter = 0
        self.last_position = None
        self.movement_threshold = 0.05
        
        self.last_position_update = time.time()
        self.position_timeout = 3.0
        self.emergency_stop = False
        self.stuck_recovery_attempts = 0
        self.max_recovery_attempts = 3
        self.last_path_plan_time = 0.0
        self.path_plan_interval = 0.2  # Replan more frequently

    def _setup_computer_vision(self):
        self.bridge = CvBridge()
        try:
            self.model = YOLO("yolo11s.pt")
            self.get_logger().info("YOLO model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            self.state = DroneState.EMERGENCY

    def position_callback(self, msg):
        self.current_position = [msg.x, msg.y, msg.z]
        self.last_position_update = time.time()

    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_image = np.array(depth_image, dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"Depth callback failed: {e}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.obstacle_info = self._process_frame_enhanced(frame)
        except Exception as e:
            self.get_logger().error(f"Image processing failed: {e}")

    def _process_frame_enhanced(self, frame: np.ndarray) -> ObstacleInfo:
        if self.model is None:
            return ObstacleInfo()
            
        results = self.model.predict(
            source=frame, 
            conf=self.nav_params.yolo_confidence, 
            stream=False
        )

        detections = []
        frame_height, frame_width = frame.shape[:2]
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                depth = self._get_depth_at_point(cx, cy)
                
                if depth and depth < self.nav_params.obstacle_threshold:
                    # Convert pixel coordinates to angles
                    angle_x = ((cx - frame_width / 2) / frame_width) * self.nav_params.camera_fov_horizontal
                    angle_y = ((cy - frame_height / 2) / frame_height) * self.nav_params.camera_fov_vertical
                    
                    # Convert to real-world coordinates relative to drone
                    world_x = depth * math.tan(angle_x)
                    world_y = depth * math.tan(angle_y)
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': (cx, cy),
                        'depth': depth,
                        'world_x': world_x,
                        'world_y': world_y,
                        'conf': float(box.conf[0]),
                        'cls': int(box.cls[0])
                    })

        clusters = self._create_obstacle_clusters(detections, frame.shape)
        safe_dirs, blocked_dirs, escape_angle = self._analyze_safe_directions(clusters, frame.shape)
        
        obstacle_info = ObstacleInfo(
            detected=len(clusters) > 0,
            clusters=clusters,
            min_distance=min([c.min_distance for c in clusters]) if clusters else float('inf'),
            safe_directions=safe_dirs,
            blocked_directions=blocked_dirs,
            escape_angle=escape_angle
        )

        self._draw_enhanced_visualization(frame, clusters, safe_dirs, blocked_dirs)
        
        return obstacle_info

    def _create_obstacle_clusters(self, detections: List[dict], frame_shape: Tuple[int, int]) -> List[ObstacleCluster]:
        if not detections:
            return []
            
        clusters = []
        frame_height, frame_width = frame_shape[:2]
        used = [False] * len(detections)
        
        for i, det in enumerate(detections):
            if used[i]:
                continue
                
            cluster_detections = [det]
            used[i] = True
            
            for j, other_det in enumerate(detections):
                if used[j]:
                    continue
                    
                dx = det['center'][0] - other_det['center'][0]
                dy = det['center'][1] - other_det['center'][1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < self.nav_params.cluster_merge_threshold:
                    cluster_detections.append(other_det)
                    used[j] = True
            
            if cluster_detections:
                cluster = self._create_cluster_from_detections(cluster_detections, frame_width)
                clusters.append(cluster)
        
        clusters = self._merge_overlapping_clusters(clusters)
        
        return clusters

    def _merge_overlapping_clusters(self, clusters: List[ObstacleCluster]) -> List[ObstacleCluster]:
        if len(clusters) <= 1:
            return clusters
            
        merged = []
        used = [False] * len(clusters)
        
        for i, c1 in enumerate(clusters):
            if used[i]:
                continue
                
            cluster_dets = [c1]
            used[i] = True
            
            for j, c2 in enumerate(clusters):
                if used[j]:
                    continue
                    
                dx = c1.center_x - c2.center_x
                dy = c1.center_y - c2.center_y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < self.nav_params.cluster_merge_threshold * 1.5:
                    cluster_dets.append(c2)
                    used[j] = True
            
            if cluster_dets:
                merged.append(self._create_cluster_from_clusters(cluster_dets))
        
        return merged

    def _create_cluster_from_clusters(self, clusters: List[ObstacleCluster]) -> ObstacleCluster:
        min_x = min(c.center_x - c.width/2 for c in clusters)
        max_x = max(c.center_x + c.width/2 for c in clusters)
        min_y = min(c.center_y - c.height/2 for c in clusters)
        max_y = max(c.center_y + c.height/2 for c in clusters)
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        width = max_x - min_x
        height = max_y - min_y
        
        min_distance = min(c.min_distance for c in clusters)
        pixel_count = sum(c.pixel_count for c in clusters)
        danger_level = max(c.danger_level for c in clusters)
        
        # Average the world coordinates
        world_x = sum(c.world_x for c in clusters) / len(clusters)
        world_y = sum(c.world_y for c in clusters) / len(clusters)
        
        return ObstacleCluster(
            center_x=center_x,
            center_y=center_y,
            min_distance=min_distance,
            width=width,
            height=height,
            pixel_count=pixel_count,
            danger_level=danger_level,
            world_x=world_x,
            world_y=world_y
        )

    def _create_cluster_from_detections(self, detections: List[dict], frame_width: int) -> ObstacleCluster:
        min_x = min(det['bbox'][0] for det in detections)
        max_x = max(det['bbox'][2] for det in detections)
        min_y = min(det['bbox'][1] for det in detections)
        max_y = max(det['bbox'][3] for det in detections)
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        width = max_x - min_x
        height = max_y - min_y
        
        min_distance = min(det['depth'] for det in detections)
        pixel_count = len(detections)
        
        world_x = sum(det['world_x'] for det in detections) / len(detections)
        world_y = sum(det['world_y'] for det in detections) / len(detections)
        
        if min_distance < self.nav_params.emergency_threshold:
            danger_level = 3
        elif min_distance < self.nav_params.critical_threshold:
            danger_level = 2
        else:
            danger_level = 1
            
        return ObstacleCluster(
            center_x=center_x,
            center_y=center_y,
            min_distance=min_distance,
            width=width,
            height=height,
            pixel_count=pixel_count,
            danger_level=danger_level,
            world_x=world_x,
            world_y=world_y
        )

    def _analyze_safe_directions(self, clusters: List[ObstacleCluster], 
                               frame_shape: Tuple[int, int]) -> Tuple[List[str], List[str], Optional[float]]:
        if not clusters:
            return ["left", "right", "up", "forward"], [], None
            
        frame_height, frame_width = frame_shape[:2]
        
        left_third = frame_width // 3
        right_third = 2 * frame_width // 3
        
        left_blocked = any(c.center_x < left_third and c.danger_level > 1 for c in clusters)
        center_blocked = any(left_third <= c.center_x <= right_third and c.danger_level > 1 for c in clusters)
        right_blocked = any(c.center_x > right_third and c.danger_level > 1 for c in clusters)
        
        gaps = self._find_gaps_between_clusters(clusters, frame_width)
        
        safe_dirs = []
        blocked_dirs = []
        
        if not left_blocked or any(gap[0] < left_third for gap in gaps):
            safe_dirs.append("left")
        else:
            blocked_dirs.append("left")
            
        if not right_blocked or any(gap[0] > right_third for gap in gaps):
            safe_dirs.append("right")
        else:
            blocked_dirs.append("right")
            
        if not center_blocked or any(left_third <= gap[0] <= right_third for gap in gaps):
            safe_dirs.append("forward")
        else:
            blocked_dirs.append("forward")
            
        safe_dirs.append("up")
        
        escape_angle = None
        if safe_dirs and "forward" not in safe_dirs:
            if gaps:
                largest_gap = max(gaps, key=lambda x: x[1])
                gap_center = largest_gap[0]
                escape_angle = math.atan2(0, gap_center - frame_width/2)
            elif "left" in safe_dirs:
                escape_angle = math.pi
            elif "right" in safe_dirs:
                escape_angle = 0
                
        return safe_dirs, blocked_dirs, escape_angle

    def _find_gaps_between_clusters(self, clusters: List[ObstacleCluster], frame_width: int) -> List[Tuple[float, float]]:
        if len(clusters) <= 1:
            return [(frame_width/2, frame_width)]
            
        sorted_clusters = sorted(clusters, key=lambda c: c.center_x)
        gaps = []
        
        left_edge = sorted_clusters[0].center_x - sorted_clusters[0].width/2
        if left_edge > self.nav_params.min_safe_gap:
            gaps.append((left_edge/2, left_edge))
            
        for i in range(len(sorted_clusters)-1):
            c1 = sorted_clusters[i]
            c2 = sorted_clusters[i+1]
            
            right_edge = c1.center_x + c1.width/2
            left_edge = c2.center_x - c2.width/2
            gap_width = left_edge - right_edge
            
            if gap_width > self.nav_params.min_safe_gap:
                gap_center = (right_edge + left_edge) / 2
                gaps.append((gap_center, gap_width))
        
        right_edge = sorted_clusters[-1].center_x + sorted_clusters[-1].width/2
        if frame_width - right_edge > self.nav_params.min_safe_gap:
            gaps.append(((right_edge + frame_width)/2, frame_width - right_edge))
        
        return gaps

    def _get_depth_at_point(self, x: int, y: int) -> Optional[float]:
        if self.depth_image is None:
            return None
            
        try:
            if (0 <= x < self.depth_image.shape[1]) and (0 <= y < self.depth_image.shape[0]):
                sample_size = 7
                depths = []
                
                for dx in range(-sample_size, sample_size + 1):
                    for dy in range(-sample_size, sample_size + 1):
                        sx, sy = x + dx, y + dy
                        if (0 <= sx < self.depth_image.shape[1]) and (0 <= sy < self.depth_image.shape[0]):
                            depth = float(self.depth_image[sy, sx])
                            if self.nav_params.min_obstacle_distance < depth < self.nav_params.max_obstacle_distance:
                                depths.append(depth)
                
                if depths:
                    return np.median(depths)
                    
        except Exception as e:
            self.get_logger().debug(f"Depth reading error: {e}")
        
        return None

    def _draw_enhanced_visualization(self, frame: np.ndarray, clusters: List[ObstacleCluster],
                                   safe_dirs: List[str], blocked_dirs: List[str]):
        for cluster in clusters:
            if cluster.danger_level == 3:
                color = (0, 0, 255)
            elif cluster.danger_level == 2:
                color = (0, 165, 255)
            else:
                color = (0, 255, 255)
                
            x1 = int(cluster.center_x - cluster.width/2)
            y1 = int(cluster.center_y - cluster.height/2)
            x2 = int(cluster.center_x + cluster.width/2)
            y2 = int(cluster.center_y + cluster.height/2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"D:{cluster.min_distance:.1f}m L:{cluster.danger_level} WX:{cluster.world_x:.1f} WY:{cluster.world_y:.1f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        h, w = frame.shape[:2]
        if "left" in safe_dirs:
            cv2.arrowedLine(frame, (w//4, h//2), (w//4 - 50, h//2), (0, 255, 0), 3)
        if "right" in safe_dirs:
            cv2.arrowedLine(frame, (3*w//4, h//2), (3*w//4 + 50, h//2), (0, 255, 0), 3)
        if "forward" in safe_dirs:
            cv2.arrowedLine(frame, (w//2, h//2), (w//2, h//2 - 50), (0, 255, 0), 3)
            
        status_text = f"Safe: {','.join(safe_dirs)} | Blocked: {','.join(blocked_dirs)}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Enhanced Smart Navigator", frame)
        cv2.waitKey(1)

    def timer_callback(self):
        try:
            if not self._safety_check():
                return
                
            self.publish_offboard_control_mode()

            if self.offboard_setpoint_counter == 10:
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.arm()

            self._handle_enhanced_state_machine()

            if self.offboard_setpoint_counter < 100:
                self.offboard_setpoint_counter += 1
                
        except Exception as e:
            self.get_logger().error(f"Timer callback error: {e}")
            self.state = DroneState.EMERGENCY

    def _safety_check(self) -> bool:
        if time.time() - self.last_position_update > self.position_timeout:
            self.get_logger().error("Position data timeout - entering emergency mode")
            self.state = DroneState.EMERGENCY
            return False
            
        if (self.obstacle_info.detected and 
            self.obstacle_info.min_distance < self.nav_params.emergency_threshold):
            self.emergency_stop = True
            self.get_logger().warn("Emergency stop activated - obstacle too close!")
            
        if self.current_position and self.last_position:
            movement = math.dist(self.current_position, self.last_position)
            if movement < self.movement_threshold:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
                
            if self.stuck_counter > 30:
                self.get_logger().warn("Drone appears stuck - attempting recovery")
                return self._handle_stuck_condition()
                
        self.last_position = self.current_position.copy() if self.current_position else None
        return True

    def _handle_stuck_condition(self) -> bool:
        if self.stuck_recovery_attempts >= self.max_recovery_attempts:
            self.get_logger().error("Max recovery attempts reached - entering emergency mode")
            self.state = DroneState.EMERGENCY
            return False
            
        self.stuck_recovery_attempts += 1
        
        if self.state == DroneState.AVOIDING:
            self.state = DroneState.RETREATING
            self.get_logger().info("Switching to retreat strategy")
        elif self.state == DroneState.RETREATING:
            self.state = DroneState.CIRCLING
            self.get_logger().info("Switching to circling strategy")
        else:
            if self.current_position:
                self.current_waypoint = [
                    self.current_position[0],
                    self.current_position[1],
                    max(self.nav_params.max_climb_altitude, 
                        self.current_position[2] - self.nav_params.vertical_climb_height)
                ]
                self.get_logger().info("Attempting vertical avoidance")
        
        self.stuck_counter = 0
        return True

    def _handle_enhanced_state_machine(self):
        if self.state == DroneState.INIT:
            self._handle_init_state()
        elif self.state == DroneState.TAKEOFF:
            self._handle_takeoff_state()
        elif self.state == DroneState.MOVE:
            self._handle_enhanced_move_state()
        elif self.state == DroneState.AVOIDING:
            self._handle_enhanced_avoiding_state()
        elif self.state == DroneState.RETREATING:
            self._handle_retreating_state()
        elif self.state == DroneState.CIRCLING:
            self._handle_circling_state()
        elif self.state == DroneState.HOLD:
            self._handle_hold_state()
        elif self.state == DroneState.LANDING:
            self._handle_landing_state()
        elif self.state == DroneState.EMERGENCY:
            self._handle_emergency_state()

    def _handle_init_state(self):
        self.publish_setpoint(*self.takeoff_point, self.yaw)
        
        if self.offboard_setpoint_counter == 10:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.arm()
            self.get_logger().info("Switched to Offboard mode and armed the drone")

        if self.offboard_setpoint_counter >= 20:
            self.state = DroneState.TAKEOFF
            self.get_logger().info("Entering takeoff phase...")

    def _handle_takeoff_state(self):
        self.publish_setpoint(*self.takeoff_point, self.yaw)
        
        if self.current_position:
            altitude_error = abs(self.current_position[2] - self.takeoff_point[2])
            if altitude_error < 0.3:
                self.state = DroneState.MOVE
                self.get_logger().info("Takeoff complete, moving to destination...")

    def _handle_enhanced_move_state(self):
        if self.emergency_stop:
            self.publish_setpoint(*self.current_position, self.yaw)
            self.get_logger().warn("Emergency stop active - holding position")
            return
            
        current_time = time.time()
        replan_needed = (current_time - self.last_path_plan_time >= self.path_plan_interval or 
                        not self.waypoint_queue)
        
        if replan_needed and self.current_position:
            # Adjust obstacle world coordinates based on drone's current position
            for cluster in self.obstacle_info.clusters:
                cluster.world_x += self.current_position[0]
                cluster.world_y += self.current_position[1]
            
            waypoints = self.path_planner.plan_avoidance_path(
                self.current_position, self.destination_point, self.obstacle_info)
            self.waypoint_queue = deque(waypoints)
            self.current_waypoint = self.waypoint_queue.popleft() if self.waypoint_queue else None
            self.last_path_plan_time = current_time
            self.get_logger().info(f"Replanned path: {waypoints}")
            
        if self.current_waypoint:
            self.publish_setpoint(*self.current_waypoint, self.yaw)
            
            if self.current_position:
                distance = math.dist(self.current_position, self.current_waypoint)
                if distance < 0.5:
                    if self.waypoint_queue:
                        self.current_waypoint = self.waypoint_queue.popleft()
                        self.get_logger().info(f"Moving to next waypoint: {self.current_waypoint}")
                    else:
                        self.current_waypoint = None
                        self.get_logger().info("Reached intermediate waypoint, replanning...")
                
                distance_to_destination = math.dist(self.current_position[:2], self.destination_point[:2])
                if distance_to_destination < self.nav_params.arrival_threshold:
                    self.state = DroneState.HOLD
                    self.get_logger().info("Arrived at destination")
        else:
            self.publish_setpoint(*self.destination_point, self.yaw)

    def _handle_enhanced_avoiding_state(self):
        current_time = time.time()
        
        if (self.avoidance_start_time and 
            current_time - self.avoidance_start_time > self.nav_params.max_avoidance_time):
            self.get_logger().warn("Avoidance timeout - switching to retreat")
            self.state = DroneState.RETREATING
            return
            
        if current_time - self.last_path_plan_time >= self.path_plan_interval:
            if self.current_position:
                for cluster in self.obstacle_info.clusters:
                    cluster.world_x += self.current_position[0]
                    cluster.world_y += self.current_position[1]
                
                waypoints = self.path_planner.plan_avoidance_path(
                    self.current_position, self.destination_point, self.obstacle_info)
                self.waypoint_queue = deque(waypoints)
                self.current_waypoint = self.waypoint_queue.popleft() if self.waypoint_queue else None
                self.last_path_plan_time = current_time
                self.get_logger().info(f"Replanned avoidance path: {waypoints}")
            
        if self.current_waypoint:
            self.publish_setpoint(*self.current_waypoint, self.yaw)
            
            if self.current_position:
                distance = math.dist(self.current_position, self.current_waypoint)
                if distance < 0.5:
                    if self.waypoint_queue:
                        self.current_waypoint = self.waypoint_queue.popleft()
                        self.get_logger().info(f"Moving to next waypoint: {self.current_waypoint}")
                    else:
                        self.state = DroneState.MOVE
                        self.current_waypoint = None
                        self.emergency_stop = False
                        self.stuck_recovery_attempts = 0
                        self.get_logger().info("Avoidance complete")
        else:
            self.state = DroneState.MOVE
            self.get_logger().info("No waypoints in avoidance, switching to MOVE")

    def _handle_retreating_state(self):
        if self.current_position:
            retreat_pos = [
                self.current_position[0] - self.nav_params.retreat_distance,
                self.current_position[1],
                self.current_position[2]
            ]
            self.publish_setpoint(*retreat_pos, self.yaw)
            
            distance = math.dist(self.current_position, retreat_pos)
            if distance < 0.5:
                self.state = DroneState.MOVE
                self.emergency_stop = False
                self.stuck_recovery_attempts = 0
                self.get_logger().info("Retreat complete - resuming navigation")

    def _handle_circling_state(self):
        if self.current_position:
            waypoints = self.path_planner.plan_avoidance_path(
                self.current_position, self.destination_point, self.obstacle_info)
            self.publish_setpoint(*waypoints[0], self.yaw)
            
            if not self.obstacle_info.detected or len(self.obstacle_info.safe_directions) > 2:
                self.state = DroneState.MOVE
                self.emergency_stop = False
                self.stuck_recovery_attempts = 0
                self.get_logger().info("Circling complete - path appears clear")

    def _handle_hold_state(self):
        self.publish_setpoint(*self.destination_point, self.yaw)
        self.get_logger().info("Holding position - preparing for landing...")
        time.sleep(2)
        self.state = DroneState.LANDING

    def _handle_landing_state(self):
        self.get_logger().info("Initiating landing sequence...")
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        
        if (self.current_position and 
            self.current_position[2] > self.nav_params.landing_altitude):
            self.get_logger().info("Landing complete - disarming...")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
            self.state = DroneState.DISARMED

    def _handle_emergency_state(self):
        self.get_logger().error("Emergency state - attempting emergency landing")
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

    def publish_setpoint(self, x: float, y: float, z: float, yaw: float):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.yaw = float(yaw)
        
        if self.current_position:
            dx = x - self.current_position[0]
            dy = y - self.current_position[1]
            dz = z - self.current_position[2]
            
            max_vel = self.nav_params.max_speed
            if self.obstacle_info.detected:
                scale = max(0.3, self.obstacle_info.min_distance / self.nav_params.critical_threshold)
                max_vel = max_vel * scale * self.nav_params.adaptive_speed_factor
            
            velocity_magnitude = math.sqrt(dx*dx + dy*dy + dz*dz)
            if velocity_magnitude > max_vel:
                scale = max_vel / velocity_magnitude
                msg.velocity = [dx * scale, dy * scale, dz * scale]
            else:
                msg.velocity = [dx, dy, dz]
        
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(msg)

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_pub.publish(msg)

    def publish_vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0):
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
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("Enhanced drone armed")

    def __del__(self):
        cv2.destroyAllWindows()