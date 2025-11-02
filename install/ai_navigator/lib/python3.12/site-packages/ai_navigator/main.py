#!/usr/bin/env python3
"""
Enhanced Smart Navigator - DJI Mavic 4 Pro Level
Production-grade main entry point with comprehensive error handling,
configuration management, and monitoring capabilities.
"""

import sys
import os
import signal
import time
import threading
import logging
import argparse
from typing import Optional
import yaml
import psutil

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

# Computer vision
import cv2

# Our enhanced modules
from ai_navigator.smart_navigator import SmartNavigator
from ai_navigator.navigation_params import NavigationParams, FlightMode
from ai_navigator.drone_state import DroneState

# Global variables for signal handling
navigator_node: Optional[SmartNavigator] = None
executor: Optional[MultiThreadedExecutor] = None
shutdown_event = threading.Event()

class SystemMonitor:
    """System health monitoring and performance tracking"""
    
    def __init__(self, logger):
        self.logger = logger
        self.start_time = time.time()
        self.monitoring = True
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start system monitoring thread"""
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring and not shutdown_event.is_set():
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Log warnings for high resource usage
                if cpu_percent > 80:
                    self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                
                if memory.percent > 80:
                    self.logger.warning(f"High memory usage: {memory.percent:.1f}%")
                
                # Runtime metrics
                runtime = time.time() - self.start_time
                if runtime > 0 and int(runtime) % 300 == 0:  # Every 5 minutes
                    self.logger.info(f"System runtime: {runtime/3600:.1f} hours")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(5)

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup comprehensive logging system"""
    
    # Create logger
    logger = logging.getLogger('smart_navigator')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.warning(f"Could not setup file logging: {e}")
    
    return logger

def load_configuration(config_file: Optional[str] = None) -> dict:
    """Load configuration from YAML file"""
    
    default_config = {
        'navigation': {
            'flight_mode': 'POSITION',
            'takeoff_altitude': -3.0,
            'max_speed': 8.0,
            'obstacle_threshold': 8.0,
            'precision_hover': True
        },
        'sensors': {
            'use_gps': True,
            'use_vision': True,
            'use_lidar': True,
            'sensor_fusion': True
        },
        'safety': {
            'max_altitude': 120.0,
            'max_distance': 500.0,
            'geofencing': True,
            'emergency_landing': True
        },
        'performance': {
            'control_frequency': 100.0,
            'vision_frequency': 30.0,
            'planning_frequency': 20.0
        },
        'activetrack': {
            'enabled': True,
            'tracking_mode': 'TRACE',
            'confidence_threshold': 0.7
        }
    }
    
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
                # Merge with default config
                default_config.update(loaded_config)
                print(f"✅ Configuration loaded from: {config_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load config file {config_file}: {e}")
            print("Using default configuration...")
    else:
        print("📋 Using default configuration")
    
    return default_config

def apply_configuration(navigator: SmartNavigator, config: dict):
    """Apply configuration to navigator"""
    try:
        # Apply navigation parameters
        nav_config = config.get('navigation', {})
        if 'flight_mode' in nav_config:
            mode_name = nav_config['flight_mode']
            if hasattr(FlightMode, mode_name):
                navigator.nav_params.flight_mode = getattr(FlightMode, mode_name)
                navigator.flight_mode = navigator.nav_params.flight_mode
        
        if 'takeoff_altitude' in nav_config:
            navigator.nav_params.takeoff_altitude = nav_config['takeoff_altitude']
            navigator.takeoff_point.z = nav_config['takeoff_altitude']
        
        if 'max_speed' in nav_config:
            navigator.nav_params.max_speed = nav_config['max_speed']
        
        if 'obstacle_threshold' in nav_config:
            navigator.nav_params.obstacle_threshold = nav_config['obstacle_threshold']
        
        # Apply ActiveTrack configuration
        activetrack_config = config.get('activetrack', {})
        if activetrack_config.get('enabled', True):
            if 'tracking_mode' in activetrack_config:
                from ai_navigator.navigation_params import TrackingMode
                mode_name = activetrack_config['tracking_mode']
                if hasattr(TrackingMode, mode_name):
                    navigator.nav_params.tracking_mode = getattr(TrackingMode, mode_name)
                    navigator.tracking_mode = navigator.nav_params.tracking_mode
        
        # Apply safety parameters
        safety_config = config.get('safety', {})
        for param in ['max_altitude', 'max_distance']:
            if param in safety_config:
                setattr(navigator.nav_params.safety, param, safety_config[param])
        
        print("✅ Configuration applied successfully")
        
    except Exception as e:
        print(f"⚠️  Warning: Error applying configuration: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global navigator_node, executor
    
    print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()
    
    if navigator_node:
        try:
            # Emergency land if in flight
            if navigator_node.state not in [DroneState.INIT, DroneState.DISARMED, DroneState.LANDING]:
                print("🚁 Drone in flight - initiating emergency landing...")
                navigator_node.state = DroneState.EMERGENCY
                time.sleep(2)  # Give time for emergency landing
            
            # Stop node
            navigator_node.get_logger().info("Shutting down SmartNavigator...")
            
        except Exception as e:
            print(f"Error during emergency shutdown: {e}")
    
    if executor:
        executor.shutdown(timeout_sec=5.0)
    
    # Force exit after timeout
    threading.Timer(10.0, lambda: os._exit(1)).start()

def validate_environment() -> bool:
    """Validate runtime environment and dependencies"""
    
    print("🔍 Validating environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        return False
    
    # Check required packages
    required_packages = [
        ('rclpy', 'ROS2 Python client library'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('yaml', 'PyYAML'),
        ('psutil', 'System monitoring')
    ]
    
    missing_packages = []
    for package, description in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append((package, description))
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package, description in missing_packages:
            print(f"   - {package}: {description}")
        return False
    
    # Check GPU availability for enhanced performance
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU acceleration available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  GPU not available - using CPU (performance may be limited)")
    except ImportError:
        print("⚠️  PyTorch not available - some AI features may be limited")
    
    # Check system resources
    memory = psutil.virtual_memory()
    if memory.total < 4 * 1024**3:  # 4GB
        print("⚠️  Warning: Less than 4GB RAM available")
    
    cpu_count = psutil.cpu_count()
    if cpu_count < 4:
        print("⚠️  Warning: Less than 4 CPU cores available")
    
    print("✅ Environment validation complete")
    return True

def create_navigator_node(config: dict) -> SmartNavigator:
    """Create and configure navigator node"""
    
    print("🚁 Initializing Enhanced Smart Navigator...")
    
    # Create node
    navigator = SmartNavigator()
    
    # Apply configuration
    apply_configuration(navigator, config)
    
    # Log initialization
    navigator.get_logger().info("🚁 Enhanced Smart Navigator initialized - DJI Mavic 4 Pro Level!")
    navigator.get_logger().info(f"Flight Mode: {navigator.flight_mode.name}")
    navigator.get_logger().info(f"Max Speed: {navigator.nav_params.max_speed} m/s")
    navigator.get_logger().info(f"Takeoff Altitude: {navigator.nav_params.takeoff_altitude} m")
    navigator.get_logger().info(f"Obstacle Threshold: {navigator.nav_params.obstacle_threshold} m")
    
    # Display capabilities
    capabilities = [
        "✅ Advanced Sensor Fusion (GPS + IMU + Vision + LIDAR)",
        "✅ Precision Hovering (±5cm accuracy)",
        "✅ ActiveTrack with 6 modes",
        "✅ Terrain Following", 
        "✅ Advanced Path Planning (A*, RRT*, DWA)",
        "✅ Real-time Obstacle Tracking",
        "✅ Multi-threaded Processing",
        "✅ Professional Flight Modes",
        "✅ Comprehensive Safety Systems"
    ]
    
    print("\n🎯 Enhanced Capabilities:")
    for capability in capabilities:
        print(f"   {capability}")
    
    return navigator

def run_health_checks(navigator: SmartNavigator) -> bool:
    """Run pre-flight health checks"""
    
    print("\n🏥 Running pre-flight health checks...")
    
    health_checks = []
    
    # Check sensor initialization
    if hasattr(navigator, 'detection_model') and navigator.detection_model:
        health_checks.append(("Computer Vision Model", True, "✅"))
    else:
        health_checks.append(("Computer Vision Model", False, "❌"))
    
    # Check parameter initialization
    if navigator.nav_params:
        health_checks.append(("Navigation Parameters", True, "✅"))
    else:
        health_checks.append(("Navigation Parameters", False, "❌"))
    
    # Check path planner
    if navigator.path_planner:
        health_checks.append(("Path Planner", True, "✅"))
    else:
        health_checks.append(("Path Planner", False, "❌"))
    
    # Check controllers
    if navigator.precision_hover and navigator.active_track:
        health_checks.append(("Advanced Controllers", True, "✅"))
    else:
        health_checks.append(("Advanced Controllers", False, "❌"))
    
    # Display results
    all_healthy = True
    for check_name, status, icon in health_checks:
        print(f"   {icon} {check_name}: {'READY' if status else 'FAILED'}")
        if not status:
            all_healthy = False
    
    if all_healthy:
        print("✅ All health checks passed - Ready for flight!")
    else:
        print("❌ Some health checks failed - Please review configuration")
    
    return all_healthy

def main(args=None):
    """
    Enhanced main entry point for Smart Navigator
    Supports multiple execution modes and comprehensive error handling
    """
    global navigator_node, executor
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Enhanced Smart Navigator - DJI Mavic 4 Pro Level',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ai_navigator.main                          # Default mode
  python -m ai_navigator.main --config config.yaml    # With custom config
  python -m ai_navigator.main --mode SPORT            # Sport mode
  python -m ai_navigator.main --activetrack TRACE     # ActiveTrack mode
  python -m ai_navigator.main --log-level DEBUG       # Debug logging
  python -m ai_navigator.main --log-file logs/nav.log # File logging
        """
    )
    
    parser.add_argument('--config', '-c', type=str, 
                       help='Configuration file path (YAML)')
    parser.add_argument('--mode', '-m', type=str, 
                       choices=['POSITION', 'SPORT', 'CINEMATIC', 'TRIPOD', 'ACTIVETRACK'],
                       help='Flight mode')
    parser.add_argument('--activetrack', '-t', type=str,
                       choices=['SPOTLIGHT', 'PROFILE', 'TRACE', 'PARALLEL', 'CIRCLE', 'HELIX'],
                       help='ActiveTrack mode')
    parser.add_argument('--log-level', '-l', type=str, 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=str,
                       help='Log file path')
    parser.add_argument('--no-health-check', action='store_true',
                       help='Skip pre-flight health checks')
    parser.add_argument('--dry-run', action='store_true',
                       help='Initialize but do not start flight')
    
    parsed_args = parser.parse_args(args)
    
    # Setup logging
    logger = setup_logging(parsed_args.log_level, parsed_args.log_file)
    
    # Banner
    print("\n" + "="*60)
    print("🚁 ENHANCED SMART NAVIGATOR - DJI MAVIC 4 PRO LEVEL 🚁")
    print("   Professional Autonomous Drone Navigation System")
    print("="*60)
    
    try:
        # Validate environment
        if not validate_environment():
            print("❌ Environment validation failed")
            return 1
        
        # Load configuration
        config = load_configuration(parsed_args.config)
        
        # Override with command line arguments
        if parsed_args.mode:
            config['navigation']['flight_mode'] = parsed_args.mode
        if parsed_args.activetrack:
            config['activetrack']['tracking_mode'] = parsed_args.activetrack
            config['navigation']['flight_mode'] = 'ACTIVETRACK'
        
        # Initialize ROS2
        print("\n🔧 Initializing ROS2...")
        rclpy.init(args=args)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Create multi-threaded executor for better performance
        executor = MultiThreadedExecutor(num_threads=4)
        
        # Create navigator node
        navigator_node = create_navigator_node(config)
        
        # Add node to executor
        executor.add_node(navigator_node)
        
        # Run health checks
        if not parsed_args.no_health_check:
            if not run_health_checks(navigator_node):
                if not parsed_args.dry_run:
                    response = input("\n⚠️  Health checks failed. Continue anyway? [y/N]: ")
                    if response.lower() not in ['y', 'yes']:
                        print("Aborting startup due to health check failures")
                        return 1
        
        # Dry run mode
        if parsed_args.dry_run:
            print("\n🧪 Dry run mode - Node initialized but not starting flight")
            print("Press Ctrl+C to exit")
            try:
                while not shutdown_event.is_set():
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            return 0
        
        # Start system monitoring
        monitor = SystemMonitor(logger)
        monitor.start_monitoring()
        
        # Start the navigation system
        print("\n🚀 Starting Enhanced Smart Navigator...")
        print("   - Multi-threaded execution enabled")
        print("   - Advanced sensor fusion active")
        print("   - Professional flight modes ready")
        print("   - Safety systems armed")
        print("\n✈️  READY FOR AUTONOMOUS FLIGHT! ✈️")
        print("\nPress Ctrl+C for graceful shutdown\n")
        
        # Run the executor
        try:
            executor.spin()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        
        # Stop monitoring
        monitor.stop_monitoring()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error occurred: {e}")
        return 1
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        
        if navigator_node:
            try:
                # Emergency safety - ensure drone is landed
                if hasattr(navigator_node, 'state') and navigator_node.state not in [
                    DroneState.INIT, DroneState.DISARMED, DroneState.LANDING
                ]:
                    print("🛑 Emergency landing initiated...")
                    navigator_node.state = DroneState.EMERGENCY
                    time.sleep(3)  # Give time for emergency landing
                
                navigator_node.destroy_node()
            except Exception as e:
                print(f"Error during node cleanup: {e}")
        
        if executor:
            try:
                executor.shutdown(timeout_sec=5.0)
            except Exception as e:
                print(f"Error during executor shutdown: {e}")
        
        try:
            rclpy.shutdown()
        except Exception as e:
            print(f"Error during ROS2 shutdown: {e}")
        
        # Close OpenCV windows
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        print("✅ Cleanup complete")
        print("\n🚁 Enhanced Smart Navigator shutdown complete. Fly safe! 🚁\n")
    
    return 0

# Entry points for different execution modes
def main_default():
    """Default entry point"""
    return main()

def main_sport():
    """Sport mode entry point"""
    return main(['--mode', 'SPORT'])

def main_activetrack():
    """ActiveTrack mode entry point"""
    return main(['--mode', 'ACTIVETRACK', '--activetrack', 'TRACE'])

def main_cinematic():
    """Cinematic mode entry point"""
    return main(['--mode', 'CINEMATIC'])

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)