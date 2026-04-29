#!/usr/bin/env python3
"""
Smart Navigator — main entry point.

Handles ROS 2 argument separation, configuration loading, system monitoring,
and clean lifecycle management.
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

import rclpy
from rclpy.utilities import remove_ros_args
from rclpy.executors import MultiThreadedExecutor

import cv2

from ai_navigator.smart_navigator import SmartNavigator
from ai_navigator.navigation_params import FlightMode
from ai_navigator.drone_state import DroneState

# ── Globals touched by signal handler ─────────────────────────────────────────
_executor: Optional[MultiThreadedExecutor] = None
_shutdown_event = threading.Event()


# ═══════════════════════════════════════════════════════════════════════════════
# System monitor
# ═══════════════════════════════════════════════════════════════════════════════

class SystemMonitor:
    """
    Daemon thread that logs warnings when CPU or memory usage is high.

    Uses psutil.cpu_percent(interval=None) — non-blocking; measures since last
    call rather than sleeping for 1 s inside the psutil call itself.
    """

    POLL_INTERVAL = 30   # seconds between checks
    RUNTIME_LOG_INTERVAL = 300  # seconds between runtime log entries

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._start  = time.monotonic()
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Prime the non-blocking measurement
        psutil.cpu_percent(interval=None)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._logger.info("System monitor started")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._logger.info("System monitor stopped")

    def _loop(self):
        last_runtime_log = 0.0
        while not self._stop.is_set() and not _shutdown_event.is_set():
            try:
                cpu   = psutil.cpu_percent(interval=None)
                mem   = psutil.virtual_memory()
                runtime = time.monotonic() - self._start

                if cpu > 80:
                    self._logger.warning(f"High CPU usage: {cpu:.1f}%")
                if mem.percent > 80:
                    self._logger.warning(f"High memory usage: {mem.percent:.1f}%")

                if runtime - last_runtime_log >= self.RUNTIME_LOG_INTERVAL:
                    self._logger.info(f"Runtime: {runtime/3600:.2f} h  "
                                      f"CPU: {cpu:.1f}%  RAM: {mem.percent:.1f}%")
                    last_runtime_log = runtime

            except Exception as e:
                self._logger.error(f"Monitor error: {e}")

            self._stop.wait(timeout=self.POLL_INTERVAL)


# ═══════════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(log_level: str = "INFO",
                  log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger('smart_navigator')
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    fmt_simple   = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fmt_detailed = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_simple)
    logger.addHandler(ch)

    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt_detailed)
            logger.addHandler(fh)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.warning(f"Could not set up file logging: {e}")

    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_CONFIG: dict = {
    'navigation': {
        'flight_mode':        'POSITION',
        'takeoff_altitude':   -3.0,
        'max_speed':           8.0,
        'obstacle_threshold':  8.0,
        'precision_hover':     True,
    },
    'sensors': {
        'use_gps':      True,
        'use_vision':   True,
        'use_lidar':    True,
        'sensor_fusion': True,
    },
    'safety': {
        'max_altitude':      120.0,
        'max_distance':      500.0,
        'geofencing':         True,
        'emergency_landing':  True,
    },
    'performance': {
        'control_frequency': 100.0,
        'vision_frequency':   30.0,
        'planning_frequency': 20.0,
    },
    'activetrack': {
        'enabled':               True,
        'tracking_mode':        'TRACE',
        'confidence_threshold':  0.7,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_configuration(config_file: Optional[str] = None) -> dict:
    """
    Load YAML config and deep-merge over defaults.

    A partial config file (e.g. only the 'navigation' section) merges
    correctly without wiping unrelated default subkeys.
    """
    import copy
    config = copy.deepcopy(_DEFAULT_CONFIG)

    if config_file:
        if not os.path.exists(config_file):
            print(f"Warning: config file not found: {config_file} — using defaults")
            return config
        try:
            with open(config_file, 'r') as f:
                loaded = yaml.safe_load(f) or {}
            config = _deep_merge(config, loaded)
            print(f"Configuration loaded from: {config_file}")
        except Exception as e:
            print(f"Warning: could not load config file {config_file}: {e} — using defaults")
    else:
        print("Using default configuration")

    return config


def apply_configuration(navigator: SmartNavigator, config: dict):
    """Apply loaded config to a live navigator node."""
    nav_cfg = config.get('navigation', {})

    if 'flight_mode' in nav_cfg:
        mode_name = nav_cfg['flight_mode']
        if hasattr(FlightMode, mode_name):
            navigator.nav_params.flight_mode = getattr(FlightMode, mode_name)
            navigator.flight_mode = navigator.nav_params.flight_mode
        else:
            print(f"Warning: unknown flight_mode '{mode_name}' — ignored")

    if 'takeoff_altitude' in nav_cfg:
        navigator.nav_params.takeoff_altitude = nav_cfg['takeoff_altitude']
        navigator.takeoff_point.z = nav_cfg['takeoff_altitude']

    if 'max_speed' in nav_cfg:
        navigator.nav_params.max_speed = nav_cfg['max_speed']

    if 'obstacle_threshold' in nav_cfg:
        navigator.nav_params.obstacle_threshold = nav_cfg['obstacle_threshold']

    activetrack_cfg = config.get('activetrack', {})
    if activetrack_cfg.get('enabled', True) and 'tracking_mode' in activetrack_cfg:
        from ai_navigator.navigation_params import TrackingMode
        mode_name = activetrack_cfg['tracking_mode']
        if hasattr(TrackingMode, mode_name):
            navigator.nav_params.tracking_mode = getattr(TrackingMode, mode_name)
            navigator.tracking_mode = navigator.nav_params.tracking_mode
        else:
            print(f"Warning: unknown tracking_mode '{mode_name}' — ignored")

    safety_cfg = config.get('safety', {})
    for param in ['max_altitude', 'max_distance']:
        if param in safety_cfg:
            setattr(navigator.nav_params.safety, param, safety_cfg[param])


# ═══════════════════════════════════════════════════════════════════════════════
# Signal handling
# ═══════════════════════════════════════════════════════════════════════════════

def _signal_handler(signum, frame):
    """
    Set the shutdown event and cancel the executor.

    Does NOT sleep, does NOT call rclpy.shutdown() — the main finally block
    owns all cleanup.  Calling executor.cancel() causes spin() to return so
    the normal cleanup path runs.
    """
    print(f"\nReceived signal {signum} — shutting down")
    _shutdown_event.set()
    if _executor is not None:
        _executor.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# Environment validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_environment() -> bool:
    print("Validating environment...")

    if sys.version_info < (3, 8):
        print("Python 3.8 or higher required")
        return False

    required = [
        ('rclpy', 'ROS2 Python client library'),
        ('cv2',   'OpenCV'),
        ('numpy', 'NumPy'),
        ('yaml',  'PyYAML'),
        ('psutil','psutil'),
    ]
    missing = [pkg for pkg, _ in required if not _importable(pkg)]
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        return False

    mem = psutil.virtual_memory()
    if mem.total < 4 * 1024**3:
        print(f"Warning: only {mem.total // 1024**3} GB RAM — 4 GB recommended")

    print("Environment OK")
    return True


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Health checks
# ═══════════════════════════════════════════════════════════════════════════════

def run_health_checks(navigator: SmartNavigator) -> bool:
    """
    Check that navigator subsystems are instantiated.

    These are existence checks, not operational readiness checks — they will
    catch missing initialisation but not runtime sensor faults.
    """
    print("\nPre-flight checks:")

    checks = [
        ("nav_params",    bool(navigator.nav_params),
         "Navigation parameters"),
        ("path_planner",  bool(navigator.path_planner),
         "Path planner"),
        ("detection_model", bool(getattr(navigator, 'detection_model', None)),
         "Detection model"),
        ("controllers",   bool(getattr(navigator, 'precision_hover', None) and
                               getattr(navigator, 'active_track',    None)),
         "Controllers (precision_hover, active_track)"),
    ]

    all_ok = True
    for key, status, label in checks:
        mark = "OK  " if status else "FAIL"
        print(f"  [{mark}] {label}")
        if not status:
            all_ok = False

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    """
    Entry point.

    ROS 2 argument separation
    -------------------------
    ros2 run passes the full sys.argv (including --ros-args ...) to main().
    rclpy.utilities.remove_ros_args() strips the --ros-args block so argparse
    only sees application-level flags.  rclpy.init() then gets the original
    argv (or the caller-supplied list) so it can parse the ROS args itself.
    """
    global _executor

    # Separate ROS args from application args before argparse sees them
    raw_args   = sys.argv[1:] if args is None else list(args)
    app_args   = remove_ros_args(raw_args)   # strips --ros-args and everything after

    parser = argparse.ArgumentParser(
        description='Smart Navigator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ros2 run ai_navigator autonomous_navigation
  ros2 run ai_navigator autonomous_navigation -- --config config.yaml --mode SPORT
  ros2 run ai_navigator autonomous_navigation -- --dry-run
        """,
    )
    parser.add_argument('--config', '-c', type=str,
                        help='Configuration file path (YAML)')
    parser.add_argument('--mode', '-m', type=str,
                        choices=['POSITION', 'SPORT', 'CINEMATIC', 'TRIPOD', 'ACTIVETRACK'],
                        help='Flight mode')
    parser.add_argument('--activetrack', '-t', type=str,
                        choices=['SPOTLIGHT', 'PROFILE', 'TRACE', 'PARALLEL', 'CIRCLE', 'HELIX'],
                        help='ActiveTrack tracking mode')
    parser.add_argument('--log-level', '-l', type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=str,
                        help='Log file path')
    parser.add_argument('--no-health-check', action='store_true',
                        help='Skip pre-flight health checks')
    parser.add_argument('--force', action='store_true',
                        help='Continue even if health checks fail')
    parser.add_argument('--dry-run', action='store_true',
                        help='Initialise node but do not spin executor')

    parsed = parser.parse_args(app_args)
    logger = setup_logging(parsed.log_level, parsed.log_file)

    print("=" * 55)
    print("  Smart Navigator")
    print("=" * 55)

    monitor: Optional[SystemMonitor] = None
    navigator_node: Optional[SmartNavigator] = None
    ros_initialised = False

    try:
        if not validate_environment():
            return 1

        config = load_configuration(parsed.config)

        if parsed.mode:
            config['navigation']['flight_mode'] = parsed.mode
        if parsed.activetrack:
            config['activetrack']['tracking_mode'] = parsed.activetrack
            config['navigation']['flight_mode'] = 'ACTIVETRACK'

        # rclpy.init receives the original argv so it can parse --ros-args
        rclpy.init(args=raw_args)
        ros_initialised = True

        signal.signal(signal.SIGINT,  _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        _executor = MultiThreadedExecutor(num_threads=4)

        print("Initialising SmartNavigator node...")
        navigator_node = SmartNavigator()
        apply_configuration(navigator_node, config)
        _executor.add_node(navigator_node)

        navigator_node.get_logger().info(
            f"Navigator ready — mode={navigator_node.flight_mode.name}  "
            f"max_speed={navigator_node.nav_params.max_speed} m/s  "
            f"takeoff_alt={navigator_node.nav_params.takeoff_altitude} m"
        )

        if not parsed.no_health_check:
            healthy = run_health_checks(navigator_node)
            if not healthy:
                if parsed.force:
                    print("Health checks failed — continuing because --force was given")
                else:
                    print("Health checks failed — aborting (use --force to override)")
                    return 1

        if parsed.dry_run:
            print("Dry run — node initialised, not spinning. Ctrl+C to exit.")
            _shutdown_event.wait()
            return 0

        monitor = SystemMonitor(logger)
        monitor.start()

        print("Navigator running. Ctrl+C to stop.\n")
        _executor.spin()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    finally:
        print("\nShutting down...")

        if monitor is not None:
            monitor.stop()

        if navigator_node is not None:
            try:
                if (hasattr(navigator_node, 'state') and
                        navigator_node.state not in
                        [DroneState.INIT, DroneState.DISARMED, DroneState.LANDING]):
                    print("Setting EMERGENCY state before shutdown")
                    navigator_node.state = DroneState.EMERGENCY
                navigator_node.destroy_node()
            except Exception as e:
                print(f"Node cleanup error: {e}")

        if _executor is not None:
            try:
                _executor.shutdown()
            except Exception as e:
                print(f"Executor shutdown error: {e}")

        if ros_initialised and rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception as e:
                print(f"ROS2 shutdown error: {e}")

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("Shutdown complete")

    return 0


# ── Mode-specific entry points ─────────────────────────────────────────────────

def main_default():
    return main()

def main_sport():
    return main(['--mode', 'SPORT'])

def main_activetrack():
    return main(['--mode', 'ACTIVETRACK', '--activetrack', 'TRACE'])

def main_cinematic():
    return main(['--mode', 'CINEMATIC'])


if __name__ == '__main__':
    sys.exit(main())
