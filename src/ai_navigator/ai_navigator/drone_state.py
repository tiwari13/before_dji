"""
Enhanced Drone State Management System
Supports professional flight modes and advanced state transitions
"""

import math
import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

class DroneState(Enum):
    """Enhanced drone states for professional flight operations"""
    
    # Basic flight states
    INIT = "INIT"                           # System initialization
    PREFLIGHT = "PREFLIGHT"                 # Pre-flight checks
    TAKEOFF = "TAKEOFF"                     # Taking off
    MOVE = "MOVE"                           # Normal navigation
    HOLD = "HOLD"                           # Position hold/hover
    LANDING = "LANDING"                     # Landing sequence
    LANDED = "LANDED"                       # Landed and disarmed
    DISARMED = "DISARMED"                   # System disarmed
    
    # Advanced navigation states
    AVOIDING = "AVOIDING"                   # Obstacle avoidance
    RETREATING = "RETREATING"               # Emergency retreat
    CIRCLING = "CIRCLING"                   # Circling maneuver
    CLIMBING = "CLIMBING"                   # Vertical climb
    DESCENDING = "DESCENDING"               # Controlled descent
    
    # Professional mission states
    ACTIVETRACK = "ACTIVETRACK"             # Subject tracking
    TERRAIN_FOLLOW = "TERRAIN_FOLLOW"       # Terrain following
    WAYPOINT_NAV = "WAYPOINT_NAV"          # Waypoint navigation
    ORBIT = "ORBIT"                         # Orbit around point
    RTH = "RTH"                            # Return to home
    PRECISION_LAND = "PRECISION_LAND"       # Precision landing
    
    # Safety and emergency states
    EMERGENCY = "EMERGENCY"                 # Emergency state
    EMERGENCY_LAND = "EMERGENCY_LAND"       # Emergency landing
    FAILSAFE = "FAILSAFE"                  # Failsafe mode
    GPS_LOST = "GPS_LOST"                  # GPS signal lost
    LOW_BATTERY = "LOW_BATTERY"            # Low battery warning
    CRITICAL_BATTERY = "CRITICAL_BATTERY"   # Critical battery
    
    # Maintenance and calibration states
    CALIBRATING = "CALIBRATING"             # Sensor calibration
    SYSTEM_CHECK = "SYSTEM_CHECK"           # System diagnostics
    FIRMWARE_UPDATE = "FIRMWARE_UPDATE"     # Firmware updating
    MAINTENANCE = "MAINTENANCE"             # Maintenance mode

class StateTransitionResult(Enum):
    """Results of state transition attempts"""
    SUCCESS = "SUCCESS"                     # Transition successful
    DENIED = "DENIED"                       # Transition not allowed
    PENDING = "PENDING"                     # Transition in progress
    ERROR = "ERROR"                         # Error during transition

class StatePriority(Enum):
    """Priority levels for state transitions"""
    LOW = 1                                 # Normal operations
    MEDIUM = 2                              # Mission critical
    HIGH = 3                                # Safety critical
    EMERGENCY = 4                           # Emergency override
    SYSTEM = 5                              # System level override

@dataclass
class StateTransition:
    """Represents a state transition with metadata"""
    from_state: DroneState
    to_state: DroneState
    timestamp: float
    priority: StatePriority
    reason: str
    conditions_met: bool = True
    duration: Optional[float] = None        # Expected duration
    
class DroneStateManager:
    """Advanced state management system for professional drone operations"""
    
    def __init__(self):
        self.current_state = DroneState.INIT
        self.previous_state = DroneState.INIT
        self.state_entry_time = time.time()
        self.state_duration = 0.0
        
        # State history for analysis
        self.state_history: List[StateTransition] = []
        self.max_history_length = 100
        
        # State transition rules
        self.transition_rules = self._initialize_transition_rules()
        
        # State timers and timeouts
        self.state_timeouts = self._initialize_state_timeouts()
        self.timeout_warnings_sent = set()
        
        # Performance metrics
        self.transition_count = 0
        self.error_count = 0
        self.emergency_count = 0
        
    def _initialize_transition_rules(self) -> Dict[DroneState, Dict[DroneState, StatePriority]]:
        """Initialize state transition rules matrix"""
        rules = {}
        
        # Define allowed transitions for each state
        # Format: current_state: {allowed_next_state: minimum_priority}
        
        rules[DroneState.INIT] = {
            DroneState.PREFLIGHT: StatePriority.LOW,
            DroneState.CALIBRATING: StatePriority.LOW,
            DroneState.SYSTEM_CHECK: StatePriority.LOW,
            DroneState.EMERGENCY: StatePriority.EMERGENCY,
            DroneState.MAINTENANCE: StatePriority.MEDIUM
        }
        
        rules[DroneState.PREFLIGHT] = {
            DroneState.TAKEOFF: StatePriority.LOW,
            DroneState.CALIBRATING: StatePriority.MEDIUM,
            DroneState.INIT: StatePriority.LOW,
            DroneState.EMERGENCY: StatePriority.EMERGENCY
        }
        
        rules[DroneState.TAKEOFF] = {
            DroneState.MOVE: StatePriority.LOW,
            DroneState.HOLD: StatePriority.LOW,
            DroneState.EMERGENCY_LAND: StatePriority.HIGH,
            DroneState.EMERGENCY: StatePriority.EMERGENCY
        }
        
        rules[DroneState.MOVE] = {
            DroneState.HOLD: StatePriority.LOW,
            DroneState.AVOIDING: StatePriority.MEDIUM,
            DroneState.ACTIVETRACK: StatePriority.LOW,
            DroneState.TERRAIN_FOLLOW: StatePriority.LOW,
            DroneState.WAYPOINT_NAV: StatePriority.LOW,
            DroneState.ORBIT: StatePriority.LOW,
            DroneState.RTH: StatePriority.MEDIUM,
            DroneState.LANDING: StatePriority.LOW,
            DroneState.EMERGENCY: StatePriority.EMERGENCY,
            DroneState.GPS_LOST: StatePriority.HIGH,
            DroneState.LOW_BATTERY: StatePriority.HIGH
        }
        
        rules[DroneState.HOLD] = {
            DroneState.MOVE: StatePriority.LOW,
            DroneState.ACTIVETRACK: StatePriority.LOW,
            DroneState.TERRAIN_FOLLOW: StatePriority.LOW,
            DroneState.WAYPOINT_NAV: StatePriority.LOW,
            DroneState.ORBIT: StatePriority.LOW,
            DroneState.RTH: StatePriority.MEDIUM,
            DroneState.LANDING: StatePriority.LOW,
            DroneState.PRECISION_LAND: StatePriority.LOW,
            DroneState.EMERGENCY: StatePriority.EMERGENCY,
            DroneState.GPS_LOST: StatePriority.HIGH,
            DroneState.LOW_BATTERY: StatePriority.HIGH
        }
        
        rules[DroneState.AVOIDING] = {
            DroneState.MOVE: StatePriority.LOW,
            DroneState.HOLD: StatePriority.LOW,
            DroneState.RETREATING: StatePriority.MEDIUM,
            DroneState.CIRCLING: StatePriority.MEDIUM,
            DroneState.CLIMBING: StatePriority.MEDIUM,
            DroneState.RTH: StatePriority.HIGH,
            DroneState.EMERGENCY: StatePriority.EMERGENCY
        }
        
        rules[DroneState.ACTIVETRACK] = {
            DroneState.MOVE: StatePriority.LOW,
            DroneState.HOLD: StatePriority.LOW,
            DroneState.AVOIDING: StatePriority.MEDIUM,
            DroneState.RTH: StatePriority.MEDIUM,
            DroneState.EMERGENCY: StatePriority.EMERGENCY,
            DroneState.LOW_BATTERY: StatePriority.HIGH
        }
        
        rules[DroneState.TERRAIN_FOLLOW] = {
            DroneState.MOVE: StatePriority.LOW,
            DroneState.HOLD: StatePriority.LOW,
            DroneState.AVOIDING: StatePriority.MEDIUM,
            DroneState.CLIMBING: StatePriority.HIGH,  # Terrain emergency
            DroneState.RTH: StatePriority.MEDIUM,
            DroneState.EMERGENCY: StatePriority.EMERGENCY
        }
        
        rules[DroneState.RTH] = {
            DroneState.LANDING: StatePriority.LOW,
            DroneState.PRECISION_LAND: StatePriority.LOW,
            DroneState.HOLD: StatePriority.MEDIUM,
            DroneState.AVOIDING: StatePriority.MEDIUM,
            DroneState.EMERGENCY: StatePriority.EMERGENCY,
            DroneState.CRITICAL_BATTERY: StatePriority.HIGH
        }
        
        rules[DroneState.LANDING] = {
            DroneState.LANDED: StatePriority.LOW,
            DroneState.HOLD: StatePriority.MEDIUM,  # Abort landing
            DroneState.EMERGENCY_LAND: StatePriority.HIGH,
            DroneState.EMERGENCY: StatePriority.EMERGENCY
        }
        
        rules[DroneState.PRECISION_LAND] = {
            DroneState.LANDED: StatePriority.LOW,
            DroneState.LANDING: StatePriority.MEDIUM,  # Fallback
            DroneState.HOLD: StatePriority.MEDIUM,    # Abort
            DroneState.EMERGENCY: StatePriority.EMERGENCY
        }
        
        # Emergency states - can transition to limited states
        rules[DroneState.EMERGENCY] = {
            DroneState.EMERGENCY_LAND: StatePriority.LOW,
            DroneState.HOLD: StatePriority.MEDIUM,
            DroneState.RTH: StatePriority.MEDIUM
        }
        
        rules[DroneState.GPS_LOST] = {
            DroneState.HOLD: StatePriority.LOW,       # Vision positioning
            DroneState.EMERGENCY_LAND: StatePriority.MEDIUM,
            DroneState.MOVE: StatePriority.MEDIUM,    # GPS recovered
            DroneState.EMERGENCY: StatePriority.HIGH
        }
        
        # Battery states
        rules[DroneState.LOW_BATTERY] = {
            DroneState.RTH: StatePriority.LOW,
            DroneState.LANDING: StatePriority.MEDIUM,
            DroneState.CRITICAL_BATTERY: StatePriority.HIGH,
            DroneState.EMERGENCY: StatePriority.EMERGENCY
        }
        
        rules[DroneState.CRITICAL_BATTERY] = {
            DroneState.EMERGENCY_LAND: StatePriority.LOW,
            DroneState.EMERGENCY: StatePriority.HIGH
        }
        
        return rules
    
    def _initialize_state_timeouts(self) -> Dict[DroneState, float]:
        """Initialize timeout values for each state (in seconds)"""
        return {
            DroneState.INIT: 30.0,              # Max initialization time
            DroneState.PREFLIGHT: 60.0,         # Pre-flight check time
            DroneState.TAKEOFF: 15.0,           # Max takeoff time
            DroneState.LANDING: 30.0,           # Max landing time
            DroneState.PRECISION_LAND: 45.0,    # Max precision landing
            DroneState.AVOIDING: 30.0,          # Max avoidance time
            DroneState.RETREATING: 10.0,        # Max retreat time
            DroneState.CIRCLING: 60.0,          # Max circling time
            DroneState.RTH: 300.0,              # Max RTH time (5 min)
            DroneState.CALIBRATING: 120.0,      # Max calibration time
            DroneState.SYSTEM_CHECK: 60.0,      # Max system check time
            DroneState.EMERGENCY: 300.0,        # Emergency timeout
            DroneState.GPS_LOST: 60.0,          # GPS recovery timeout
            # No timeout for: MOVE, HOLD, ACTIVETRACK, TERRAIN_FOLLOW (mission states)
        }
    
    def transition_to(self, new_state: DroneState, priority: StatePriority = StatePriority.LOW,
                     reason: str = "Manual transition") -> StateTransitionResult:
        """Attempt to transition to a new state"""
        
        # Check if transition is allowed
        if not self._is_transition_allowed(self.current_state, new_state, priority):
            return StateTransitionResult.DENIED
        
        # Record transition
        transition = StateTransition(
            from_state=self.current_state,
            to_state=new_state,
            timestamp=time.time(),
            priority=priority,
            reason=reason
        )
        
        # Update state
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_duration = time.time() - self.state_entry_time
        self.state_entry_time = time.time()
        
        # Add to history
        self.state_history.append(transition)
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)
        
        # Update metrics
        self.transition_count += 1
        if new_state in [DroneState.EMERGENCY, DroneState.EMERGENCY_LAND, DroneState.FAILSAFE]:
            self.emergency_count += 1
        
        # Clear timeout warnings for new state
        self.timeout_warnings_sent.discard(new_state)
        
        return StateTransitionResult.SUCCESS
    
    def _is_transition_allowed(self, from_state: DroneState, to_state: DroneState,
                              priority: StatePriority) -> bool:
        """Check if state transition is allowed"""
        
        # Emergency override - always allowed
        if priority == StatePriority.EMERGENCY:
            return True
        
        # Check transition rules
        if from_state not in self.transition_rules:
            return False
        
        allowed_transitions = self.transition_rules[from_state]
        if to_state not in allowed_transitions:
            return False
        
        # Check priority requirements
        required_priority = allowed_transitions[to_state]
        return priority.value >= required_priority.value
    
    def get_current_state(self) -> DroneState:
        """Get current drone state"""
        return self.current_state
    
    def get_state_duration(self) -> float:
        """Get time spent in current state"""
        return time.time() - self.state_entry_time
    
    def is_in_flight(self) -> bool:
        """Check if drone is currently in flight"""
        flight_states = {
            DroneState.TAKEOFF, DroneState.MOVE, DroneState.HOLD,
            DroneState.AVOIDING, DroneState.RETREATING, DroneState.CIRCLING,
            DroneState.CLIMBING, DroneState.DESCENDING, DroneState.ACTIVETRACK,
            DroneState.TERRAIN_FOLLOW, DroneState.WAYPOINT_NAV, DroneState.ORBIT,
            DroneState.RTH, DroneState.LANDING, DroneState.PRECISION_LAND,
            DroneState.GPS_LOST
        }
        return self.current_state in flight_states
    
    def is_emergency_state(self) -> bool:
        """Check if drone is in emergency state"""
        emergency_states = {
            DroneState.EMERGENCY, DroneState.EMERGENCY_LAND, DroneState.FAILSAFE,
            DroneState.GPS_LOST, DroneState.CRITICAL_BATTERY
        }
        return self.current_state in emergency_states
    
    def is_autonomous_mode(self) -> bool:
        """Check if drone is in autonomous navigation mode"""
        autonomous_states = {
            DroneState.MOVE, DroneState.AVOIDING, DroneState.ACTIVETRACK,
            DroneState.TERRAIN_FOLLOW, DroneState.WAYPOINT_NAV, DroneState.ORBIT,
            DroneState.RTH
        }
        return self.current_state in autonomous_states
    
    def is_mission_critical_state(self) -> bool:
        """Check if current state is mission critical"""
        critical_states = {
            DroneState.TAKEOFF, DroneState.LANDING, DroneState.PRECISION_LAND,
            DroneState.EMERGENCY, DroneState.EMERGENCY_LAND, DroneState.RTH,
            DroneState.CRITICAL_BATTERY, DroneState.GPS_LOST
        }
        return self.current_state in critical_states
    
    def check_timeout(self) -> Optional[str]:
        """Check if current state has timed out"""
        if self.current_state not in self.state_timeouts:
            return None
        
        timeout_duration = self.state_timeouts[self.current_state]
        current_duration = self.get_state_duration()
        
        if current_duration > timeout_duration:
            return f"State {self.current_state.name} timed out after {current_duration:.1f}s (limit: {timeout_duration:.1f}s)"
        
        # Warning at 80% of timeout
        warning_threshold = timeout_duration * 0.8
        if (current_duration > warning_threshold and 
            self.current_state not in self.timeout_warnings_sent):
            self.timeout_warnings_sent.add(self.current_state)
            return f"State {self.current_state.name} approaching timeout ({current_duration:.1f}s / {timeout_duration:.1f}s)"
        
        return None
    
    def get_allowed_transitions(self) -> List[DroneState]:
        """Get list of states that can be transitioned to from current state"""
        if self.current_state not in self.transition_rules:
            return []
        
        return list(self.transition_rules[self.current_state].keys())
    
    def get_state_history(self, limit: int = 10) -> List[StateTransition]:
        """Get recent state transition history"""
        return self.state_history[-limit:] if limit else self.state_history
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """Get state management statistics"""
        if not self.state_history:
            return {}
        
        # Calculate state durations
        state_durations = {}
        for i, transition in enumerate(self.state_history):
            if i < len(self.state_history) - 1:
                duration = self.state_history[i + 1].timestamp - transition.timestamp
                state = transition.to_state.name
                if state not in state_durations:
                    state_durations[state] = []
                state_durations[state].append(duration)
        
        # Calculate averages
        avg_durations = {}
        for state, durations in state_durations.items():
            avg_durations[state] = sum(durations) / len(durations)
        
        return {
            'total_transitions': self.transition_count,
            'emergency_count': self.emergency_count,
            'current_state': self.current_state.name,
            'current_state_duration': self.get_state_duration(),
            'average_state_durations': avg_durations,
            'most_common_states': self._get_most_common_states(),
            'error_rate': self.error_count / max(1, self.transition_count)
        }
    
    def _get_most_common_states(self) -> List[Tuple[str, int]]:
        """Get most commonly used states"""
        state_counts = {}
        for transition in self.state_history:
            state = transition.to_state.name
            state_counts[state] = state_counts.get(state, 0) + 1
        
        return sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def reset(self):
        """Reset state manager to initial state"""
        self.current_state = DroneState.INIT
        self.previous_state = DroneState.INIT
        self.state_entry_time = time.time()
        self.state_duration = 0.0
        self.timeout_warnings_sent.clear()
    
    def force_emergency_state(self, reason: str = "Manual emergency"):
        """Force transition to emergency state (bypasses all rules)"""
        self.transition_to(DroneState.EMERGENCY, StatePriority.EMERGENCY, reason)
    
    def __str__(self) -> str:
        """String representation of current state"""
        duration = self.get_state_duration()
        return f"DroneState: {self.current_state.name} (duration: {duration:.1f}s)"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return (f"DroneStateManager(current={self.current_state.name}, "
                f"previous={self.previous_state.name}, "
                f"duration={self.get_state_duration():.1f}s, "
                f"transitions={self.transition_count})")

# Convenience functions for backward compatibility
def get_drone_state_manager() -> DroneStateManager:
    """Get a global drone state manager instance"""
    if not hasattr(get_drone_state_manager, '_instance'):
        get_drone_state_manager._instance = DroneStateManager()
    return get_drone_state_manager._instance