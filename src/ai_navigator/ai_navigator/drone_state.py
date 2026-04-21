import math
from enum import Enum
class DroneState(Enum):
    """Enumeration for drone states"""
    INIT = "INIT"
    TAKEOFF = "TAKEOFF"
    MOVE = "MOVE"
    AVOIDING = "AVOIDING"
    RETREATING = "RETREATING"
    CIRCLING = "CIRCLING"
    HOLD = "HOLD"
    LANDING = "LANDING"
    DISARMED = "DISARMED"
    EMERGENCY = "EMERGENCY"
    # Advanced lifecycle states
    RTH = "RTH"              # Return to Home (safety / battery / geofence trigger)
    GPS_DENIED = "GPS_DENIED"  # VIO-only navigation — GPS unavailable
    # NOTE: ACTIVETRACK and TERRAIN_FOLLOW are FlightMode values, not lifecycle states.
    # Feature behavior is selected via self.flight_mode inside the MOVE state.