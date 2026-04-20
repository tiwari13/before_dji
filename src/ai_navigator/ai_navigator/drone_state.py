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
    # Advanced autonomy states
    RTH = "RTH"                          # Return to Home
    ACTIVETRACK = "ACTIVETRACK"          # Subject following
    TERRAIN_FOLLOW = "TERRAIN_FOLLOW"    # Terrain following mode
    GPS_DENIED = "GPS_DENIED"            # VIO-only navigation