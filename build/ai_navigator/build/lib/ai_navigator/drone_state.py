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