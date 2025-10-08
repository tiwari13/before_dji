import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/mcfb/ros2_ws/src/ai_navigator/install/ai_navigator'
