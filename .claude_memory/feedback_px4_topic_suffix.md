---
name: PX4 topic suffix convention
description: PX4 state output topics use _v1 suffix on this setup
type: feedback
---

All PX4 vehicle state output topics use the `_v1` suffix:
- `/fmu/out/vehicle_local_position_v1`  (not vehicle_local_position)
- `/fmu/out/vehicle_status_v1`
- `/fmu/out/battery_status_v1`
- `/fmu/out/home_position_v1`

The non-v1 versions exist in topic list but have 0 publishers / no data.

**Why:** Newer PX4 ROS2 bridge API versioning scheme used in Jazzy/PX4 v1.15+.

**How to apply:** Always subscribe to `_v1` variants for vehicle state. Verify with `ros2 topic list | grep fmu/out` when writing new nodes.
