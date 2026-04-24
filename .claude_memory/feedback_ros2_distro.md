---
name: ROS2 Distribution
description: This project uses ROS2 Jazzy, not Humble
type: feedback
---

Always use ROS2 Jazzy (not Humble) for this project.

**Why:** The installed ROS2 distribution is Jazzy (`/opt/ros/jazzy`). Using Humble would source the wrong setup files and fail.

**How to apply:** When running `source /opt/ros/...`, always use `/opt/ros/jazzy/setup.bash`. When checking for package availability, use jazzy packages.
