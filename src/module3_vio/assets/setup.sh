#!/bin/bash
# Deploy PX4 sim assets for x500_oakd_pro_w_front
# Run once on any machine after cloning the repo.
# Usage: bash src/module3_vio/assets/setup.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR="$HOME/PX4-Autopilot"
ROS2_WS="$HOME/ros2_ws"

echo "=== Deploying x500_oakd_pro_w_front assets ==="

# ── 1. Gazebo model ──────────────────────────────────────────────────────────
MODEL_DST="$PX4_DIR/Tools/simulation/gz/models/x500_oakd_pro_w_front"
mkdir -p "$MODEL_DST"
cp "$SCRIPT_DIR/x500_oakd_pro_w_front_model.sdf" "$MODEL_DST/model.sdf"
echo "  [OK] Gazebo model → $MODEL_DST/model.sdf"

# ── 2. Airframe file ─────────────────────────────────────────────────────────
AIRFRAME_DST="$PX4_DIR/ROMFS/px4fmu_common/init.d-posix/airframes/22112_gz_x500_oakd_pro_w_front"
cp "$SCRIPT_DIR/22112_gz_x500_oakd_pro_w_front" "$AIRFRAME_DST"
chmod +x "$AIRFRAME_DST"
echo "  [OK] Airframe → $AIRFRAME_DST"

# ── 3. Register airframe in CMakeLists if not already there ──────────────────
CMAKE="$PX4_DIR/ROMFS/px4fmu_common/init.d-posix/airframes/CMakeLists.txt"
if ! grep -q "22112_gz_x500_oakd_pro_w_front" "$CMAKE"; then
    sed -i 's/22111_gz_x500_skydio/22111_gz_x500_skydio\n\t22112_gz_x500_oakd_pro_w_front/' "$CMAKE"
    echo "  [OK] Registered airframe in CMakeLists.txt"
else
    echo "  [--] Airframe already registered in CMakeLists.txt"
fi

# ── 4. Bridge config ─────────────────────────────────────────────────────────
cp "$SCRIPT_DIR/camera_bridge_oakd_pro_w_front.yaml" "$ROS2_WS/camera_bridge_oakd_pro_w_front.yaml"
echo "  [OK] Bridge config → $ROS2_WS/camera_bridge_oakd_pro_w_front.yaml"

echo ""
echo "=== Done. Now rebuild PX4 SITL: ==="
echo "  cd $PX4_DIR && make px4_sitl gz_x500_oakd_pro_w_front"
echo ""
echo "=== Then launch sequence: ==="
echo "  T1: MicroXRCEAgent UDP4 --port 8888"
echo "  T2: cd $PX4_DIR && make px4_sitl gz_x500_oakd_pro_w_front"
echo "  T3: ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=\$HOME/ros2_ws/camera_bridge_oakd_pro_w_front.yaml"
echo "  T4: ros2 run module1_camera_verification camera_verifier"
