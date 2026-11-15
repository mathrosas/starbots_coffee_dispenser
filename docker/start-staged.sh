#!/usr/bin/env bash
# Starbots ROS 2 Humble / Gazebo Classic staged startup.
# Keep nounset (-u) disabled: ROS setup.bash uses unset variables.
set -Eeo pipefail

source /opt/ros/humble/setup.bash
source /home/user/ros2_ws/install/setup.bash

pids=()

cleanup() {
    trap - EXIT INT TERM
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

server_running() {
    pgrep -x gzserver >/dev/null
}

wait_for_model() {
    local model="$1"
    local names

    for attempt in {1..60}; do
        if ! server_running; then
            echo "ERROR: Gazebo server stopped while waiting for $model." >&2
            return 1
        fi

        names=$(timeout 5 ros2 topic echo \
            /gazebo/model_states --field name --once \
            2>/dev/null || true)

        if grep -Fqw "$model" <<< "$names"; then
            echo "Model ready: $model"
            return 0
        fi

        sleep 2
    done

    echo "ERROR: Model $model did not appear." >&2
    return 1
}

# Query the ROS 2 service directly, rather than parsing ANSI-colored
# output from `ros2 control list_controllers`.
wait_for_controllers() {
    python3 - <<'PY'
import sys
import time

import rclpy
from controller_manager_msgs.srv import ListControllers

REQUIRED = {
    'joint_state_broadcaster',
    'joint_trajectory_controller',
    'gripper_controller',
}
SERVICE = '/controller_manager/list_controllers'
DEADLINE_SECONDS = 120.0

rclpy.init()
node = rclpy.create_node('starbots_startup_controller_check')
client = node.create_client(ListControllers, SERVICE)

deadline = time.monotonic() + DEADLINE_SECONDS
next_report = time.monotonic()
states = {}

try:
    while rclpy.ok() and time.monotonic() < deadline:
        if not client.wait_for_service(timeout_sec=1.0):
            if time.monotonic() >= next_report:
                print(f'Waiting for ROS service {SERVICE} ...', flush=True)
                next_report = time.monotonic() + 10.0
            continue

        request = ListControllers.Request()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)

        if future.done():
            try:
                response = future.result()
                if response is not None:
                    states = {controller.name: controller.state
                              for controller in response.controller}
                    if all(states.get(name) == 'active' for name in REQUIRED):
                        for name in sorted(REQUIRED):
                            print(f'Controller ready: {name} (active)', flush=True)
                        sys.exit(0)
            except Exception as exc:
                print(f'Controller service call failed: {exc}', flush=True)

        if time.monotonic() >= next_report:
            print(f'Waiting for controllers. Current states: {states}', flush=True)
            next_report = time.monotonic() + 10.0
        time.sleep(1.0)

    print('ERROR: Controllers did not all become active within 120 seconds.',
          file=sys.stderr, flush=True)
    print(f'Last controller states: {states}', file=sys.stderr, flush=True)
    sys.exit(1)
finally:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
PY
}

echo '========== STAGE 1: STARTING GAZEBO =========='
ros2 launch the_construct_office_gazebo \
    start_world_standalone.launch.py &
pids+=("$!")

ready=false
for attempt in {1..120}; do
    if server_running && \
       ros2 service list 2>/dev/null | grep -Fxq /spawn_entity; then
        if timeout 10 ros2 topic echo \
            /gazebo/model_states --once >/dev/null 2>&1; then
            ready=true
            break
        fi
    fi

    if ! kill -0 "${pids[0]}" 2>/dev/null; then
        echo 'ERROR: Gazebo launch stopped.' >&2
        exit 1
    fi
    sleep 2
done

if [[ "$ready" != true ]]; then
    echo 'ERROR: Gazebo did not become ready.' >&2
    exit 1
fi
echo 'Gazebo is ready.'

echo '========== STAGE 2: SPAWNING UR3e =========='
ros2 launch ur_simulation_gazebo ur_sim_starbots.launch.py &
pids+=("$!")
wait_for_model ur

echo '========== STAGE 3: WAITING FOR CONTROLLERS =========='
wait_for_controllers
echo 'All UR3e controllers are active.'

echo '========== STAGE 4: SPAWNING BARISTA =========='
ros2 launch barista_gazebo barista_1.launch.py &
pids+=("$!")
wait_for_model barista_1

echo '============================================'
echo '       STARBOTS SIMULATION READY'
echo '============================================'

while server_running; do
    for pid in "${pids[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo 'ERROR: A launch process stopped.' >&2
            exit 1
        fi
    done
    sleep 3
done

echo 'ERROR: Gazebo server stopped.' >&2
exit 1