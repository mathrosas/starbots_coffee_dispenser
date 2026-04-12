#!/bin/bash
set -e
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/home/user/ros2_ws/install/setup.bash"
exec bash -c "$@"
