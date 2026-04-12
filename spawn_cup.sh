#!/bin/bash
# Spawn a coffee cup in the Gazebo simulation within reach of the UR3e arm.
# Each invocation creates a uniquely named entity to avoid collisions.

RANDOM_ID=$(( RANDOM % 9000 + 1000 ))
ENTITY_NAME="cup${RANDOM_ID}"

MODEL_PATH="/home/user/ros2_ws/src/universal_robot_ros2/the_construct_office_gazebo/models/portable_cup_2/model.sdf"

# Spawn pose (on the table, within UR3e workspace)
X=14.1
Y=-18.2
Z=1.1
R=1.57
P=0
YAW=0

source /opt/ros/humble/setup.bash
ros2 run gazebo_ros spawn_entity.py \
    -file "${MODEL_PATH}" \
    -x "${X}" -y "${Y}" -z "${Z}" \
    -R "${R}" -P "${P}" -Y "${YAW}" \
    -entity "${ENTITY_NAME}"
