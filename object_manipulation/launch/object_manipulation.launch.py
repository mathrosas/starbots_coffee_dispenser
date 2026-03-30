import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory("object_detection"),
        "rviz",
        "config.rviz",
    )
    bt_xml_path = os.path.join(
        get_package_share_directory("object_manipulation"),
        "bt_config",
        "deliver_cup_tree.xml",
    )

    moveit_config = (
        MoveItConfigsBuilder("name", package_name="my_moveit_config")
        .robot_description(file_path="config/name.urdf.xacro")
        .robot_description_semantic(file_path="config/name.srdf")
        .sensors_3d(file_path="config/sensors_3d.yaml")
        .planning_pipelines(
            default_planning_pipeline="ompl",
            pipelines=["ompl", "pilz_industrial_motion_planner"],
        )
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"trajectory_execution.allowed_execution_duration_scaling": 5.0},
            {"publish_robot_description_semantic": True},
            {
                "planning_pipelines": [
                    "ompl",
                    "pilz_industrial_motion_planner",
                ]
            },
            {"default_planning_pipeline": "ompl"},
            {"use_sim_time": False},
        ],
    )

    add_scene_node = Node(
        package="object_manipulation",
        executable="add_cafeteria_scene",
        name="add_cafeteria_scene",
        output="screen",
        parameters=[{"use_sim_time": False}],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[moveit_config.to_dict(), {"use_sim_time": False}],
    )

    manipulation_node = Node(
        package="object_manipulation",
        executable="object_manipulation",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": False},
            {"bt_xml_path": bt_xml_path},
            {"bt_enable_groot": True},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    delayed_manipulation_node = TimerAction(
        period=3.0,
        actions=[manipulation_node],
    )

    deliver_cup_bridge_node = Node(
        package="object_manipulation",
        executable="deliver_cup_bridge",
        name="deliver_cup_bridge",
        output="screen",
        parameters=[{"use_sim_time": False}],
    )

    return LaunchDescription(
        [
            move_group_node,
            add_scene_node,
            rviz_node,
            deliver_cup_bridge_node,
            delayed_manipulation_node,
        ]
    )
