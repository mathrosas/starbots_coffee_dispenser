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
        "deliver_coffee_tree.xml",
    )

    moveit_config = (
        MoveItConfigsBuilder("name", package_name="my_moveit_config")
        .planning_pipelines(
            default_planning_pipeline="ompl",
            pipelines=["ompl", "pilz_industrial_motion_planner"],
        )
        .to_moveit_configs()
    )

    object_detection_node = Node(
        package="object_detection",
        executable="object_detection",
        name="object_detection",
        output="screen",
        arguments=["--ros-args", "--log-level", "info"],
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
        executable="add_coffee_scene",
        name="add_coffee_scene",
        output="screen",
        parameters=[{"use_sim_time": False}],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": False}],
    )

    object_manipulation_node = Node(
        name="object_manipulation",
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

    delayed_object_manipulation_node = TimerAction(
        period=5.0,
        actions=[object_manipulation_node],
    )

    return LaunchDescription(
        [
            object_detection_node,
            move_group_node,
            add_scene_node,
            rviz_node,
            delayed_object_manipulation_node,
        ]
    )
