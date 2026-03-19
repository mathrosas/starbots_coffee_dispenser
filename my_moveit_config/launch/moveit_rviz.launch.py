import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("name", package_name="my_moveit_config")
        .robot_description(file_path="config/name.urdf.xacro")
        .robot_description_semantic(file_path="config/name.srdf")
        .sensors_3d(file_path="config/sensors_3d.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )

    rviz_config = os.path.join(
        get_package_share_directory("my_moveit_config"),
        "config",
        "moveit.rviz",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[moveit_config.to_dict(), {"use_sim_time": False}],
    )

    return LaunchDescription([rviz_node])
