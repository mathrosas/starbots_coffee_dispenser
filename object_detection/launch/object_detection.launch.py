import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory("object_detection"),
        "rviz",
        "config.rviz",
    )

    depth_to_points_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("object_detection"),
                "launch",
                "depth_to_points.launch.py",
            )
        )
    )

    object_detection = Node(
        package="object_detection",
        executable="object_detection",
        name="object_detection",
        output="screen",
        parameters=[{"pointcloud_topic": "/D415/barista_points"}, {"use_sim_time": False}],
        arguments=["--ros-args", "--log-level", "info"],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config, "--ros-args", "--log-level", "info"],
        parameters=[{"use_sim_time": False}],
    )

    return LaunchDescription([depth_to_points_launch, object_detection, rviz_node])
