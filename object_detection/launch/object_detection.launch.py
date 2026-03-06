from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    object_detection = Node(
        package="object_detection",
        executable="object_detection",
        name="object_detection",
        output="screen",
        arguments=["--ros-args", "--log-level", "info"],
    )

    return LaunchDescription([object_detection])
