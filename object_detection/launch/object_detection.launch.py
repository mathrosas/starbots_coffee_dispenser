from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pointcloud_filter = Node(
        package='pointcloud_filter',
        executable='pointcloud_filter',
        name='pointcloud_filter',
        output='screen'
    )

    object_detection = Node(
        package='object_detection',
        executable='object_detection',
        name='object_detection',
        output='screen',
        arguments=["--ros-args", "--log-level", "info"]
    )

    return LaunchDescription([pointcloud_filter, object_detection])
