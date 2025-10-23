import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('object_detection')
    rviz_config = os.path.join(pkg_share, 'rviz', 'config.rviz')

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
        output='screen'
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config]
    )

    return LaunchDescription([pointcloud_filter, object_detection, rviz])
