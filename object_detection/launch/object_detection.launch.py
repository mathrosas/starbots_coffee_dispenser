import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory("object_detection"),
        "rviz",
        "config.rviz",
    )

    detection_params = {
        "use_sim_time": True,
        "color_topic": "/wrist_rgbd_depth_sensor/image_raw",
        "depth_topic": "/wrist_rgbd_depth_sensor/depth/image_raw",
        "camera_info_topic": "/wrist_rgbd_depth_sensor/camera_info",
        "pointcloud_topic": "/wrist_rgbd_depth_sensor/points",
    }

    object_detection = Node(
        package="object_detection",
        executable="object_detection",
        name="object_detection",
        output="screen",
        parameters=[detection_params],
        arguments=["--ros-args", "--log-level", "info"],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config, "--ros-args", "--log-level", "info"],
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription([object_detection, rviz_node])
