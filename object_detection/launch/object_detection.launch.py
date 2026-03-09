from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    pointcloud_filter = Node(
        package="pointcloud_filter",
        executable="pointcloud_filter_node",
        name="pointcloud_filter_node",
        output="screen",
        parameters=[
            {
                "pointcloud_topic": "/wrist_rgbd_depth_sensor/points",
                "filtered_pc_topic": "/wrist_rgbd_depth_sensor/points_filtered",
            }
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    object_detection = Node(
        package="object_detection",
        executable="object_detection",
        name="object_detection",
        output="screen",
        arguments=["--ros-args", "--log-level", "info"],
    )

    delayed_object_detection = TimerAction(period=1.0, actions=[object_detection])

    return LaunchDescription([pointcloud_filter, delayed_object_detection])
