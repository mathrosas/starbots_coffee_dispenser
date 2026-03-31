from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    point_cloud_container = ComposableNodeContainer(
        name="wrist_pointcloud_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        arguments=["--ros-args", "--log-level", "warn"],
        composable_node_descriptions=[
            ComposableNode(
                package="depth_image_proc",
                plugin="depth_image_proc::PointCloudXyzrgbNode",
                name="point_cloud_converter_node",
                parameters=[{"queue_size": 15}],
                remappings=[
                    ("rgb/camera_info", "/wrist_rgbd_depth_sensor/camera_info"),
                    ("rgb/image_rect_color", "/wrist_rgbd_depth_sensor/image_raw"),
                    ("depth_registered/image_rect", "/wrist_rgbd_depth_sensor/depth/image_raw"),
                    ("points", "/wrist_rgbd_depth_sensor/points_be"),
                ],
            ),
        ],
        output="screen",
    )

    pcl_qos_bridge = Node(
        package="object_detection",
        executable="pcl_qos_conv",
        name="pcl_qos_conv",
        output="screen",
    )

    return LaunchDescription(
        [
            point_cloud_container,
            pcl_qos_bridge,
        ]
    )
