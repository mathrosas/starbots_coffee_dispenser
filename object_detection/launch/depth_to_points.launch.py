from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    static_tf_world_to_d415 = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_world_to_d415",
        arguments=[
            "-0.408",
            "-0.290",
            "0.395",
            "-0.107",
            "0.107",
            "0.215",
            "0.210",
            "world",
            "D415_link",
        ],
        output="screen",
    )

    point_cloud_container = ComposableNodeContainer(
        name="d415_pointcloud_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        arguments=["--ros-args", "--log-level", "warn"],
        composable_node_descriptions=[
            ComposableNode(
                package="depth_image_proc",
                plugin="depth_image_proc::PointCloudXyzrgbNode",
                name="point_cloud_converter_node",
                remappings=[
                    ("rgb/camera_info", "/D415/aligned_depth_to_color/camera_info"),
                    ("rgb/image_rect_color", "/D415/color/image_raw"),
                    ("depth_registered/image_rect", "/D415/aligned_depth_to_color/image_raw"),
                    ("points", "/D415/barista_points_be"),
                ],
            ),
        ],
        output="screen",
    )

    pcl_qos_conv_node = Node(
        package="object_detection",
        executable="pcl_qos_conv",
        name="pcl_qos_conv",
        output="screen",
    )

    return LaunchDescription(
        [
            static_tf_world_to_d415,
            point_cloud_container,
            pcl_qos_conv_node,
        ]
    )
