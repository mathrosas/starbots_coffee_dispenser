#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2


class PclQosConv(Node):
    """Convert best-effort wrist pointcloud stream into reliable stream."""

    def __init__(self) -> None:
        super().__init__("pcl_qos_conv_node")

        pub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        sub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )

        self.publisher_ = self.create_publisher(
            PointCloud2, "/wrist_rgbd_depth_sensor/points", pub_qos
        )
        self.subscription_ = self.create_subscription(
            PointCloud2,
            "/wrist_rgbd_depth_sensor/points_be",
            self.callback,
            sub_qos,
        )

        self.get_logger().info("PointCloud QoS bridge initialized")

    def callback(self, msg: PointCloud2) -> None:
        self.publisher_.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PclQosConv()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
