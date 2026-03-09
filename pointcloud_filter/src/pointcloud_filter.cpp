#include <cmath>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

class PointCloudFilter : public rclcpp::Node {
public:
  PointCloudFilter() : Node("pointcloud_filter_node") {
    this->declare_parameter<std::string>("pointcloud_topic",
                                         "/wrist_rgbd_depth_sensor/points");
    this->declare_parameter<std::string>(
        "filtered_pc_topic", "/wrist_rgbd_depth_sensor/points_filtered");

    std::string pointcloud_topic;
    std::string filtered_pc_topic;
    this->get_parameter("pointcloud_topic", pointcloud_topic);
    this->get_parameter("filtered_pc_topic", filtered_pc_topic);

    filtered_pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        filtered_pc_topic, 10);

    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic, 10,
        std::bind(&PointCloudFilter::pointCloudCallback, this,
                  std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(),
                "pointcloud_filter_node started. topics: in=%s out=%s | "
                "ROI: x>=-0.35, y in [-0.15,0.30], z<=0.9",
                pointcloud_topic.c_str(), filtered_pc_topic.c_str());
  }

private:
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    sensor_msgs::PointCloud2Iterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*msg, "z");

    sensor_msgs::msg::PointCloud2 filtered_cloud;
    filtered_cloud.header = msg->header;
    filtered_cloud.is_dense = msg->is_dense;
    filtered_cloud.is_bigendian = msg->is_bigendian;
    filtered_cloud.fields = msg->fields;
    filtered_cloud.point_step = msg->point_step;
    filtered_cloud.row_step = 0;
    filtered_cloud.data.clear();
    filtered_cloud.data.reserve(msg->data.size());

    size_t valid_points = 0;
    for (size_t i = 0; i < msg->width * msg->height;
         ++i, ++iter_x, ++iter_y, ++iter_z) {
      const float x = *iter_x;
      const float y = *iter_y;
      const float z = *iter_z;

      if (std::isnan(x) || std::isnan(y) || std::isnan(z) || std::isinf(x) ||
          std::isinf(y) || std::isinf(z)) {
        continue;
      }

      if (x >= -0.35f && y >= -0.15f && y <= 0.3f && z <= 0.9f) {
        filtered_cloud.data.insert(
            filtered_cloud.data.end(), msg->data.begin() + i * msg->point_step,
            msg->data.begin() + (i + 1) * msg->point_step);
        ++valid_points;
      }
    }

    if (valid_points == 0) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "No points left after filtering.");
      return;
    }

    filtered_cloud.width = valid_points;
    filtered_cloud.height = 1;
    filtered_cloud.row_step = filtered_cloud.width * filtered_cloud.point_step;
    filtered_pc_pub_->publish(filtered_cloud);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      pointcloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_pc_pub_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PointCloudFilter>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
