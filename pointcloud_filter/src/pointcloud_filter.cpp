#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <pcl/common/common.h>
#include <pcl/filters/filter.h> // removeNaNFromPointCloud
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

class PointCloudFilter : public rclcpp::Node {
public:
  PointCloudFilter() : Node("pointcloud_filter") {
    // --- Define ROS Parameters ---
    // Topics
    this->declare_parameter<std::string>("pointcloud_topic",
                                         "/wrist_rgbd_depth_sensor/points");
    this->declare_parameter<std::string>(
        "filtered_pc_topic", "/wrist_rgbd_depth_sensor/points_filtered");
    this->declare_parameter<std::string>("laser_scan_topic", "/scan");

    // Filtering Parameters (RGB-D safe defaults: treat z as DEPTH, keep filter
    // off)
    this->declare_parameter<float>("min_height", 0.10f);  // depth min (m)
    this->declare_parameter<float>("max_height", 3.00f);  // depth max (m)
    this->declare_parameter<bool>("filter_rings", false); // LiDAR-only feature
    this->declare_parameter<bool>("filter_height",
                                  false); // OFF by default for RGB-D
    this->declare_parameter<bool>("publish_filtered_pointcloud", true);
    this->declare_parameter<bool>("publish_laserscan", false);

    // LaserScan Parameters
    this->declare_parameter<float>("scan_angle_min", -M_PI);
    this->declare_parameter<float>("scan_angle_max", M_PI);
    this->declare_parameter<float>("scan_angle_increment", M_PI / 180.0 / 10.0);
    this->declare_parameter<float>("scan_range_min", 0.2f);
    this->declare_parameter<float>("scan_range_max", 200.0f);

    // --- Get Topic Names ---
    std::string pointcloud_topic, filtered_pc_topic, laser_scan_topic;
    this->get_parameter("pointcloud_topic", pointcloud_topic);
    this->get_parameter("filtered_pc_topic", filtered_pc_topic);
    this->get_parameter("laser_scan_topic", laser_scan_topic);

    // --- Publishers ---
    filtered_pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        filtered_pc_topic, 10);
    laser_scan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>(
        laser_scan_topic, 10);

    // --- Subscribers ---
    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic, 10,
        std::bind(&PointCloudFilter::pointCloudCallback, this,
                  std::placeholders::_1));

    // --- Get ROS Parameters (values) ---
    this->get_parameter("min_height", min_height_);
    this->get_parameter("max_height", max_height_);
    this->get_parameter("filter_rings", filter_rings_);
    this->get_parameter("filter_height", filter_height_);
    this->get_parameter("publish_filtered_pointcloud",
                        pub_filtered_pointcloud_);
    this->get_parameter("publish_laserscan", pub_laserscan_);
    this->get_parameter("scan_angle_min", scan_angle_min_);
    this->get_parameter("scan_angle_max", scan_angle_max_);
    this->get_parameter("scan_angle_increment", scan_angle_increment_);
    this->get_parameter("scan_range_min", scan_range_min_);
    this->get_parameter("scan_range_max", scan_range_max_);

    // Calculate LaserScan bins
    num_bins_ = static_cast<int>((scan_angle_max_ - scan_angle_min_) /
                                 scan_angle_increment_) +
                1;

    RCLCPP_INFO(this->get_logger(),
                "pointcloud_filter started. topics: in=%s out=%s | "
                "filter_height=%s [%0.2f..%0.2f] rings=%s",
                pointcloud_topic.c_str(), filtered_pc_topic.c_str(),
                filter_height_ ? "ON" : "OFF", min_height_, max_height_,
                filter_rings_ ? "ON" : "OFF");
  }

private:
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // --- Step 0: Detect intensity field (RGB-D typically no intensity) ---
    bool has_intensity = false;
    for (const auto &field : msg->fields) {
      if (field.name == "intensity") {
        has_intensity = true;
        break;
      }
    }

    // --- Step 1: Convert to a unified PCL type (XYZI) & drop NaNs ---
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud(
        new pcl::PointCloud<pcl::PointXYZI>());
    if (has_intensity) {
      pcl::fromROSMsg(*msg, *input_cloud);
    } else {
      pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(
          new pcl::PointCloud<pcl::PointXYZ>());
      pcl::fromROSMsg(*msg, *temp_cloud);
      input_cloud->points.reserve(temp_cloud->points.size());
      for (const auto &pt : temp_cloud->points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) ||
            !std::isfinite(pt.z))
          continue;
        pcl::PointXYZI pti;
        pti.x = pt.x;
        pti.y = pt.y;
        pti.z = pt.z;
        pti.intensity = 0.0f;
        input_cloud->points.push_back(pti);
      }
      input_cloud->header = temp_cloud->header;
    }
    {
      // Ensure no NaNs remain
      std::vector<int> idx;
      pcl::removeNaNFromPointCloud(*input_cloud, *input_cloud, idx);
    }
    if (input_cloud->points.empty()) {
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "Incoming cloud is empty after conversion/NaN removal.");
      return;
    }

    // --- Step 2: "Ring" filter (keep OFF for RGB-D). If enabled, apply block
    // sampling ---
    pcl::PointCloud<pcl::PointXYZI>::Ptr intermediate_cloud(
        new pcl::PointCloud<pcl::PointXYZI>());
    if (filter_rings_) {
      const int block_size = 16;
      const size_t total_points = input_cloud->points.size();
      const size_t num_blocks = total_points / block_size;
      intermediate_cloud->points.reserve(total_points / 2);
      intermediate_cloud->header = input_cloud->header;

      for (size_t i = 0; i < num_blocks; ++i) {
        if ((i % 2) == 0) { // keep even-indexed blocks
          const size_t base = i * block_size;
          for (int j = 0; j < block_size; ++j) {
            const size_t idx = base + j;
            if (idx < input_cloud->points.size()) {
              const auto &pt = input_cloud->points[idx];
              if (std::isfinite(pt.x) && std::isfinite(pt.y) &&
                  std::isfinite(pt.z)) {
                intermediate_cloud->points.push_back(pt);
              }
            }
          }
        }
      }
      if (intermediate_cloud->points.empty()) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "Ring filter removed all points; passing through.");
        intermediate_cloud = input_cloud;
      }
    } else {
      intermediate_cloud = input_cloud;
    }

    // --- Step 3: "Height" filter (treat z as DEPTH for RGB-D) ---
    pcl::PointCloud<pcl::PointXYZI>::Ptr height_filtered_cloud(
        new pcl::PointCloud<pcl::PointXYZI>());
    if (filter_height_) {
      height_filtered_cloud->points.reserve(intermediate_cloud->points.size());
      for (const auto &pt : intermediate_cloud->points) {
        if (pt.z >= min_height_ && pt.z <= max_height_) {
          height_filtered_cloud->points.push_back(pt);
        }
      }
      if (height_filtered_cloud->points.empty()) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "Depth filter [%0.2f..%0.2f] removed all points; passing through.",
            min_height_, max_height_);
        height_filtered_cloud = intermediate_cloud;
      }
    } else {
      height_filtered_cloud = intermediate_cloud;
    }
    height_filtered_cloud->header = intermediate_cloud->header;

    // --- Step 4: Publish the Filtered 3D Point Cloud ---
    if (pub_filtered_pointcloud_) {
      if (has_intensity) {
        publishFilteredPointCloud(msg->header, height_filtered_cloud);
      } else {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_intensity(
            new pcl::PointCloud<pcl::PointXYZ>());
        cloud_no_intensity->points.reserve(
            height_filtered_cloud->points.size());
        cloud_no_intensity->header = height_filtered_cloud->header;
        for (const auto &pt : height_filtered_cloud->points) {
          cloud_no_intensity->points.emplace_back(
              pcl::PointXYZ{pt.x, pt.y, pt.z});
        }
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud_no_intensity, cloud_msg);
        cloud_msg.header = msg->header;
        filtered_pc_pub_->publish(cloud_msg);
      }
    }

    // --- Step 5: Optional LaserScan ---
    if (pub_laserscan_) {
      publishLaserScan(msg->header, height_filtered_cloud);
    }
  }

  void publishFilteredPointCloud(const std_msgs::msg::Header &header,
                                 pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header = header;
    filtered_pc_pub_->publish(cloud_msg);
  }

  void publishLaserScan(const std_msgs::msg::Header &header,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
    if (cloud->points.empty())
      return;

    sensor_msgs::msg::LaserScan scan_msg;
    scan_msg.header = header;
    scan_msg.angle_min = scan_angle_min_;
    scan_msg.angle_max = scan_angle_max_;
    scan_msg.angle_increment = scan_angle_increment_;
    scan_msg.range_min = scan_range_min_;
    scan_msg.range_max = scan_range_max_;
    scan_msg.ranges.resize(num_bins_, std::numeric_limits<float>::infinity());

    for (const auto &pt : cloud->points) {
      const float angle = std::atan2(pt.y, pt.x);
      const float range = std::hypot(pt.x, pt.y);
      if (range < scan_range_min_ || range > scan_range_max_)
        continue;

      const int index = static_cast<int>((angle - scan_msg.angle_min) /
                                         scan_msg.angle_increment);
      if (index >= 0 && index < num_bins_) {
        scan_msg.ranges[index] = std::min(scan_msg.ranges[index], range);
      }
    }
    laser_scan_pub_->publish(scan_msg);
  }

  // ROS2 interfaces
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      pointcloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_pc_pub_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_pub_;

  // Filtering parameters
  float min_height_;
  float max_height_;
  bool filter_rings_;
  bool filter_height_;
  bool pub_filtered_pointcloud_;
  bool pub_laserscan_;

  // LaserScan parameters
  float scan_angle_min_;
  float scan_angle_max_;
  float scan_angle_increment_;
  float scan_range_min_;
  float scan_range_max_;
  int num_bins_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PointCloudFilter>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
