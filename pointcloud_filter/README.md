# pointcloud_filter

Single-stream point cloud ROI filter for the wrist RGB-D camera.

## Behavior
- Subscribes to one input `sensor_msgs/PointCloud2` topic.
- Publishes one filtered `sensor_msgs/PointCloud2` topic.
- Keeps points that satisfy:
  - `x >= -0.35`
  - `y >= -0.15`
  - `y <= 0.30`
  - `z <= 0.90`
- Drops NaN/Inf points.

## ROS Parameters
- `pointcloud_topic` (default: `/wrist_rgbd_depth_sensor/points`)
- `filtered_pc_topic` (default: `/wrist_rgbd_depth_sensor/points_filtered`)

## Executable and Node Name
- Executable: `pointcloud_filter_node`
- Node: `pointcloud_filter_node`
