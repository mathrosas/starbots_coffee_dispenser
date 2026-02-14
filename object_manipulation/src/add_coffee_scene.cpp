#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <string>
#include <thread>

class AddStarbotsScene : public rclcpp::Node {
public:
  AddStarbotsScene() : Node("add_starbots_scene") {
    // Initialize PlanningSceneInterface
    planning_scene_interface_ =
        std::make_shared<moveit::planning_interface::PlanningSceneInterface>();

    // Add table as a box collision object
    moveit_msgs::msg::CollisionObject table;
    table.header.frame_id = "world"; // Relative to base_link since attached
    table.id = "table";
    table.operation = moveit_msgs::msg::CollisionObject::ADD;

    // Table shape: Box with dimensions [length, width, height] in meters
    shape_msgs::msg::SolidPrimitive table_primitive;
    table_primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    table_primitive.dimensions = {0.85, 1.81,
                                  0.05}; // Flat table top; adjust size
    table.primitives.push_back(table_primitive);

    // Table pose: Positioned under UR3 base (e.g., UR3 base at z=0.05 above
    // table, in left bottom corner of table with 0.1 padding)
    geometry_msgs::msg::PoseStamped table_pose;
    table_pose.header.frame_id = "world";
    table_pose.pose.position.x = 14.2;
    table_pose.pose.position.y = -18.2;
    table_pose.pose.position.z = 1.0;

    table_pose.pose.orientation.x = 0.0;
    table_pose.pose.orientation.y = 0.0;
    table_pose.pose.orientation.z = 0.0;
    table_pose.pose.orientation.w = 1.0;
    table.primitive_poses.push_back(table_pose.pose);

    // Apply to scene
    planning_scene_interface_->applyCollisionObject(table);
    RCLCPP_INFO(get_logger(),
                "Added table to planning scene as external obstacle");

    // auto allowCollision = [&](const std::string &a, const std::string &b) {
    //   moveit_msgs::msg::PlanningScene ps;
    //   ps.is_diff = true;

    //   auto &acm = ps.allowed_collision_matrix;
    //   acm.entry_names = {a, b};

    //   moveit_msgs::msg::AllowedCollisionEntry row_a;
    //   row_a.enabled = {true, true}; // columns: {a, b}

    //   moveit_msgs::msg::AllowedCollisionEntry row_b;
    //   row_b.enabled = {true, true}; // columns: {a, b}

    //   acm.entry_values = {row_a, row_b};

    //   // Apply the diff (no need to publish to /planning_scene yourself)
    //   planning_scene_interface_->applyPlanningScene(ps);
    // };
    // allowCollision("table", "base_link");
    // RCLCPP_INFO(get_logger(), "ACM: allowed table <-> base_link");

    // Add wall (tall thin box)
    moveit_msgs::msg::CollisionObject wall;
    wall.header.frame_id = "world";
    wall.id = "wall";
    wall.operation = moveit_msgs::msg::CollisionObject::ADD;

    shape_msgs::msg::SolidPrimitive wall_primitive;
    wall_primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    wall_primitive.dimensions = {1.6, 0.05, 2.0}; // width, thickness, height
    wall.primitives.push_back(wall_primitive);

    geometry_msgs::msg::PoseStamped wall_pose;
    wall_pose.header.frame_id = "world";
    wall_pose.pose.position.x = 0.3; // Behind the table
    wall_pose.pose.position.y = 0.525;
    wall_pose.pose.position.z = 0.0;
    wall_pose.pose.orientation.w = 1.0; // No rotation
    wall.primitive_poses.push_back(wall_pose.pose);

    // Apply to scene
    // MR
    // planning_scene_interface_->applyCollisionObject(wall);
    RCLCPP_INFO(get_logger(), "Added wall to planning scene");

    RCLCPP_INFO(get_logger(),
                "Scene objects added successfully. Node spinning...");
  }

private:
  std::shared_ptr<moveit::planning_interface::PlanningSceneInterface>
      planning_scene_interface_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AddStarbotsScene>();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}