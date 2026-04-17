#include <geometry_msgs/msg/pose.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace std::chrono_literals;

class AddCafeteriaScene : public rclcpp::Node {
public:
  AddCafeteriaScene()
      : Node("add_cafeteria_scene"),
        planning_scene_interface_(
            std::make_shared<
                moveit::planning_interface::PlanningSceneInterface>()) {
    declare_parameter<std::string>("planning_frame", "world");
    declare_parameter<double>("robot_spawn_x", 13.9);
    declare_parameter<double>("robot_spawn_y", -18.56);
    declare_parameter<double>("robot_spawn_z", 1.032);
    declare_parameter<bool>("add_debug_cup", false);

    // Delay the scene application slightly so move_group is already up.
    timer_ = create_wall_timer(
        1500ms, std::bind(&AddCafeteriaScene::populatePlanningScene, this));
  }

private:
  struct BoxSpec {
    std::string id;
    double size_x;
    double size_y;
    double size_z;
    double world_x;
    double world_y;
    double world_z;
    double yaw_rad;
  };

  geometry_msgs::msg::Pose makePose(double x, double y, double z,
                                    double yaw_rad = 0.0) const {
    geometry_msgs::msg::Pose pose;
    pose.position.x = x;
    pose.position.y = y;
    pose.position.z = z;

    pose.orientation.x = 0.0;
    pose.orientation.y = 0.0;
    pose.orientation.z = std::sin(yaw_rad * 0.5);
    pose.orientation.w = std::cos(yaw_rad * 0.5);
    return pose;
  }

  moveit_msgs::msg::CollisionObject makeBox(const BoxSpec &spec, double robot_x,
                                            double robot_y,
                                            double robot_z) const {
    moveit_msgs::msg::CollisionObject object;
    object.header.frame_id = get_parameter("planning_frame").as_string();
    object.id = spec.id;
    object.operation = moveit_msgs::msg::CollisionObject::ADD;

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    primitive.dimensions = {spec.size_x, spec.size_y, spec.size_z};
    object.primitives.push_back(primitive);

    // MoveIt world is centered on the robot's fixed base, while the Gazebo
    // models are placed in absolute world coordinates. Convert Gazebo world
    // positions into robot-centered planning scene coordinates.
    object.primitive_poses.push_back(
        makePose(spec.world_x - robot_x, spec.world_y - robot_y,
                 spec.world_z - robot_z, spec.yaw_rad));
    return object;
  }

  void populatePlanningScene() {
    timer_->cancel();

    const auto planning_frame = get_parameter("planning_frame").as_string();
    const double robot_x = get_parameter("robot_spawn_x").as_double();
    const double robot_y = get_parameter("robot_spawn_y").as_double();
    const double robot_z = get_parameter("robot_spawn_z").as_double();
    const bool add_debug_cup = get_parameter("add_debug_cup").as_bool();

    RCLCPP_INFO(get_logger(),
                "Adding cafeteria collision scene in frame '%s' using Gazebo "
                "robot spawn (%.3f, %.3f, %.3f)",
                planning_frame.c_str(), robot_x, robot_y, robot_z);

    // These obstacles come from:
    //   worlds/starbots_complete.world
    //   models/starbots_bartender_dispenser/model.sdf
    //   models/coffee_machine/model.sdf
    //
    // The goal is to mirror the collision-relevant cafeteria geometry around
    // the UR3e with stable primitives that MoveIt can plan against.
    std::vector<BoxSpec> box_specs = {
        // Main bartender dispenser body.
        // Gazebo model pose: (14.2, -18.2, 0.5)
        // Box size from SDF: 0.5 x 1.8 x 1.0
        {"bartender_dispenser_body", 0.50, 1.80, 1.00, 14.20, -18.20, 0.50,
         0.0},

        // Table top slab on top of the dispenser.
        // Local pose inside the model: z = +0.5
        // Therefore center z in Gazebo world = 0.5 + 0.5 = 1.0
        {"bartender_dispenser_top", 0.85, 1.81, 0.05, 14.20, -18.20, 1.00, 0.0},

        // Conservative approximation of the laboratory fixture attached to the
        // right side of the dispenser table
        {"side_guard", 0.15, 0.35, 0.15, 13.75, -18.10, 1.00, 0.00},

        // Table top center in world: (14.20, -18.20, 1.00)
        // Table top size: 0.85 x 1.81 x 0.05
        // Rear edge y = -18.20 - 1.81/2 = -19.105
        // Wall thickness is along y, so its center is offset by half thickness.
        {"wall", 1.85, 0.04, 1.80, 14.00, -19.125, 0.95, 0.0},

        // Coffee machine approximation. The Gazebo coffee_machine uses a mesh,
        // so here we add a conservative box that protects the planning space.
        // The world pose in the world file is (14.0, -17.7, 1.0) with yaw 1.57.
        // The chosen dimensions are an engineering approximation sized to the
        // visible machine volume, not an exact mesh fit.
        {"coffee_machine", 0.28, 0.32, 0.42, 14.00, -17.70, 1.18, 1.57},

        // Rear guard panel behind the coffee machine to approximate the bulky
        // back geometry that the mesh occupies near the dispenser edge.
        {"coffee_machine_back_guard", 0.08, 0.36, 0.55, 13.86, -17.70, 1.16,
         1.57},
    };

    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    collision_objects.reserve(box_specs.size() + (add_debug_cup ? 1 : 0));

    for (const auto &spec : box_specs) {
      auto object = makeBox(spec, robot_x, robot_y, robot_z);
      const auto &pose = object.primitive_poses.front();
      RCLCPP_INFO(get_logger(),
                  "Prepared '%s' in %s at relative pose x=%.3f y=%.3f z=%.3f",
                  object.id.c_str(), planning_frame.c_str(), pose.position.x,
                  pose.position.y, pose.position.z);
      collision_objects.push_back(object);
    }

    if (add_debug_cup) {
      moveit_msgs::msg::CollisionObject cup;
      cup.header.frame_id = planning_frame;
      cup.id = "debug_cup";
      cup.operation = moveit_msgs::msg::CollisionObject::ADD;

      shape_msgs::msg::SolidPrimitive primitive;
      primitive.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
      primitive.dimensions = {0.15, 0.03};
      cup.primitives.push_back(primitive);

      // Cup pose from the world file.
      cup.primitive_poses.push_back(
          makePose(14.16 - robot_x, -18.19 - robot_y, 1.025 - robot_z, 0.0));
      collision_objects.push_back(cup);
    }

    // Remove old versions first so re-running the node updates the scene
    // cleanly.
    planning_scene_interface_->removeCollisionObjects(
        {"bartender_dispenser_body", "bartender_dispenser_top", "wall",
         "side_guard", "coffee_machine", "coffee_machine_back_guard",
         "debug_cup", "table", "cup"});

    rclcpp::sleep_for(250ms);

    planning_scene_interface_->applyCollisionObjects(collision_objects);

    RCLCPP_INFO(get_logger(),
                "Applied %zu collision objects to the MoveIt planning scene.",
                collision_objects.size());
    RCLCPP_INFO(
        get_logger(),
        "If RViz does not show them immediately, make sure the Planning "
        "Scene display is enabled and its fixed frame is 'world'.");
  }

  std::shared_ptr<moveit::planning_interface::PlanningSceneInterface>
      planning_scene_interface_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AddCafeteriaScene>();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
