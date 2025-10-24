#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_listener.h>

#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("object_manipulation");
static const std::string PLANNING_GROUP_ROBOT = "ur_manipulator";
static const std::string PLANNING_GROUP_GRIPPER = "gripper";

class ObjectManipulation {
public:
  explicit ObjectManipulation(const rclcpp::Node::SharedPtr &base_node)
      : base_node_(base_node) {
    RCLCPP_INFO(LOGGER, "Initializing Pick and Place with Perception");

    // MoveGroup runs on its own node + executor
    rclcpp::NodeOptions opts;
    opts.automatically_declare_parameters_from_overrides(true);
    move_group_node_ =
        rclcpp::Node::make_shared("object_manipulation_node", opts);
    exec_.add_node(move_group_node_);
    std::thread([this]() { exec_.spin(); }).detach();

    // MoveGroup interfaces
    arm_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        move_group_node_, PLANNING_GROUP_ROBOT);
    gripper_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        move_group_node_, PLANNING_GROUP_GRIPPER);

    // TF buffer/listener (kept intact; unused with fixed pose)
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(base_node_->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Planning frame & basic settings
    target_frame_ = arm_->getPlanningFrame(); // e.g., "world" or "base_link"
    arm_->setPoseReferenceFrame(target_frame_);
    arm_->setPlanningTime(5.0);
    arm_->setNumPlanningAttempts(10);
    arm_->setGoalPositionTolerance(0.005);
    arm_->setGoalOrientationTolerance(0.05);
    arm_->setMaxVelocityScalingFactor(0.3);
    arm_->setMaxAccelerationScalingFactor(0.3);

    // Basic prints
    RCLCPP_INFO(LOGGER, "Planning frame: %s", arm_->getPlanningFrame().c_str());
    RCLCPP_INFO(LOGGER, "EE link:        %s",
                arm_->getEndEffectorLink().c_str());

    arm_->setStartStateToCurrentState();
    gripper_->setStartStateToCurrentState();

    RCLCPP_INFO(LOGGER, "Ready. Using fixed cup pose.");
  }

  void runOnce() {
    // Use fixed cup pose (no /cup_pose subscription)
    have_pose_ = true;
    cup_pose_base_.pose.position.x = 0.302;
    cup_pose_base_.pose.position.y = 0.330;
    cup_pose_base_.pose.position.z = 0.035;

    RCLCPP_INFO(LOGGER, "Cup pose in planning frame '%s': [%.3f, %.3f, %.3f]",
                target_frame_.c_str(), cup_pose_base_.pose.position.x,
                cup_pose_base_.pose.position.y, cup_pose_base_.pose.position.z);

    // === First step only: hover 29 cm above the cup and stop ===
    const double hover_above = 0.29;

    // (optional) Ensure gripper is open before moving
    setGripperNamed("gripper_open");
    planAndExecGripper();
    gripper_->setStartStateToCurrentState();

    geometry_msgs::msg::Pose hover = cup_pose_base_.pose;
    hover.position.y -= 0.015;
    hover.position.z += hover_above;

    // Tool Z pointing down: 180° about Y → quaternion (0, 1, 0, 0)
    hover.orientation.x = 0.0;
    hover.orientation.y = 1.0;
    hover.orientation.z = 0.0;
    hover.orientation.w = 0.0;

    arm_->setPoseTarget(hover);
    planAndExecArm("Hover_29cm");
    arm_->setStartStateToCurrentState();

    RCLCPP_INFO(LOGGER, "Stopped at hover position 29 cm above the cup. "
                        "Exiting without approach/grasp.");
  }

private:
  using Plan = moveit::planning_interface::MoveGroupInterface::Plan;

  // Nodes/executor
  rclcpp::Node::SharedPtr base_node_;
  rclcpp::Node::SharedPtr move_group_node_;
  rclcpp::executors::SingleThreadedExecutor exec_;

  // MoveIt
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> gripper_;

  // TF2 (kept, unused for fixed pose)
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::string target_frame_; // planning frame

  // (subscription removed)

  // Data
  bool have_pose_{false};
  geometry_msgs::msg::PoseStamped
      cup_pose_base_; // now directly assigned (fixed pose)

  // (poseCb removed)

  // Helpers
  void planAndExecArm(const char *tag) {
    Plan plan;
    auto ok = (arm_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    if (!ok) {
      RCLCPP_WARN(LOGGER, "[%s] arm plan failed", tag);
      return;
    }
    arm_->execute(plan);
    RCLCPP_INFO(LOGGER, "[%s] arm execute success", tag);
  }

  void setGripperNamed(const std::string &name) {
    gripper_->setNamedTarget(name);
  }

  void planAndExecGripper() {
    Plan plan;
    auto ok = (gripper_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    if (!ok) {
      RCLCPP_WARN(LOGGER, "gripper plan failed");
      return;
    }
    gripper_->execute(plan);
    RCLCPP_INFO(LOGGER, "gripper execute success");
  }

  void cartesianDelta(double dx, double dy, double dz, const char *tag) {
    std::vector<geometry_msgs::msg::Pose> wps;
    auto cur = arm_->getCurrentPose().pose;
    wps.push_back(cur);
    cur.position.x += dx;
    cur.position.y += dy;
    cur.position.z += dz;
    wps.push_back(cur);

    moveit_msgs::msg::RobotTrajectory traj;
    const double eef_step = 0.01;
    const double jump_thr = 0.0;
    double fraction = arm_->computeCartesianPath(wps, eef_step, jump_thr, traj);
    if (fraction <= 0.0) {
      RCLCPP_WARN(LOGGER, "[%s] cartesian plan failed (fraction=%.2f)", tag,
                  fraction);
      return;
    }
    arm_->execute(traj);
    RCLCPP_INFO(LOGGER, "[%s] cartesian execute success (fraction=%.2f)", tag,
                fraction);
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto base = std::make_shared<rclcpp::Node>("object_manipulation");
  ObjectManipulation app(base);
  app.runOnce();
  rclcpp::shutdown();
  return 0;
}
