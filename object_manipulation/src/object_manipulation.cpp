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
    RCLCPP_INFO(LOGGER, "Initializing Object Manipulation");

    // Subscribe to /cup_pose (PoseStamped, typically camera/depth frame)
    sub_pose_ =
        base_node_->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/cup_pose", rclcpp::QoS(10),
            std::bind(&ObjectManipulation::poseCb, this,
                      std::placeholders::_1));

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

    // TF buffer/listener to transform cup pose into planning frame
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

    RCLCPP_INFO(LOGGER, "Ready. Waiting for /cup_pose (will fallback to fixed "
                        "position if not received)...");
  }

  void runOnce() {
    // === 0) Go to home, open gripper ===
    if (!goNamedArm("home")) {
      RCLCPP_WARN(LOGGER, "Failed to go to 'home', continuing anyway.");
    }
    if (!goNamedGripper("open")) {
      RCLCPP_WARN(LOGGER, "Failed to open gripper, continuing anyway.");
    }

    // === 1) Wait briefly for /cup_pose, else fallback to fixed cup_position_
    // ===
    geometry_msgs::msg::PoseStamped cup_in_base;
    if (!waitForCupPose(cup_in_base, 2.0 /*seconds*/)) {
      RCLCPP_WARN(LOGGER,
                  "No /cup_pose received in time; using fallback fixed pose.");
      cup_in_base.header.frame_id = target_frame_;
      cup_in_base.pose.position.x = cup_guess_.x;
      cup_in_base.pose.position.y = cup_guess_.y;
      cup_in_base.pose.position.z = cup_guess_.z;
      // Tool Z pointing down: 180° about Y → quaternion (0,1,0,0)
      cup_in_base.pose.orientation.x = 0.0;
      cup_in_base.pose.orientation.y = 1.0;
      cup_in_base.pose.orientation.z = 0.0;
      cup_in_base.pose.orientation.w = 0.0;
    }

    RCLCPP_INFO(
        LOGGER,
        "Cup pose in '%s': [%.3f, %.3f, %.3f] (using orientation Z-down)",
        target_frame_.c_str(), cup_in_base.pose.position.x,
        cup_in_base.pose.position.y, cup_in_base.pose.position.z);

    // === 2) Compute waypoints ===
    // Hover directly above the cup (adjust Y slightly if you need offset)
    geometry_msgs::msg::Pose hover = cup_in_base.pose;
    hover.position.z += HOVER_ABOVE_;
    applyToolZDown(hover);

    // Grasp pose: descend vertically to just above/at the cup
    geometry_msgs::msg::Pose grasp = hover;
    grasp.position.z = cup_in_base.pose.position.z + GRASP_CLEARANCE_;
    applyToolZDown(grasp);

    // Lift pose: go back up after closing
    geometry_msgs::msg::Pose lift = grasp;
    lift.position.z = hover.position.z;
    applyToolZDown(lift);

    // === 3) Execute sequence: hover → descend → close → lift ===
    if (!goPoseArm(hover, "HoverAboveCup"))
      return;
    if (!cartesianTo(grasp, "DescendToGrasp"))
      return;

    if (!goNamedGripper("close")) {
      RCLCPP_WARN(LOGGER, "Failed to close gripper.");
    }

    if (!cartesianTo(lift, "LiftObject"))
      return;

    RCLCPP_INFO(LOGGER, "Pick sequence complete.");
  }

private:
  using Plan = moveit::planning_interface::MoveGroupInterface::Plan;

  // Tunables
  static constexpr double HOVER_ABOVE_ = 0.15;     // meters above cup
  static constexpr double GRASP_CLEARANCE_ = 0.01; // meters above surface
  static constexpr double EEF_STEP_ = 0.01;
  static constexpr double JUMP_THR_ = 0.0;

  struct CupGuess {
    double x{0.218};
    double y{0.33};
    double z{0.035};
  } cup_guess_;

  // Nodes/executor
  rclcpp::Node::SharedPtr base_node_;
  rclcpp::Node::SharedPtr move_group_node_;
  rclcpp::executors::SingleThreadedExecutor exec_;

  // MoveIt
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> gripper_;

  // TF2
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::string target_frame_; // planning frame

  // Subscriptions
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_pose_;

  // Data
  bool have_pose_{false};
  geometry_msgs::msg::PoseStamped
      cup_pose_base_; // transformed to planning frame

  // ── Callbacks ────────────────────────────────────────────────────────────
  void poseCb(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    if (have_pose_)
      return;

    const auto &src_frame = msg->header.frame_id;
    if (!tf_buffer_->canTransform(target_frame_, src_frame, msg->header.stamp,
                                  rclcpp::Duration::from_seconds(0.2))) {
      RCLCPP_WARN(LOGGER, "No TF %s -> %s yet; skipping this sample",
                  src_frame.c_str(), target_frame_.c_str());
      return;
    }
    try {
      auto tf_msg = tf_buffer_->transform(*msg, target_frame_);
      cup_pose_base_ = tf_msg;
      // Force tool Z-down orientation here; you can also keep vision yaw if
      // needed.
      applyToolZDown(cup_pose_base_.pose);
      have_pose_ = true;
      RCLCPP_INFO(LOGGER, "Transformed /cup_pose %s -> %s", src_frame.c_str(),
                  target_frame_.c_str());
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(LOGGER, "TF transform failed: %s", ex.what());
    }
  }

  // ── Helpers ──────────────────────────────────────────────────────────────
  static void applyToolZDown(geometry_msgs::msg::Pose &p) {
    p.orientation.x = 0.0;
    p.orientation.y = 1.0; // 180° about Y
    p.orientation.z = 0.0;
    p.orientation.w = 0.0;
  }

  bool waitForCupPose(geometry_msgs::msg::PoseStamped &out, double timeout_s) {
    rclcpp::Time start = base_node_->now();
    rclcpp::Rate r(100);
    while (rclcpp::ok()) {
      if (have_pose_) {
        out = cup_pose_base_;
        return true;
      }
      if ((base_node_->now() - start).seconds() > timeout_s)
        break;
      rclcpp::spin_some(base_node_);
      r.sleep();
    }
    return false;
  }

  bool goNamedArm(const std::string &name) {
    arm_->setNamedTarget(name);
    return planAndExecArm(("ArmNamed:" + name).c_str());
  }

  bool goNamedGripper(const std::string &name) {
    gripper_->setNamedTarget(name);
    return planAndExecGripper(("GripperNamed:" + name).c_str());
  }

  bool goPoseArm(const geometry_msgs::msg::Pose &pose, const char *tag) {
    arm_->setPoseTarget(pose);
    return planAndExecArm(tag);
  }

  bool cartesianTo(const geometry_msgs::msg::Pose &target, const char *tag) {
    std::vector<geometry_msgs::msg::Pose> wps;
    wps.push_back(arm_->getCurrentPose().pose);
    wps.push_back(target);

    moveit_msgs::msg::RobotTrajectory traj;
    double fraction =
        arm_->computeCartesianPath(wps, EEF_STEP_, JUMP_THR_, traj);
    if (fraction <= 0.0) {
      RCLCPP_WARN(LOGGER, "[%s] cartesian plan failed (fraction=%.2f)", tag,
                  fraction);
      return false;
    }
    auto ret = arm_->execute(traj);
    bool ok = (ret == moveit::core::MoveItErrorCode::SUCCESS);
    RCLCPP_INFO(LOGGER, "[%s] cartesian execute %s (fraction=%.2f)", tag,
                ok ? "success" : "failed", fraction);
    return ok;
  }

  bool planAndExecArm(const char *tag) {
    Plan plan;
    auto ok = (arm_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    if (!ok) {
      RCLCPP_WARN(LOGGER, "[%s] arm plan failed", tag);
      return false;
    }
    auto ret = arm_->execute(plan);
    ok = (ret == moveit::core::MoveItErrorCode::SUCCESS);
    RCLCPP_INFO(LOGGER, "[%s] arm execute %s", tag, ok ? "success" : "failed");
    return ok;
  }

  bool planAndExecGripper(const char *tag) {
    Plan plan;
    auto ok = (gripper_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    if (!ok) {
      RCLCPP_WARN(LOGGER, "[%s] gripper plan failed", tag);
      return false;
    }
    auto ret = gripper_->execute(plan);
    ok = (ret == moveit::core::MoveItErrorCode::SUCCESS);
    RCLCPP_INFO(LOGGER, "[%s] gripper execute %s", tag,
                ok ? "success" : "failed");
    return ok;
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
