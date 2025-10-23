// object_manipulation.cpp
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

#include <chrono>
#include <functional>
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
    RCLCPP_INFO(LOGGER, "Initializing Object Manipulation (fixed cup pose)");

    rclcpp::NodeOptions opts;
    opts.automatically_declare_parameters_from_overrides(true);
    move_group_node_ =
        rclcpp::Node::make_shared("object_manipulation_node", opts);
    exec_.add_node(move_group_node_);
    std::thread([this]() { exec_.spin(); }).detach();

    arm_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        move_group_node_, PLANNING_GROUP_ROBOT);
    gripper_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        move_group_node_, PLANNING_GROUP_GRIPPER);

    // Our cup pose is defined in base_link → make pose targets use that frame.
    arm_->setPoseReferenceFrame("base_link");
    target_frame_ = arm_->getPoseReferenceFrame();

    // Planner tuning
    arm_->setPlanningTime(5.0);
    arm_->setNumPlanningAttempts(10);
    arm_->setGoalPositionTolerance(0.004);
    arm_->setGoalOrientationTolerance(0.02);
    arm_->setMaxVelocityScalingFactor(0.3);
    arm_->setMaxAccelerationScalingFactor(0.3);

    RCLCPP_INFO(LOGGER, "Planning frame: %s", arm_->getPlanningFrame().c_str());
    RCLCPP_INFO(LOGGER, "Pose ref frame: %s",
                arm_->getPoseReferenceFrame().c_str());
    RCLCPP_INFO(LOGGER, "EE link:        %s",
                arm_->getEndEffectorLink().c_str());

    // Initialize start states
    arm_->setStartStateToCurrentState();
    gripper_->setStartStateToCurrentState();
  }

  void runOnce() {
    // 0) Go to 'home' and open gripper
    if (!goNamedArm("home"))
      RCLCPP_WARN(LOGGER, "Failed to go to 'home', continuing.");
    else
      std::this_thread::sleep_for(std::chrono::milliseconds(1500));

    if (!goNamedGripper("open"))
      RCLCPP_WARN(LOGGER, "Failed to open gripper, continuing.");
    else
      std::this_thread::sleep_for(std::chrono::milliseconds(1200));

    // 1) Build poses from fixed cup position (in base_link)
    geometry_msgs::msg::Pose cup_pose{};
    cup_pose.position.x = 0.298;
    cup_pose.position.y = 0.330;
    cup_pose.position.z = 0.035;
    applyToolZDown(cup_pose);

    geometry_msgs::msg::Pose hover = cup_pose;
    hover.position.z += HOVER_ABOVE_;
    applyToolZDown(hover);

    geometry_msgs::msg::Pose grasp = hover;
    grasp.position.z = cup_pose.position.z + GRASP_CLEARANCE_;
    applyToolZDown(grasp);

    geometry_msgs::msg::Pose lift = grasp;
    lift.position.z = hover.position.z;
    applyToolZDown(lift);

    RCLCPP_INFO(LOGGER,
                "Executing pick at fixed pose in '%s': [%.3f, %.3f, %.3f]",
                target_frame_.c_str(), cup_pose.position.x, cup_pose.position.y,
                cup_pose.position.z);

    // 2) Execute: hover → descend → close → lift (with pauses)
    if (!goPoseArm(hover, "HoverAboveCup"))
      return;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    if (!cartesianTo(grasp, "DescendToGrasp"))
      return;
    std::this_thread::sleep_for(std::chrono::milliseconds(800));

    if (!goNamedGripper("close"))
      RCLCPP_WARN(LOGGER, "Failed to close gripper.");
    else
      std::this_thread::sleep_for(std::chrono::milliseconds(800));

    if (!cartesianTo(lift, "LiftObject"))
      return;
    std::this_thread::sleep_for(std::chrono::milliseconds(800));

    RCLCPP_INFO(LOGGER, "Pick sequence complete (fixed pose).");
  }

private:
  using Plan = moveit::planning_interface::MoveGroupInterface::Plan;

  // Tunables (adjusted)
  static constexpr double HOVER_ABOVE_ = 0.30; // reduced to shorten descent
  static constexpr double GRASP_CLEARANCE_ = 0.015; // a bit safer above surface
  static constexpr double EEF_STEP_ = 0.005;        // finer interpolation
  static constexpr double JUMP_THR_ = 0.0;

  // Nodes/executor
  rclcpp::Node::SharedPtr base_node_;
  rclcpp::Node::SharedPtr move_group_node_;
  rclcpp::executors::SingleThreadedExecutor exec_;

  // MoveIt
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> gripper_;
  std::string target_frame_;

  // ── Helpers ──────────────────────────────────────────────────────────────
  static void applyToolZDown(geometry_msgs::msg::Pose &p) {
    // 180° about Y → end-effector Z axis points down
    p.orientation.x = 0.0;
    p.orientation.y = 1.0;
    p.orientation.z = 0.0;
    p.orientation.w = 0.0;
  }

  // Ensure we have a fresh start state; clear previous goals to avoid
  // contamination
  bool syncStartState(const char *tag, double timeout_s = 1.0) {
    arm_->stop();
    arm_->clearPoseTargets();

    // MoveIt2 API: getCurrentState(wait_seconds). Returns nullptr if it times
    // out.
    moveit::core::RobotStatePtr st = arm_->getCurrentState(timeout_s);
    if (!st) {
      RCLCPP_WARN(LOGGER, "[%s] current state not fresh within %.2fs", tag,
                  timeout_s);
      arm_->setStartStateToCurrentState(); // fallback
    } else {
      arm_->setStartState(*st); // explicit, freshest snapshot
    }
    return true;
  }

  bool goNamedArm(const std::string &name) {
    syncStartState(("goNamedArm:" + name).c_str());
    arm_->setNamedTarget(name);
    return planAndExecArm(("ArmNamed:" + name).c_str());
  }

  bool goNamedGripper(const std::string &name) {
    gripper_->stop();
    gripper_->clearPoseTargets();
    gripper_->setStartStateToCurrentState();
    gripper_->setNamedTarget(name);
    return planAndExecGripper(("GripperNamed:" + name).c_str());
  }

  bool goPoseArm(const geometry_msgs::msg::Pose &pose, const char *tag) {
    syncStartState(tag);
    arm_->setPoseTarget(pose);
    return planAndExecArm(tag);
  }

  bool cartesianTo(const geometry_msgs::msg::Pose &target, const char *tag) {
    // Build a straight 2-waypoint Z move from the *current* pose
    std::vector<geometry_msgs::msg::Pose> wps;
    wps.push_back(arm_->getCurrentPose().pose);
    wps.push_back(target);

    moveit_msgs::msg::RobotTrajectory traj;
    syncStartState(tag);

    // (1) Try with collision checking enabled
    double fraction = arm_->computeCartesianPath(
        wps, EEF_STEP_, JUMP_THR_,
        traj /* avoid_collisions = true by default */);

    if (fraction < 0.99) {
      RCLCPP_WARN(LOGGER,
                  "[%s] cartesian fraction=%.2f with collisions; retry "
                  "avoid_collisions=false for short Z move",
                  tag, fraction);
      // (2) Short, well-understood Z move: safe fallback to
      // avoid_collisions=false
      fraction = arm_->computeCartesianPath(wps, EEF_STEP_, JUMP_THR_, traj,
                                            /*avoid_collisions=*/false);
    }

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
    auto planned = arm_->plan(plan);
    if (planned != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(LOGGER, "[%s] arm plan failed", tag);
      return false;
    }
    auto ret = arm_->execute(plan);
    bool ok = (ret == moveit::core::MoveItErrorCode::SUCCESS);
    RCLCPP_INFO(LOGGER, "[%s] arm execute %s", tag, ok ? "success" : "failed");
    return ok;
  }

  bool planAndExecGripper(const char *tag) {
    Plan plan;
    auto planned = gripper_->plan(plan);
    if (planned != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(LOGGER, "[%s] gripper plan failed", tag);
      return false;
    }
    auto ret = gripper_->execute(plan);
    bool ok = (ret == moveit::core::MoveItErrorCode::SUCCESS);
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
