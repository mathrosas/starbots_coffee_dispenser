#pragma once

#include "object_manipulation/motion_policy.hpp"

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/constraints.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace object_manipulation {

class MotionExecutor {
public:
  using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;
  using Plan = MoveGroupInterface::Plan;
  using JointModelGroup = moveit::core::JointModelGroup;

  MotionExecutor(std::shared_ptr<MoveGroupInterface> arm,
                 std::shared_ptr<MoveGroupInterface> gripper,
                 const JointModelGroup *arm_joint_model_group,
                 const rclcpp::Logger &logger);

  void set_motion_policies(MotionPolicyMap policies);
  void set_watch_guards_enabled(bool enabled);
  void set_joint_state_timeout_sec(double timeout_sec);
  void set_start_state_max_joint_delta(double max_joint_delta);
  void set_settle_delay_ms(int settle_delay_ms);

  moveit_msgs::msg::Constraints
  make_gripper_down_constraints(const std::string &end_effector_link,
                                const std::string &reference_frame,
                                double abs_x_tolerance,
                                double abs_y_tolerance,
                                double abs_z_tolerance, double weight) const;

  bool execute_pose_goal(MotionStage stage, const geometry_msgs::msg::Pose &pose,
                         const std::string &pose_reference_frame,
                         const std::optional<moveit_msgs::msg::Constraints>
                             &constraints,
                         std::string &reason, std::size_t *trajectory_points);

  bool execute_joint_goal(MotionStage stage, const std::vector<double> &joints,
                          std::string &reason,
                          std::size_t *trajectory_points);

  bool execute_linear_delta(
      MotionStage stage, double x_delta, double y_delta, double z_delta,
      const std::optional<moveit_msgs::msg::Constraints> &constraints,
      std::string &reason, std::size_t *trajectory_points);

  bool execute_gripper_named(const std::string &named_target, std::string &reason);

  geometry_msgs::msg::Pose current_pose() const;

  void log_state(const std::string &tag) const;

private:
  const MotionPolicy &policy_for(MotionStage stage) const;
  bool plan_execute_arm_with_active_target(
      MotionStage stage,
      const std::optional<moveit_msgs::msg::Constraints> &constraints,
      std::string &reason, std::size_t *trajectory_points);
  bool execute_arm_plan_with_watch_guards(MotionStage stage, const Plan &plan,
                                          std::string &reason);
  void apply_arm_policy(const MotionPolicy &policy);
  void maybe_settle_delay() const;
  static std::size_t count_points(const Plan &plan);

  std::shared_ptr<MoveGroupInterface> arm_;
  std::shared_ptr<MoveGroupInterface> gripper_;
  const JointModelGroup *arm_joint_model_group_{nullptr};
  rclcpp::Logger logger_;

  MotionPolicyMap policies_;

  bool watch_guards_enabled_{true};
  double joint_state_timeout_sec_{2.0};
  double start_state_max_joint_delta_{0.01};
  int settle_delay_ms_{0};

  Plan arm_plan_;
  Plan gripper_plan_;
};

} // namespace object_manipulation
