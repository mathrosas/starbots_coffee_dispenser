#include "object_manipulation/motion_executor.hpp"

#include "object_manipulation/scoped_constraints.hpp"
#include "object_manipulation/watch_guards.hpp"

#include <moveit_msgs/msg/orientation_constraint.hpp>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <thread>
#include <utility>

namespace object_manipulation {

MotionExecutor::MotionExecutor(std::shared_ptr<MoveGroupInterface> arm,
                               std::shared_ptr<MoveGroupInterface> gripper,
                               const JointModelGroup *arm_joint_model_group,
                               const rclcpp::Logger &logger)
    : arm_(std::move(arm)), gripper_(std::move(gripper)),
      arm_joint_model_group_(arm_joint_model_group), logger_(logger) {}

void MotionExecutor::set_motion_policies(MotionPolicyMap policies) {
  policies_ = std::move(policies);
}

void MotionExecutor::set_watch_guards_enabled(bool enabled) {
  watch_guards_enabled_ = enabled;
}

void MotionExecutor::set_joint_state_timeout_sec(double timeout_sec) {
  joint_state_timeout_sec_ = timeout_sec;
}

void MotionExecutor::set_start_state_max_joint_delta(double max_joint_delta) {
  start_state_max_joint_delta_ = max_joint_delta;
}

void MotionExecutor::set_settle_delay_ms(int settle_delay_ms) {
  settle_delay_ms_ = (settle_delay_ms < 0) ? 0 : settle_delay_ms;
}

moveit_msgs::msg::Constraints MotionExecutor::make_gripper_down_constraints(
    const std::string &end_effector_link, const std::string &reference_frame,
    double abs_x_tolerance, double abs_y_tolerance, double abs_z_tolerance,
    double weight) const {
  moveit_msgs::msg::OrientationConstraint ocm;
  ocm.link_name = end_effector_link;
  ocm.header.frame_id = reference_frame;
  // Equivalent to RPY(pi, 0, 0): keep gripper pointing down.
  ocm.orientation.x = -1.0;
  ocm.orientation.y = 0.0;
  ocm.orientation.z = 0.0;
  ocm.orientation.w = 0.0;
  ocm.absolute_x_axis_tolerance = abs_x_tolerance;
  ocm.absolute_y_axis_tolerance = abs_y_tolerance;
  ocm.absolute_z_axis_tolerance = abs_z_tolerance;
  ocm.weight = weight;

  moveit_msgs::msg::Constraints constraints;
  constraints.orientation_constraints.push_back(ocm);
  return constraints;
}

bool MotionExecutor::execute_pose_goal(
    MotionStage stage, const geometry_msgs::msg::Pose &pose,
    const std::string &pose_reference_frame,
    const std::optional<moveit_msgs::msg::Constraints> &constraints,
    std::string &reason, std::size_t *trajectory_points) {
  if (!arm_) {
    reason = "Arm move group interface is null";
    return false;
  }

  if (!pose_reference_frame.empty()) {
    arm_->setPoseReferenceFrame(pose_reference_frame);
  }

  arm_->setStartStateToCurrentState();
  arm_->setPoseTarget(pose);
  const bool ok =
      plan_execute_arm_with_active_target(stage, constraints, reason,
                                          trajectory_points);
  arm_->clearPoseTargets();
  return ok;
}

bool MotionExecutor::execute_joint_goal(MotionStage stage,
                                        const std::vector<double> &joints,
                                        std::string &reason,
                                        std::size_t *trajectory_points) {
  if (!arm_) {
    reason = "Arm move group interface is null";
    return false;
  }

  arm_->setStartStateToCurrentState();
  arm_->setJointValueTarget(joints);
  const bool ok =
      plan_execute_arm_with_active_target(stage, std::nullopt, reason,
                                          trajectory_points);
  arm_->clearPoseTargets();
  return ok;
}

bool MotionExecutor::execute_linear_delta(
    MotionStage stage, double x_delta, double y_delta, double z_delta,
    const std::optional<moveit_msgs::msg::Constraints> &constraints,
    std::string &reason, std::size_t *trajectory_points) {
  if (!arm_) {
    reason = "Arm move group interface is null";
    return false;
  }

  const auto planning_frame = arm_->getPlanningFrame();
  arm_->setPoseReferenceFrame(planning_frame);

  auto target_pose = arm_->getCurrentPose().pose;
  target_pose.position.x += x_delta;
  target_pose.position.y += y_delta;
  target_pose.position.z += z_delta;

  arm_->setStartStateToCurrentState();
  arm_->setPoseTarget(target_pose);

  const bool ok =
      plan_execute_arm_with_active_target(stage, constraints, reason,
                                          trajectory_points);
  arm_->clearPoseTargets();
  return ok;
}

bool MotionExecutor::execute_gripper_named(const std::string &named_target,
                                           std::string &reason) {
  if (!gripper_) {
    reason = "Gripper move group interface is null";
    return false;
  }

  gripper_->setStartStateToCurrentState();
  gripper_->setNamedTarget(named_target);

  const auto plan_code = gripper_->plan(gripper_plan_);
  if (plan_code != moveit::core::MoveItErrorCode::SUCCESS) {
    std::ostringstream ss;
    ss << "Gripper planning failed for named target '" << named_target << "'";
    reason = ss.str();
    RCLCPP_ERROR(logger_, "%s", reason.c_str());
    return false;
  }

  const auto execute_code = gripper_->execute(gripper_plan_);
  if (execute_code != moveit::core::MoveItErrorCode::SUCCESS) {
    std::ostringstream ss;
    ss << "Gripper execution failed for named target '" << named_target
       << "'";
    reason = ss.str();
    RCLCPP_ERROR(logger_, "%s", reason.c_str());
    return false;
  }

  maybe_settle_delay();
  return true;
}

geometry_msgs::msg::Pose MotionExecutor::current_pose() const {
  if (!arm_) {
    return geometry_msgs::msg::Pose();
  }
  return arm_->getCurrentPose().pose;
}

void MotionExecutor::log_state(const std::string &tag) const {
  if (!arm_) {
    RCLCPP_WARN(logger_, "[STATE:%s] Arm move group interface is null",
                tag.c_str());
    return;
  }

  const auto ee = arm_->getCurrentPose().pose;
  auto state = arm_->getCurrentState(2.0);
  if (!state || !arm_joint_model_group_) {
    RCLCPP_WARN(logger_,
                "[STATE:%s] Unable to read current robot state for joints.",
                tag.c_str());
    return;
  }

  std::vector<double> joints;
  state->copyJointGroupPositions(arm_joint_model_group_, joints);

  std::ostringstream joint_ss;
  joint_ss << std::fixed << std::setprecision(4);
  for (std::size_t i = 0; i < joints.size(); ++i) {
    if (i > 0) {
      joint_ss << ", ";
    }
    joint_ss << "j" << (i + 1) << "=" << joints[i];
  }

  RCLCPP_INFO(logger_,
              "[STATE:%s] ee=(%.3f, %.3f, %.3f) quat=(%.3f, %.3f, %.3f, %.3f)"
              " joints=[%s]",
              tag.c_str(), ee.position.x, ee.position.y, ee.position.z,
              ee.orientation.x, ee.orientation.y, ee.orientation.z,
              ee.orientation.w, joint_ss.str().c_str());
}

const MotionPolicy &MotionExecutor::policy_for(MotionStage stage) const {
  const auto it = policies_.find(stage);
  if (it != policies_.end()) {
    return it->second;
  }

  static const MotionPolicy fallback_policy{};
  return fallback_policy;
}

bool MotionExecutor::plan_execute_arm_with_active_target(
    MotionStage stage,
    const std::optional<moveit_msgs::msg::Constraints> &constraints,
    std::string &reason, std::size_t *trajectory_points) {
  if (!arm_) {
    reason = "Arm move group interface is null";
    return false;
  }

  const auto &policy = policy_for(stage);
  apply_arm_policy(policy);

  if (policy.require_orientation_constraint && !constraints.has_value()) {
    std::ostringstream ss;
    ss << "Stage '" << to_string(stage)
       << "' requires orientation constraints, but none were provided";
    reason = ss.str();
    RCLCPP_ERROR(logger_, "%s", reason.c_str());
    return false;
  }

  if (watch_guards_enabled_) {
    if (!WatchGuards::ensure_joint_state_available(arm_, logger_,
                                                   joint_state_timeout_sec_,
                                                   reason)) {
      return false;
    }
  }

  arm_->setStartStateToCurrentState();

  ScopedPathConstraintsGuard constraint_guard(arm_, constraints);

  if (watch_guards_enabled_) {
    if (!WatchGuards::ensure_constraints_state(
            arm_, constraints.has_value(), logger_, reason)) {
      return false;
    }
  }

  auto plan_code = arm_->plan(arm_plan_);
  if (plan_code != moveit::core::MoveItErrorCode::SUCCESS) {
    std::ostringstream ss;
    ss << "Planning failed for stage '" << to_string(stage) << "'";
    reason = ss.str();
    RCLCPP_ERROR(logger_, "%s", reason.c_str());
    return false;
  }

  if (watch_guards_enabled_) {
    if (!WatchGuards::ensure_min_trajectory_points(
            arm_plan_, policy.min_trajectory_points, logger_, reason)) {
      return false;
    }

    if (!WatchGuards::ensure_start_state_matches_plan(
            arm_, arm_plan_, start_state_max_joint_delta_, logger_, reason)) {
      if (!policy.replan_on_start_state_drift) {
        return false;
      }

      RCLCPP_WARN(logger_,
                  "WatchGuard: replanning stage '%s' due to start-state drift",
                  to_string(stage));
      arm_->setStartStateToCurrentState();
      plan_code = arm_->plan(arm_plan_);
      if (plan_code != moveit::core::MoveItErrorCode::SUCCESS) {
        std::ostringstream ss;
        ss << "Replan failed for stage '" << to_string(stage) << "'";
        reason = ss.str();
        RCLCPP_ERROR(logger_, "%s", reason.c_str());
        return false;
      }

      if (!WatchGuards::ensure_min_trajectory_points(
              arm_plan_, policy.min_trajectory_points, logger_, reason)) {
        return false;
      }

      if (!WatchGuards::ensure_start_state_matches_plan(
              arm_, arm_plan_, start_state_max_joint_delta_, logger_, reason)) {
        return false;
      }
    }
  }

  if (trajectory_points) {
    *trajectory_points = count_points(arm_plan_);
  }

  return execute_arm_plan_with_watch_guards(stage, arm_plan_, reason);
}

bool MotionExecutor::execute_arm_plan_with_watch_guards(MotionStage stage,
                                                        const Plan &plan,
                                                        std::string &reason) {
  if (watch_guards_enabled_) {
    if (!WatchGuards::ensure_start_state_matches_plan(
            arm_, plan, start_state_max_joint_delta_, logger_, reason)) {
      return false;
    }
  }

  log_state(std::string("pre_execute_") + to_string(stage));
  const auto execute_code = arm_->execute(plan);
  if (execute_code != moveit::core::MoveItErrorCode::SUCCESS) {
    std::ostringstream ss;
    ss << "Execution failed for stage '" << to_string(stage) << "'";
    reason = ss.str();
    RCLCPP_ERROR(logger_, "%s", reason.c_str());
    return false;
  }

  log_state(std::string("post_execute_") + to_string(stage));
  maybe_settle_delay();
  return true;
}

void MotionExecutor::apply_arm_policy(const MotionPolicy &policy) {
  if (!arm_) {
    return;
  }

  arm_->setPlanningPipelineId(policy.pipeline_id);
  arm_->setPlannerId(policy.planner_id);
  arm_->setPlanningTime(policy.planning_time_sec);
  arm_->setNumPlanningAttempts(policy.num_planning_attempts);
  arm_->setMaxVelocityScalingFactor(policy.velocity_scaling);
  arm_->setMaxAccelerationScalingFactor(policy.acceleration_scaling);
}

void MotionExecutor::maybe_settle_delay() const {
  if (settle_delay_ms_ <= 0) {
    return;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(settle_delay_ms_));
}

std::size_t MotionExecutor::count_points(const Plan &plan) {
  return plan.trajectory_.joint_trajectory.points.size();
}

} // namespace object_manipulation
