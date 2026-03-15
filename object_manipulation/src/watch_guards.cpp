#include "object_manipulation/watch_guards.hpp"

#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/msg/constraints.hpp>

#include <cmath>
#include <sstream>

namespace object_manipulation {

namespace {
bool has_any_constraints(const moveit_msgs::msg::Constraints &constraints) {
  return !constraints.name.empty() || !constraints.joint_constraints.empty() ||
         !constraints.position_constraints.empty() ||
         !constraints.orientation_constraints.empty() ||
         !constraints.visibility_constraints.empty();
}
} // namespace

bool WatchGuards::ensure_joint_state_available(
    const std::shared_ptr<MoveGroupInterface> &move_group,
    const rclcpp::Logger &logger, double timeout_sec, std::string &reason) {
  if (!move_group) {
    reason = "MoveGroupInterface is null";
    RCLCPP_ERROR(logger, "%s", reason.c_str());
    return false;
  }

  auto state = move_group->getCurrentState(timeout_sec);
  if (!state) {
    std::ostringstream ss;
    ss << "WatchGuard failed: no current joint state within " << timeout_sec
       << " seconds";
    reason = ss.str();
    RCLCPP_ERROR(logger, "%s", reason.c_str());
    return false;
  }

  return true;
}

std::size_t WatchGuards::trajectory_points(const Plan &plan) {
  return plan.trajectory_.joint_trajectory.points.size();
}

bool WatchGuards::ensure_min_trajectory_points(const Plan &plan,
                                               std::size_t min_points,
                                               const rclcpp::Logger &logger,
                                               std::string &reason) {
  const auto points = trajectory_points(plan);
  if (points < min_points) {
    std::ostringstream ss;
    ss << "WatchGuard failed: trajectory has " << points
       << " points, requires at least " << min_points;
    reason = ss.str();
    RCLCPP_ERROR(logger, "%s", reason.c_str());
    return false;
  }
  return true;
}

bool WatchGuards::ensure_constraints_state(
    const std::shared_ptr<MoveGroupInterface> &move_group,
    bool expected_constrained, const rclcpp::Logger &logger,
    std::string &reason) {
  if (!move_group) {
    reason = "MoveGroupInterface is null";
    RCLCPP_ERROR(logger, "%s", reason.c_str());
    return false;
  }

  const bool constrained = has_any_constraints(move_group->getPathConstraints());
  if (constrained != expected_constrained) {
    std::ostringstream ss;
    ss << "WatchGuard failed: constraints state mismatch (expected "
       << (expected_constrained ? "constrained" : "unconstrained")
       << ", got " << (constrained ? "constrained" : "unconstrained")
       << ")";
    reason = ss.str();
    RCLCPP_ERROR(logger, "%s", reason.c_str());
    return false;
  }

  return true;
}

bool WatchGuards::ensure_start_state_matches_plan(
    const std::shared_ptr<MoveGroupInterface> &move_group, const Plan &plan,
    double max_joint_delta, const rclcpp::Logger &logger,
    std::string &reason) {
  if (!move_group) {
    reason = "MoveGroupInterface is null";
    RCLCPP_ERROR(logger, "%s", reason.c_str());
    return false;
  }

  const auto &traj = plan.trajectory_.joint_trajectory;
  if (traj.points.empty()) {
    reason = "WatchGuard failed: trajectory has no points";
    RCLCPP_ERROR(logger, "%s", reason.c_str());
    return false;
  }

  if (traj.joint_names.empty()) {
    reason = "WatchGuard failed: trajectory has no joint names";
    RCLCPP_ERROR(logger, "%s", reason.c_str());
    return false;
  }

  const auto &start = traj.points.front();
  if (start.positions.size() != traj.joint_names.size()) {
    reason = "WatchGuard failed: trajectory start positions size mismatch";
    RCLCPP_ERROR(logger, "%s", reason.c_str());
    return false;
  }

  auto state = move_group->getCurrentState(1.0);
  if (!state) {
    reason = "WatchGuard failed: no current state before execute";
    RCLCPP_ERROR(logger, "%s", reason.c_str());
    return false;
  }

  for (std::size_t i = 0; i < traj.joint_names.size(); ++i) {
    const double current = state->getVariablePosition(traj.joint_names[i]);
    const double expected = start.positions[i];
    const double delta = std::fabs(current - expected);
    if (delta > max_joint_delta) {
      std::ostringstream ss;
      ss << "WatchGuard failed: start-state drift on joint '"
         << traj.joint_names[i] << "' (expected=" << expected
         << ", current=" << current << ", delta=" << delta
         << ", limit=" << max_joint_delta << ")";
      reason = ss.str();
      RCLCPP_ERROR(logger, "%s", reason.c_str());
      return false;
    }
  }

  return true;
}

} // namespace object_manipulation
