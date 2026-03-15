#pragma once

#include <moveit/move_group_interface/move_group_interface.h>

#include <rclcpp/rclcpp.hpp>

#include <cstddef>
#include <string>

namespace object_manipulation {

class WatchGuards {
public:
  using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;
  using Plan = MoveGroupInterface::Plan;

  static bool ensure_joint_state_available(
      const std::shared_ptr<MoveGroupInterface> &move_group,
      const rclcpp::Logger &logger, double timeout_sec, std::string &reason);

  static std::size_t trajectory_points(const Plan &plan);

  static bool ensure_min_trajectory_points(const Plan &plan,
                                           std::size_t min_points,
                                           const rclcpp::Logger &logger,
                                           std::string &reason);

  static bool ensure_constraints_state(
      const std::shared_ptr<MoveGroupInterface> &move_group,
      bool expected_constrained, const rclcpp::Logger &logger,
      std::string &reason);

  static bool ensure_start_state_matches_plan(
      const std::shared_ptr<MoveGroupInterface> &move_group, const Plan &plan,
      double max_joint_delta, const rclcpp::Logger &logger,
      std::string &reason);
};

} // namespace object_manipulation

