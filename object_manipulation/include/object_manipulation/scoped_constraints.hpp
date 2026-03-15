#pragma once

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/constraints.hpp>

#include <memory>
#include <optional>

namespace object_manipulation {

class ScopedPathConstraintsGuard {
public:
  using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;

  ScopedPathConstraintsGuard(
      std::shared_ptr<MoveGroupInterface> move_group,
      const std::optional<moveit_msgs::msg::Constraints> &desired_constraints)
      : move_group_(std::move(move_group)),
        previous_constraints_(move_group_ ? move_group_->getPathConstraints()
                                          : moveit_msgs::msg::Constraints()) {
    if (!move_group_) {
      return;
    }

    if (desired_constraints.has_value()) {
      move_group_->setPathConstraints(*desired_constraints);
    } else {
      move_group_->clearPathConstraints();
    }
  }

  ScopedPathConstraintsGuard(const ScopedPathConstraintsGuard &) = delete;
  ScopedPathConstraintsGuard &
  operator=(const ScopedPathConstraintsGuard &) = delete;

  ~ScopedPathConstraintsGuard() {
    if (!move_group_) {
      return;
    }

    if (has_any_constraints(previous_constraints_)) {
      move_group_->setPathConstraints(previous_constraints_);
    } else {
      move_group_->clearPathConstraints();
    }
  }

private:
  static bool has_any_constraints(const moveit_msgs::msg::Constraints &c) {
    return !c.name.empty() || !c.joint_constraints.empty() ||
           !c.position_constraints.empty() ||
           !c.orientation_constraints.empty() ||
           !c.visibility_constraints.empty();
  }

  std::shared_ptr<MoveGroupInterface> move_group_;
  moveit_msgs::msg::Constraints previous_constraints_;
};

} // namespace object_manipulation

