#pragma once

#include <behaviortree_cpp/bt_factory.h>

namespace object_manipulation {

class BtApi {
public:
  virtual ~BtApi() = default;

  virtual BT::NodeStatus bt_goal_not_canceled() = 0;
  virtual BT::NodeStatus bt_validate_detection() = 0;
  virtual BT::NodeStatus bt_pre_pick() = 0;
  virtual BT::NodeStatus bt_pick() = 0;
  virtual BT::NodeStatus bt_pre_place() = 0;
  virtual BT::NodeStatus bt_place() = 0;
  virtual BT::NodeStatus bt_put_back() = 0;
  virtual BT::NodeStatus bt_return() = 0;
};

} // namespace object_manipulation
