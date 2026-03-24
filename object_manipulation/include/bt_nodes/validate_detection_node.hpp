#pragma once

#include <behaviortree_cpp/bt_factory.h>
#include <object_manipulation/bt_api.hpp>

class ValidateDetectionNode : public BT::SyncActionNode {
public:
  ValidateDetectionNode(const std::string &name,
                        const BT::NodeConfig &config,
                        object_manipulation::BtApi *api);

  static BT::PortsList providedPorts();
  BT::NodeStatus tick() override;

private:
  object_manipulation::BtApi *api_;
};
