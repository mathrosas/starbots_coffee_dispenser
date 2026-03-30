#pragma once

#include <behaviortree_cpp/bt_factory.h>
#include <object_manipulation/bt_api.hpp>

class PrePlaceNode : public BT::SyncActionNode {
public:
  PrePlaceNode(const std::string &name,
               const BT::NodeConfig &config,
               object_manipulation::BtApi *api);

  static BT::PortsList providedPorts();
  BT::NodeStatus tick() override;

private:
  object_manipulation::BtApi *api_;
};
