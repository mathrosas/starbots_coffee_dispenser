#include <bt_nodes/return_node.hpp>

ReturnNode::ReturnNode(const std::string &name,
                       const BT::NodeConfig &config,
                       object_manipulation::BtApi *api)
    : BT::SyncActionNode(name, config), api_(api) {}

BT::PortsList ReturnNode::providedPorts() { return {}; }

BT::NodeStatus ReturnNode::tick() { return api_->bt_return(); }
