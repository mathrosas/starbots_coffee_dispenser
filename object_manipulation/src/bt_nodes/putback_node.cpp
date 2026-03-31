#include <bt_nodes/putback_node.hpp>

PutBackNode::PutBackNode(const std::string &name,
                         const BT::NodeConfig &config,
                         object_manipulation::BtApi *api)
    : BT::SyncActionNode(name, config), api_(api) {}

BT::PortsList PutBackNode::providedPorts() { return {}; }

BT::NodeStatus PutBackNode::tick() { return api_->bt_put_back(); }
