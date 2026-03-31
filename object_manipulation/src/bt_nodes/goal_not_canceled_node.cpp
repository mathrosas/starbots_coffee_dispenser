#include <bt_nodes/goal_not_canceled_node.hpp>

GoalNotCanceledNode::GoalNotCanceledNode(const std::string &name,
                                         const BT::NodeConfig &config,
                                         object_manipulation::BtApi *api)
    : BT::ConditionNode(name, config), api_(api) {}

BT::PortsList GoalNotCanceledNode::providedPorts() { return {}; }

BT::NodeStatus GoalNotCanceledNode::tick() { return api_->bt_goal_not_canceled(); }
