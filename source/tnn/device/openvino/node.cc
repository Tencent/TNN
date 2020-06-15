#include "node.h"

namespace TNN_NS {

Status NodeManager::InitNodeManager() { return TNN_OK; };

Status NodeManager::addNode(std::string node_name, std::shared_ptr<ngraph::Node> node) {
    nodes_.insert({node_name, node});
    return TNN_OK;
};

std::shared_ptr<ngraph::Node> NodeManager::getLastNode() {
    std::cout << nodes_.size() << std::endl;
    return nodes_.end()->second;
}

std::shared_ptr<ngraph::Node> NodeManager::findNode(std::string node_name) {
    std::map<std::string, std::shared_ptr<ngraph::Node>>::iterator iter = nodes_.find(node_name);
    return iter->second;
};
}