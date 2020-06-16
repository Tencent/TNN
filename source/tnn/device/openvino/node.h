#ifndef TNN_OPENVINO_NODE_MANAGER_H
#define TNN_OPENVINO_NODE_MANAGER_H

#include <map>
#include <memory>
#include <string>
#include <thread>

#include <ngraph/node.hpp>
#include "tnn/core/status.h"

namespace TNN_NS {

class NodeManager {
public:
    Status InitNodeManager();

    Status addNode(std::string node_name, std::shared_ptr<ngraph::Node> node);
    
    std::shared_ptr<ngraph::Node> findNode(std::string node_name);

private:
    std::map<std::string, std::shared_ptr<ngraph::Node>> nodes_; 
};
}

#endif