#include "tnn_optimize.h"

namespace TNN_NS {
void RemoveClone(NetStructure* net_structure, NetResource* net_resource) {
    auto& layers      = net_structure->layers;
    auto remove_clone = [&layers](const std::shared_ptr<LayerInfo>& layer_info) -> bool {
        if (layer_info->type != LAYER_CLONE) {
            return false;
        }

        const auto clone_input  = layer_info->inputs[0];
        const auto clone_output = layer_info->outputs[0];
        for (auto& layer : layers) {
//            for (auto& input : layer->inputs) {
//                if (input == clone_output) {
//                    input = clone_input;
//                }
//            }
                for (int i = 0; i < layer->inputs.size(); i++) {
                    if (layer->inputs[i] == clone_output) {
                        layer->inputs[i] = clone_input;
                    }
                }
        }

        return true;

    };

   layers.erase(std::remove_if(layers.begin(), layers.end(), remove_clone), layers.end());
}

void RemoveSingleConcat(NetStructure* net_structure, NetResource* net_resource) {
    auto& layers              = net_structure->layers;
    auto remove_single_concat = [&layers](const std::shared_ptr<LayerInfo> layer_info) -> bool {
        if (layer_info->type != LAYER_CONCAT || layer_info->inputs.size() != 1) {
            return false;
        }

        return true;
    };

    layers.erase(std::remove_if(layers.begin(), layers.end(), remove_single_concat), layers.end());
}

void TNNOptPass(NetStructure* net_structure, NetResource* net_resource) {
    RemoveClone(net_structure, net_resource);
    // RemoveSingleConcat(net_structure, net_resource);
}
}  // namespace TNN_NS
