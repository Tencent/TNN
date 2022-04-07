#include "tnn_optimize.h"
#include <memory>

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

void RemoveNormClampminExpandasDiv(NetStructure* net_structure, NetResource* net_resource) {
    auto& layers              = net_structure->layers;
    std::vector<std::shared_ptr<LayerInfo>> layers_fused;

    // Normalize <= Norm - Clampmin - Expandas - Div
    for (auto iter = layers.begin(); iter != layers.end(); iter++) {
        auto& norm_layer = *iter;
        if (norm_layer->type != TNN_NS::LAYER_NORM){
            layers_fused.push_back(norm_layer);
            continue;
        }
        
        auto clampmin_iter = iter + 1;
        auto expandas_iter = iter + 2;
        auto div_iter      = iter + 3;
        auto clampmin_layer = *clampmin_iter;
        auto expandas_layer = *expandas_iter;
        auto div_layer      = *div_iter;
        
        if (clampmin_layer->type != TNN_NS::LAYER_CLAMPMIN || expandas_layer->type != TNN_NS::LAYER_EXPANDAS || div_layer->type != TNN_NS::LAYER_DIV ||
            expandas_layer->outputs[0] != div_layer->inputs[1]) {
            continue;
        }

        auto norm_param         = std::dynamic_pointer_cast<NormLayerParam>(norm_layer->param);
        auto clampmin_param     = std::dynamic_pointer_cast<ClampminLayerParam>(clampmin_layer->param);
        const auto dim           = norm_param->dim;
        const auto p             = norm_param->p;
        const auto min           = clampmin_param->min;

        auto normalize_param     = new TNN_NS::NormalizeLayerParam;
        normalize_param->epsilon = min;
        normalize_param->axis    = dim;
        normalize_param->p       = p;
        norm_layer->param        = std::shared_ptr<TNN_NS::LayerParam>(normalize_param);
        norm_layer->type         = TNN_NS::LAYER_NORMALIZE;
        norm_layer->type_str     = "Normalize";

        norm_layer->outputs.clear();
        norm_layer->outputs = div_layer->outputs;
        layers_fused.push_back(norm_layer);
        iter = iter + 3;
    }
    net_structure->layers = layers_fused;
}

void TNNOptPass(NetStructure* net_structure, NetResource* net_resource) {
    RemoveClone(net_structure, net_resource);
    // RemoveSingleConcat(net_structure, net_resource);
    RemoveNormClampminExpandasDiv(net_structure, net_resource);
}
}  // namespace TNN_NS
