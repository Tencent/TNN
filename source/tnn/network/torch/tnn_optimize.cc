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
    

    // Normalize <= Norm - Clampmin - Expandas - Div
    for (auto iter = layers.begin(); iter + 3 != layers.end(); iter++) {
        auto& norm_layer = *iter;
        if (norm_layer->type != TNN_NS::LAYER_NORM || norm_layer->outputs.size() != 1){
            continue;
        }
        if (norm_layer->type == TNN_NS::LAYER_NORM || norm_layer->outputs.size() == 1){
            // LOGE("norm_layer type %s \n", norm_layer->type_str.c_str());
        }
        auto clampmin_iter = iter + 1;
        auto expandas_iter = iter + 2;
        auto div_iter      = iter + 3;
        auto clampmin_layer = *clampmin_iter;
        auto expandas_layer = *expandas_iter;
        auto div_layer      = *div_iter;

        if (norm_layer->type == TNN_NS::LAYER_NORM || norm_layer->outputs.size() == 1){
            // LOGE("clampmin_layer type is %s \n", clampmin_layer->type_str.c_str());
            // LOGE("expandas_layer type is %s \n", expandas_layer->type_str.c_str());
            // LOGE("div_layer type is %s \n", div_layer->type_str.c_str());
        }

        if (clampmin_layer->type != TNN_NS::LAYER_CLAMPMIN || expandas_layer->type != TNN_NS::LAYER_EXPANDAS || clampmin_layer->outputs.size() != 1){
            continue;
        }
        if (expandas_layer->type != TNN_NS::LAYER_EXPANDAS || div_layer->type != TNN_NS::LAYER_DIV || expandas_layer->outputs.size() != 1){
            continue;
        }
        if (norm_layer->outputs[0] != clampmin_layer->inputs[0] ||
            clampmin_layer->outputs[0] != expandas_layer->inputs[0] ||
            expandas_layer->outputs[0] != div_layer->inputs[1]) {
            LOGE("judgement false");
            continue;
        }

        auto norm_param         = std::dynamic_pointer_cast<NormLayerParam>(norm_layer->param);
        // LOGE("---------if norm layer param = %d -------\n", norm_layer->param == nullptr);
        auto clampmin_param     = std::dynamic_pointer_cast<ClampminLayerParam>(clampmin_layer->param);
        // LOGE("----------if norm_param = %d -------\n", norm_param == nullptr);
        // auto* expandas_param     = dynamic_cast<TNN_NS::ExpandasLayerParam*>(expandas_layer->param.get());
        // auto* div_param          = dynamic_cast<TNN_NS::DivLayerParam*>(div_layer->param.get());
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

        
        // LOGE("-------div output is %s -----\n", div_layer->outputs[0]);
        // LOGE("-------norm output is %s -----\n", norm_layer->outputs[0]);
        norm_layer->outputs.clear();
        norm_layer->outputs = div_layer->outputs;
        // printf("-------------print norm layer output----------");
        // for (int i = 0; i < norm_layer->outputs.size(); i = i+1) {
        //     printf(norm_layer->outputs[i].c_str());
        // }
        // printf("-------------print div layer output----------");
        // for (int i = 0; i < div_layer->outputs.size(); i = i+1) {
        //     printf(div_layer->outputs[i].c_str());
        // }
        layers.erase(clampmin_iter);
        printf("-------------------erase clampmin-----------------");
        expandas_iter -= 1;
        layers.erase(expandas_iter);
        div_iter -= 1;
        LOGE("div_layer type is %s \n", div_layer->type_str.c_str());
        LOGE("norm_layer type %s \n", norm_layer->type_str.c_str());
        LOGE("clampmin_layer type is %s \n", clampmin_layer->type_str.c_str());
        LOGE("expandas_layer type is %s \n", expandas_layer->type_str.c_str());
        layers.erase(div_iter);
    }
}

void TNNOptPass(NetStructure* net_structure, NetResource* net_resource) {
    RemoveClone(net_structure, net_resource);
    // RemoveSingleConcat(net_structure, net_resource);
    // LOGE("------------------ fuse normalization ----------");
    RemoveNormClampminExpandasDiv(net_structure, net_resource);
}
}  // namespace TNN_NS
