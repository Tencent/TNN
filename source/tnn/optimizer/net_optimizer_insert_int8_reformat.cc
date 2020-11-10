// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "tnn/optimizer/net_optimizer_insert_int8_reformat.h"

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#include "tnn/core/layer_type.h"
#include "tnn/core/macro.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"

namespace TNN_NS {

namespace optimizer {

    // Plast priority: reformat after all fuse
    NetOptimizerRegister<NetOptimizerInsertInt8Reformat> g_net_optimizer_insert_int8_reformat(OptPriority::PLAST);
    static const std::string reformat_name_suffix = "_int8_reformat";

    std::string NetOptimizerInsertInt8Reformat::Strategy() {
        return kNetOptimizerInsertInt8Reformat;
    }

    bool NetOptimizerInsertInt8Reformat::IsSupported(const NetworkConfig &net_config) {
        auto device = net_config.device_type;
        return device == DEVICE_ARM || device == DEVICE_NAIVE;
    }

    static std::shared_ptr<LayerInfo> CreateReformat(std::string name, bool src_quantized) {
        std::shared_ptr<LayerInfo> new_layer = std::shared_ptr<LayerInfo>(new LayerInfo());
        new_layer->type                      = LAYER_REFORMAT;
        new_layer->type_str                  = "Reformat";
        new_layer->name                      = name;
        ReformatLayerParam *param            = new ReformatLayerParam();
        new_layer->param                     = std::shared_ptr<LayerParam>(param);
        // only define quant/dequant here, layout after layer init
        param->src_type = src_quantized ? DATA_TYPE_INT8 : DATA_TYPE_FLOAT;
        param->dst_type = src_quantized ? DATA_TYPE_FLOAT : DATA_TYPE_INT8;
        return new_layer;
    }

    Status NetOptimizerInsertInt8Reformat::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_orig = structure->layers;
        const int count                                     = (const int)layers_orig.size();
        if (count <= 1) {
            return TNN_OK;
        }

        // only insert reformat before quantized layer now
        auto is_quantized_net = GetQuantizedInfoFromNetStructure(structure);
        if (!is_quantized_net) {
            return TNN_OK;
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_fused;

        for (int index = 0; index < count; index++) {
            auto cur_layer = layers_orig[index];
            layers_fused.push_back(cur_layer);

            // find blobs need reformat
            // support multi inputs/outputs
            // only quant & dequant now
            std::vector<std::string> reformat_outs;
            for (auto cur_out : cur_layer->outputs) {
                bool need_reformat = false;
                for (int next_id = index + 1; next_id < count; next_id++) {
                    auto next_layer = layers_orig[next_id];
                    for (auto next_in : next_layer->inputs) {
                        if (next_in == cur_out && next_layer->param->quantized != cur_layer->param->quantized) {
                            need_reformat = true;
                        }
                    }
                }
                if (need_reformat)
                    reformat_outs.push_back(cur_out);
            }
            if (!reformat_outs.size()) {
                continue;
            }

            std::shared_ptr<LayerInfo> new_layer =
                CreateReformat(cur_layer->name + reformat_name_suffix, cur_layer->param->quantized);

            AdjustLayer(layers_orig, structure, cur_layer, new_layer,
                        reformat_outs, reformat_name_suffix, index, count);

            LOGD("Insert int8 refomat layer: src %s dst %s\n", new_layer->inputs[0].c_str(), new_layer->outputs[0].c_str());
            layers_fused.push_back(new_layer);
        }
        structure->layers = layers_fused;

        return TNN_OK;
    }

    void NetOptimizerInsertInt8Reformat::AdjustLayer(
            std::vector<std::shared_ptr<LayerInfo>>& layers_orig,
            NetStructure *structure,
            std::shared_ptr<LayerInfo>& cur_layer,
            std::shared_ptr<LayerInfo>& new_layer,
            std::vector<std::string>& reformat_outs,
            const std::string& reformat_name_suffix,
            const int index,
            const int count) {
        // change blobs for unquantized layer for layers to read
        // int8resource correctly
        // src_type int8, change dst blob
        if (cur_layer->param->quantized) {
            new_layer->inputs = reformat_outs;
            for (auto cur_out : reformat_outs) {
                auto new_out = cur_out + reformat_name_suffix;
                new_layer->outputs.push_back(new_out);
                structure->blobs.insert(new_out);
                // change the inputs of successed int8 layers
                for (int next_id = index + 1; next_id < count; next_id++) {
                    auto next_layer = layers_orig[next_id];
                    for (auto &next_in : next_layer->inputs) {
                        // only use reformat out when quantized param diff
                        if (next_in == cur_out && next_layer->param->quantized != cur_layer->param->quantized) {
                            next_in = new_out;
                        }
                    }
                }
            }
        } else {
            // dst type int8, change src blob
            new_layer->outputs = reformat_outs;
            for (auto cur_out : reformat_outs) {
                auto new_out = cur_out + reformat_name_suffix;
                new_layer->inputs.push_back(new_out);
                structure->blobs.insert(new_out);
                for (auto &cur_layer_out : cur_layer->outputs) {
                    cur_layer_out = new_out;
                }
                // change the inputs of successed float layers
                for (int next_id = index + 1; next_id < count; next_id++) {
                    auto next_layer = layers_orig[next_id];
                    for (auto &next_in : next_layer->inputs) {
                        if (next_in == cur_out && !next_layer->param->quantized) {
                            next_in = new_out;
                        }
                    }
                }
            }
        }
    }

}  // namespace optimizer

}  // namespace TNN_NS
