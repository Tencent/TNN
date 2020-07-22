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

#include "tnn/optimizer/net_optimizer_insert_reformat.h"

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
    NetOptimizerRegister<NetOptimizerInsertReformat> g_net_optimizer_Insert_reformat(OptPriority::PLAST);
    static const std::string reformat_name_suffix                  = "_reformat";
    static std::map<LayerType, ActivationType> kLayerActivationMap = {{LAYER_RELU, ActivationType_ReLU},
                                                                    {LAYER_RELU6, ActivationType_ReLU6}};

    std::string NetOptimizerInsertReformat::Strategy() {
        return kNetOptimizerInsertReformat;
    }

    bool NetOptimizerInsertReformat::SupportDevice(DeviceType device) {
        return device == DEVICE_ARM || device == DEVICE_NAIVE;
    }

    std::shared_ptr<LayerInfo> CreateReformat(std::string name, bool src_quantized) {
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

    Status NetOptimizerInsertReformat::Optimize(NetStructure *structure, NetResource *resource) {
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
        auto quantize_layer = std::find_if(layers_orig.begin(), layers_orig.end(), [](std::shared_ptr<LayerInfo> iter) {
            return iter->param->quantized == true;
        });
        if (quantize_layer == layers_orig.end()) {
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

            LOGD("Insert refomat layer: src %s dst %s\n", new_layer->inputs[0].c_str(), new_layer->outputs[0].c_str());
            layers_fused.push_back(new_layer);
        }
        structure->layers = layers_fused;

        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS
