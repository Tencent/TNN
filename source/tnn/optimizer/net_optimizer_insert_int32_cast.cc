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

#include "tnn/optimizer/net_optimizer_insert_int32_cast.h"

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

    // Plast priority: cast when input/output is INT32
    NetOptimizerRegister<NetOptimizerInsertInt32Cast> g_net_optimizer_insert_int32_cast(OptPriority::P2);
    static const std::string float_cast_name_suffix = "_float32_cast";
    static const std::string int32_cast_name_suffix = "_int32_cast";
    static const std::string int8_cast_name_suffix = "_int8_cast";
    static const std::set<LayerType> kLayerOutputMaybeNonFloat = {LAYER_ADD, LAYER_SUB, LAYER_MUL, LAYER_DIV,
                                                                  LAYER_EQUAL, LAYER_GREATER, LAYER_SHAPE,
                                                                  LAYER_CONCAT, LAYER_UNSQUEEZE, LAYER_CAST,
                                                                  LAYER_GATHER};
    static const std::set<LayerType> kLayerCanotDealInt32 = {LAYER_ADD, LAYER_SUB, LAYER_MUL, LAYER_DIV,
                                                             LAYER_EQUAL, LAYER_GREATER, LAYER_GATHER,
                                                             LAYER_CONCAT, LAYER_UNSQUEEZE};
    std::set<std::string> layerlist_i32;
    std::set<std::string> layerlist_i8;

    std::string NetOptimizerInsertInt32Cast::Strategy() {
        return kNetOptimizerInsertInt32Cast;
    }

    bool NetOptimizerInsertInt32Cast::IsSupported(const NetworkConfig &net_config) {
        auto device = net_config.device_type;
        device_     = GetDevice(device);
        return device == DEVICE_ARM;
    }

    bool IsLayerOutputInt32(std::shared_ptr<LayerInfo> layer) {
        if (kLayerCanotDealInt32.find(layer->type) != kLayerCanotDealInt32.end()) {
            return layerlist_i32.find(layer->name) != layerlist_i32.end();
        }
        return false;
    }

    bool IsLayerOutputInt8(std::shared_ptr<LayerInfo> layer) {
        if (kLayerCanotDealInt32.find(layer->type) != kLayerCanotDealInt32.end()) {
            return layerlist_i8.find(layer->name)  != layerlist_i8.end();
        }
        return false;
    }

    void OutputSameAsInputDataType(std::shared_ptr<LayerInfo> cur_layer, std::set<std::string> &layerlist_i32, std::set<std::string> &blob_i32) {
        // if input is int32, output is view as i32
        std::vector<std::string> intersection;
        std::set<std::string> cur_input(cur_layer->inputs.begin(), cur_layer->inputs.end());
        std::set_intersection(cur_input.begin(), cur_input.end(),
                            blob_i32.begin(), blob_i32.end(), std::back_inserter(intersection));
        if (intersection.size() > 0){
            layerlist_i32.insert(cur_layer->name);
            for (auto cur_output: cur_layer->outputs){
                blob_i32.insert(cur_output);
            }
        }
    }

    bool GenLayerList_i32(NetStructure *structure, NetResource *resource, std::set<std::string> &layerlist_i32) {
        // If input is int32, some layers will propagate int32 (such as unsqueeze, gather), 
        // these layers will view as special case which don't output float

        std::set<std::string> blob_i32;

        std::vector<std::shared_ptr<LayerInfo>> layers_orig = structure->layers;
        const int count                                     = (const int)layers_orig.size();

        // get input type
        for (const auto &iter : structure->input_data_type_map) {
            const auto &name = iter.first;
            const auto type = iter.second;
            if (type == DATA_TYPE_INT32){
                blob_i32.insert(name);
            }
        }

        for (int index = 0; index < count; index++) {
            auto cur_layer = layers_orig[index];
            if (kLayerOutputMaybeNonFloat.find(cur_layer->type) != kLayerOutputMaybeNonFloat.end()){
                // process Unsqueeze & concat
                if (cur_layer->type == LAYER_UNSQUEEZE || cur_layer->type == LAYER_CONCAT) {
                    // output type is the same as input
                    OutputSameAsInputDataType(cur_layer, layerlist_i32, blob_i32);
                }
                // process gather
                else if (cur_layer->type == LAYER_GATHER) {
                    auto layer_param = dynamic_cast<GatherLayerParam *>(cur_layer->param.get());
                    CHECK_PARAM_NULL(layer_param);
                    if (layer_param->data_in_resource) {
                        auto resource_ = resource->resource_map.find(cur_layer->name)->second.get();
                        auto layer_resource = dynamic_cast<GatherLayerResource*>(resource_);
                         if (layer_resource->data.GetDataType() == DATA_TYPE_INT32){
                            layerlist_i32.insert(cur_layer->name);
                            for (auto cur_output: cur_layer->outputs){
                                blob_i32.insert(cur_output);
                            }
                        }
                    } else {
                        OutputSameAsInputDataType(cur_layer, layerlist_i32, blob_i32);
                    }
                }
                // process binary
                else if (cur_layer->type == LAYER_ADD || cur_layer->type == LAYER_SUB
                      || cur_layer->type == LAYER_MUL || cur_layer->type == LAYER_DIV) {
                    OutputSameAsInputDataType(cur_layer, layerlist_i32, blob_i32);
                }
                else if (cur_layer->type == LAYER_EQUAL || cur_layer->type == LAYER_GREATER) {
                    layerlist_i8.insert(cur_layer->name);
                }
                // process cast
                else if (cur_layer->type == LAYER_CAST) {
                    auto layer_param = dynamic_cast<CastLayerParam *>(cur_layer->param.get());
                    CHECK_PARAM_NULL(layer_param);
                    if (layer_param->to == DATA_TYPE_INT32) {
                        layerlist_i32.insert(cur_layer->name);
                        for (auto cur_output: cur_layer->outputs) {
                            blob_i32.insert(cur_output);
                        }
                    }
                    else if (layer_param->to == DATA_TYPE_INT8) {
                        layerlist_i8.insert(cur_layer->name);
                    }
                }
                // process SHAPE
                else if (cur_layer->type == LAYER_SHAPE) {
                    for (auto cur_output: cur_layer->outputs) {
                        blob_i32.insert(cur_output);
                    }
                }
                // other cases ...
            }
        }
        return true;
    }

    bool EndWith(const std::string &layer_name, const std::string &name_suffix) {
        if (layer_name.length() < name_suffix.length()) {
            return false;
        }
        int idx_x = layer_name.length() - 1;
        int idx_y = name_suffix.length() - 1;
        for (int i = 0; i < name_suffix.length(); i++) {
            if (layer_name[idx_x] != name_suffix[idx_y]) {
                return false;
            }
        }
        return true;
    }

    std::shared_ptr<LayerInfo> NetOptimizerInsertInt32Cast::CreateCast(std::string name, DataType cast_to) {
        std::shared_ptr<LayerInfo> new_layer = std::shared_ptr<LayerInfo>(new LayerInfo());
        new_layer->type                      = LAYER_CAST;
        new_layer->type_str                  = "Cast";
        new_layer->name                      = name;
        CastLayerParam *param                = new CastLayerParam();
        new_layer->param                     = std::shared_ptr<LayerParam>(param);
        new_layer->param->type               = new_layer->type_str;
        new_layer->param->name               = new_layer->name;
        param->to                            = cast_to;
        return new_layer;
    }

    Status NetOptimizerInsertInt32Cast::Optimize(NetStructure *structure, NetResource *resource) {
        //return TNN_OK;
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }
        std::vector<std::shared_ptr<LayerInfo>> layers_orig = structure->layers;
        const int count                                     = (const int)layers_orig.size();
        if (count < 1) {
            return TNN_OK;
        }
        if (!GenLayerList_i32(structure, resource, layerlist_i32)) {
            return Status(TNNERR_CONVERT_OPTIMIZE_ERROR, "Can not generate layerlist_i32");
        }
        std::vector<std::shared_ptr<LayerInfo>> layers_fused;
        const auto &constant_layers = resource->constant_layers;
        const auto &constant_blobs  = resource->constant_map;

        // 遍历所有的节点
        for (int index = 0; index < count; index++) {
            auto cur_layer = layers_orig[index];
            if (constant_layers.count(cur_layer->name) > 0 || (!IsLayerOutputInt32(cur_layer) && !IsLayerOutputInt8(cur_layer))) {
                layers_fused.push_back(cur_layer);
                continue;
            }
            // 找到该节点所有输入，插入 cast 转换操作
            std::vector<std::string> cur_inputs(cur_layer->inputs.begin(), cur_layer->inputs.end());
            for (int idx = 0; idx < cur_inputs.size(); idx++) {
                auto cur_in = cur_inputs[idx];
                if (EndWith(cur_in, float_cast_name_suffix)) {
                    // LOGD("已有前置转换，将 %s->inputs[%d] 由 %s 替换为 %s\n", cur_layer->name.c_str(), idx, cur_layer->inputs[idx].c_str(), cur_in.c_str());
                    cur_layer->inputs[idx] = cur_in;
                    continue;
                }
                std::vector<std::string> cast_ins = {cur_in};
                std::shared_ptr<LayerInfo> fake_input_layer = std::make_shared<LayerInfo>();
                fake_input_layer->param                     = std::make_shared<LayerParam>();
                std::shared_ptr<LayerInfo> new_in_layer     = CreateCast(cur_in + float_cast_name_suffix, DATA_TYPE_FLOAT);
                AdjustLayer(layers_orig, structure, fake_input_layer, new_in_layer, cast_ins, float_cast_name_suffix, -1, count);
                layers_fused.push_back(new_in_layer);
                // LOGD("插入 Cast to Float 节点: src %s dst %s\n", new_in_layer->inputs[0].c_str(), new_in_layer->outputs[0].c_str());
            }
            layers_fused.push_back(cur_layer);
            // 遍历该节点所有输出，找到需要插入 cast 的位置
            std::vector<std::string> cast_outs;
            for (auto cur_out : cur_layer->outputs) {
                cast_outs.push_back(cur_out);
            }
            // 在输出之后添加新节点
            if (IsLayerOutputInt32(cur_layer)) {
                std::shared_ptr<LayerInfo> new_out_layer = CreateCast(cur_layer->name + int32_cast_name_suffix, DATA_TYPE_INT32);
                AdjustLayer(layers_orig, structure, cur_layer, new_out_layer, cast_outs, int32_cast_name_suffix, -1, count);
                // LOGD("插入 Cast to INT32 节点: src %s dst %s\n", new_out_layer->inputs[0].c_str(), new_out_layer->outputs[0].c_str());
                layers_fused.push_back(new_out_layer);
            }
            else if (IsLayerOutputInt8(cur_layer)) {
                std::shared_ptr<LayerInfo> new_out_layer = CreateCast(cur_layer->name + int8_cast_name_suffix, DATA_TYPE_INT8);
                AdjustLayer(layers_orig, structure, cur_layer, new_out_layer, cast_outs, int8_cast_name_suffix, -1, count);
                // LOGD("插入 Cast to INT8 节点: src %s dst %s\n", new_out_layer->inputs[0].c_str(), new_out_layer->outputs[0].c_str());
                layers_fused.push_back(new_out_layer);
            }
        }
        structure->layers = layers_fused;

        return TNN_OK;
    }

    void NetOptimizerInsertInt32Cast::AdjustLayer(std::vector<std::shared_ptr<LayerInfo>> &layers_orig,
                                                     NetStructure *structure, std::shared_ptr<LayerInfo> &cur_layer,
                                                     std::shared_ptr<LayerInfo> &new_layer,
                                                     std::vector<std::string> &cast_outs,
                                                     const std::string &cast_name_suffix, const int index,
                                                     const int count) {
        new_layer->inputs = cast_outs;
        for (auto cur_out : cast_outs) {
            auto new_out = cur_out + cast_name_suffix;
            new_layer->outputs.push_back(new_out);
            structure->blobs.insert(new_out);
            for (int next_id = index + 1; next_id < count; next_id++) {
                auto next_layer = layers_orig[next_id];
                for (auto &next_in : next_layer->inputs) {
                    if (next_in == cur_out) {
                        next_in = new_out;
                        // LOGD("节点 %s 的输入由 %s 更换为 %s\n", next_layer->name.c_str(), cur_out.c_str(), new_out.c_str());
                    }
                }
            }
        }
    }

}  // namespace optimizer

}  // namespace TNN_NS
