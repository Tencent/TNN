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

#include "tnn/optimizer/net_optimizer_insert_fp16_reformat.h"

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#include "tnn/core/layer_type.h"
#include "tnn/core/macro.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"
#include "tnn/utils/cpu_utils.h"

namespace TNN_NS {

namespace optimizer {

    // Plast priority: reformat after all fuse
    NetOptimizerRegister<NetOptimizerInsertFp16Reformat> g_net_optimizer_insert_fp16_reformat(OptPriority::P2);
    static const std::string reformat_name_suffix         = "_fp16_reformat";
    static const std::set<LayerType> kLayerOutputNonFloat = {LAYER_ARG_MAX_OR_MIN};
    static const std::set<LayerType> kLayerOutputMaybeNonFloat = {LAYER_UNSQUEEZE, LAYER_GATHER};
    std::set<std::string> whitelist_i32;

    static const std::set<LayerType> kLayerNeedSpecialTreat = {LAYER_GATHER};

    // skip fp16 reformat if output of the layer is not float type
    bool IsLayerOutputFloat(std::shared_ptr<LayerInfo> layer) {
        if (kLayerOutputNonFloat.find(layer->type) != kLayerOutputNonFloat.end()) {
            return false;
        }

        if (layer->type == LAYER_CAST) {
            auto layer_param = dynamic_cast<CastLayerParam *>(layer->param.get());
            CHECK_PARAM_NULL(layer_param);
            return (layer_param->to == DATA_TYPE_FLOAT || layer_param->to == DATA_TYPE_HALF);
        }

        if (layer->type == LAYER_UNSQUEEZE) {
            return whitelist_i32.find(layer->name) == whitelist_i32.end();
        }

        if (layer->type == LAYER_GATHER) {
            return whitelist_i32.find(layer->name) == whitelist_i32.end();
        }

        return true;
    }

    void OutputSameAsInput(std::shared_ptr<LayerInfo> cur_layer, std::set<std::string> &whitelist_i32, std::set<std::string> &blob_i32) {
        // if input is int32, output is view as i32
        std::vector<std::string> intersection;
        std::set<std::string> cur_input(cur_layer->inputs.begin(), cur_layer->inputs.end());
        std::set_intersection(cur_input.begin(), cur_input.end(), 
                            blob_i32.begin(), blob_i32.end(), std::back_inserter(intersection));
        if (intersection.size() > 0){
            whitelist_i32.insert(cur_layer->name);
            for (auto cur_output: cur_layer->outputs){
                blob_i32.insert(cur_output);
            }
        }
    }

    bool GenWhiteList_i32(NetStructure *structure, NetResource *resource, std::set<std::string> &whitelist_i32){
        // If input is int32, some layers will propagate int32 (such as unsqueeze, gather), 
        // these layers will view as special case which don't output fp16

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

                // process Gather
                if (cur_layer->type == LAYER_GATHER){
                    auto layer_param = dynamic_cast<GatherLayerParam *>(cur_layer->param.get());
                    CHECK_PARAM_NULL(layer_param);
                    if (layer_param->data_in_resource == true) {
                        // if data in resource, must return type is decided by resource
                        auto resource_ = resource->resource_map.find(cur_layer->name)->second.get();
                        auto layer_resource = dynamic_cast<GatherLayerResource*>(resource_);
                        if (layer_resource->data.GetDataType() == DATA_TYPE_INT32){
                            whitelist_i32.insert(cur_layer->name);
                            for (auto cur_output: cur_layer->outputs){
                                blob_i32.insert(cur_output);
                            }
                        }
                    } else {
                        // else indice in reource, output type in the same as input
                        OutputSameAsInput(cur_layer, whitelist_i32, blob_i32);
                    }
                }

                // process Unsqueeze
                if (cur_layer->type == LAYER_UNSQUEEZE){
                    // output type is the same as input
                    OutputSameAsInput(cur_layer, whitelist_i32, blob_i32);
                }

                // other cases ...

            }
        }
        return true;
    }

    // skip some special cases which must output fp32 instead of fp16, such as gather (data in resource)
    bool IsSpecialCase_fp32(NetResource* resource, std::shared_ptr<LayerInfo> cur_layer){
        if (kLayerNeedSpecialTreat.find(cur_layer->type) != kLayerNeedSpecialTreat.end()) {
            
            // handle gather layer. When resource type if fp32, the layer will return fp32
            if (cur_layer->type == LAYER_GATHER) {
                auto layer_param = dynamic_cast<GatherLayerParam *>(cur_layer->param.get());
                CHECK_PARAM_NULL(layer_param);
                if (layer_param->data_in_resource == true) {
                    // if data in resource, must return type is decided by resource
                    auto resource_ = resource->resource_map.find(cur_layer->name)->second.get();
                    auto layer_resource = dynamic_cast<GatherLayerResource*>(resource_);
                    if (layer_resource->data.GetDataType() == DATA_TYPE_FLOAT){
                        return true;
                    }
                }
            }

            // other cases ...

        }

        return false;
    }

    std::string NetOptimizerInsertFp16Reformat::Strategy() {
        return kNetOptimizerInsertFp16Reformat;
    }

    bool NetOptimizerInsertFp16Reformat::IsSupported(const NetworkConfig &net_config) {
        auto device    = net_config.device_type;
        auto precision = net_config.precision;
        device_        = GetDevice(device);
        return (device == DEVICE_ARM || device == DEVICE_NAIVE) &&
               (precision == PRECISION_NORMAL || precision == PRECISION_AUTO) && CpuUtils::CpuSupportFp16();
    }

    std::shared_ptr<LayerInfo> NetOptimizerInsertFp16Reformat::CreateReformat(std::string name, bool src_fp16) {
        std::shared_ptr<LayerInfo> new_layer = std::shared_ptr<LayerInfo>(new LayerInfo());
        new_layer->type                      = LAYER_REFORMAT;
        new_layer->type_str                  = "Reformat";
        new_layer->name                      = name;
        ReformatLayerParam *param            = new ReformatLayerParam();
        new_layer->param                     = std::shared_ptr<LayerParam>(param);
        new_layer->param->type               = new_layer->type_str;
        new_layer->param->name               = new_layer->name;
        param->src_type                      = src_fp16 ? DATA_TYPE_HALF : DATA_TYPE_FLOAT;
        param->dst_type                      = src_fp16 ? DATA_TYPE_FLOAT : DATA_TYPE_HALF;
        if (device_->GetDeviceType() == DEVICE_ARM) {
            param->src_format = DATA_FORMAT_NC4HW4;
            param->dst_format = DATA_FORMAT_NC4HW4;
        }
        return new_layer;
    }

    Status NetOptimizerInsertFp16Reformat::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_orig = structure->layers;
        const int count                                     = (const int)layers_orig.size();
        if (count <= 1) {
            return TNN_OK;
        }

        // skip if network is quantized
        auto is_quantized_net = GetQuantizedInfoFromNetStructure(structure);
        if (is_quantized_net) {
            return TNN_OK;
        }

        // only insert reformat for fp16-implemented layer
        auto fp16_layer = std::find_if(layers_orig.begin(), layers_orig.end(), [&](std::shared_ptr<LayerInfo> iter) {
            return device_->GetImplementedPrecision(iter->type)->fp16_implemented;
        });
        if (fp16_layer == layers_orig.end()) {
            return TNN_OK;
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_fused;

        const auto &constant_layers = resource->constant_layers;
        const auto &constant_blobs  = resource->constant_map;
        // if model input is used for multiple layers with different data types,
        // reformat layers are inserted at beginning.
        // support multi inputs/outputs.
        for (const auto &iter : structure->inputs_shape_map) {
            const auto &model_input = iter.first;
            LOGD("NetOptimizerInsertFp16Reformat::Optimize, process model input: %s\n", model_input.c_str());
            if (constant_blobs.count(model_input) > 0) {
                continue;
            }
            int need_fp16_input = 0;
            int need_fp32_input = 0;
            for (const auto &cur_layer : layers_orig) {
                if (constant_layers.count(cur_layer->name) > 0) {
                    continue;
                }
                for (const auto &layer_input : cur_layer->inputs) {
                    if (layer_input == model_input) {
                        if (device_->GetImplementedPrecision(cur_layer->type)->fp16_implemented) {
                            ++need_fp16_input;
                        } else {
                            ++need_fp32_input;
                        }
                        break;
                    }
                }
            }
            if (need_fp16_input > 0 && need_fp32_input > 0) {
                std::vector<std::string> reformat_outs = {model_input};
                // create fp16 -> fp32 reformat layer
                std::shared_ptr<LayerInfo> new_layer =
                    CreateReformat(model_input + reformat_name_suffix + "__from_model_input__", true);

                AdjustLayer(layers_orig, structure, constant_layers, true, new_layer, reformat_outs,
                            reformat_name_suffix, -1, count);

                LOGD("Insert fp16 refomat layer : src %s dst %s\n", new_layer->inputs[0].c_str(),
                     new_layer->outputs[0].c_str());
                layers_fused.push_back(new_layer);
            }
        }

        if (!GenWhiteList_i32(structure, resource, whitelist_i32)){
            return Status(TNNERR_CONVERT_OPTIMIZE_ERROR, "Can not generate whitelist_i32");
        }

        for (int index = 0; index < count; index++) {
            auto cur_layer = layers_orig[index];
            layers_fused.push_back(cur_layer);
            if (constant_layers.count(cur_layer->name) > 0 || !IsLayerOutputFloat(cur_layer)) {
                continue;
            }
            // find blobs need reformat
            // support multi inputs/outputs
            std::vector<std::string> reformat_outs;
            bool is_cur_layer_fp16 = device_->GetImplementedPrecision(cur_layer->type)->fp16_implemented & !IsSpecialCase_fp32(resource, cur_layer);
            for (auto cur_out : cur_layer->outputs) {
                if (constant_blobs.count(cur_out) > 0) {
                    continue;
                }
                bool need_reformat = false;
                for (int next_id = index + 1; next_id < count; next_id++) {
                    auto next_layer = layers_orig[next_id];
                    if (constant_layers.count(next_layer->name) > 0) {
                        continue;
                    }
                    bool is_next_layer_fp16 = device_->GetImplementedPrecision(next_layer->type)->fp16_implemented & !IsSpecialCase_fp32(resource, next_layer);
                    for (auto next_in : next_layer->inputs) {
                        if (next_in == cur_out && is_next_layer_fp16 != is_cur_layer_fp16) {
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
                CreateReformat(cur_layer->name + reformat_name_suffix, is_cur_layer_fp16);

            AdjustLayer(layers_orig, structure, constant_layers, is_cur_layer_fp16, new_layer, reformat_outs,
                        reformat_name_suffix, index, count);

            LOGD("Insert fp16 refomat layer: src %s dst %s\n", new_layer->inputs[0].c_str(),
                 new_layer->outputs[0].c_str());
            layers_fused.push_back(new_layer);
        }
        structure->layers = layers_fused;

        return TNN_OK;
    }

    void NetOptimizerInsertFp16Reformat::AdjustLayer(std::vector<std::shared_ptr<LayerInfo>> &layers_orig,
                                                     NetStructure *structure,
                                                     const std::set<std::string> &constant_layers,
                                                     bool is_cur_layer_fp16, std::shared_ptr<LayerInfo> &new_layer,
                                                     std::vector<std::string> &reformat_outs,
                                                     const std::string &reformat_name_suffix, const int index,
                                                     const int count) {
        // change blobs for for layers to read blob data correctly
        new_layer->inputs = reformat_outs;
        for (auto cur_out : reformat_outs) {
            auto new_out = cur_out + reformat_name_suffix;
            new_layer->outputs.push_back(new_out);
            structure->blobs.insert(new_out);
            // change the inputs of successed layers
            for (int next_id = index + 1; next_id < count; next_id++) {
                auto next_layer = layers_orig[next_id];
                if (constant_layers.count(next_layer->name) > 0)
                    continue;
                bool is_next_layer_fp16 = device_->GetImplementedPrecision(next_layer->type)->fp16_implemented;
                for (auto &next_in : next_layer->inputs) {
                    // only use reformat out when fp16 status diff
                    if (next_in == cur_out && is_next_layer_fp16 != is_cur_layer_fp16) {
                        next_in = new_out;
                    }
                }
            }
        }
    }

}  // namespace optimizer

}  // namespace TNN_NS
