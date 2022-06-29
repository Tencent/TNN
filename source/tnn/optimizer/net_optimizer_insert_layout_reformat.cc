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

#include "tnn/optimizer/net_optimizer_insert_layout_reformat.h"

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
#include "tnn/utils/string_utils_inner.h"

namespace TNN_NS {

namespace optimizer {

    // Plast priority: reformat after all fuse and data type reformat
    NetOptimizerRegister<NetOptimizerInsertLayoutReformat> g_net_optimizer_insert_layout_reformat(OptPriority::PLAST);
    static const std::string reformat_name_suffix(DataFormat layout) {
        return std::string("_") + ToString(layout) + "_layout_reformat";
    }

    std::string NetOptimizerInsertLayoutReformat::Strategy() {
        return kNetOptimizerInsertLayoutReformat;
    }

    bool NetOptimizerInsertLayoutReformat::IsSupported(const NetworkConfig &net_config) {
        // save net_config
        net_config_    = &net_config;
        auto device    = net_config.device_type;
        auto precision = net_config.precision;
        device_        = GetDevice(device);
        // possible adapter devices
        static DeviceType adapter_device_list[2] = {DEVICE_ARM, DEVICE_X86};
        auto adaptor_device                      = device;
        for (const auto &dev : adapter_device_list) {
            if (GetDevice(dev)) {
                adaptor_device = dev;
                break;
            }
        }
        adaptor_device_ = GetDevice(adaptor_device);

        return device == DEVICE_ARM || device == DEVICE_OPENCL || device == DEVICE_METAL;
    }

    static std::shared_ptr<LayerInfo> CreateReformat(std::string name, DataFormat src_fmt, DataFormat dst_fmt) {
        std::shared_ptr<LayerInfo> new_layer = std::shared_ptr<LayerInfo>(new LayerInfo());
        new_layer->type                      = LAYER_REFORMAT;
        new_layer->type_str                  = "Reformat";
        new_layer->name                      = name;
        ReformatLayerParam *param            = new ReformatLayerParam();
        new_layer->param                     = std::shared_ptr<LayerParam>(param);
        new_layer->param->type               = new_layer->type_str;
        new_layer->param->name               = new_layer->name;
        param->src_format                    = src_fmt;
        param->dst_format                    = dst_fmt;
        return new_layer;
    }

    // only support all inputs with the same layout now
    static DataFormat GetInputLayout(const NetworkConfig *config, const DeviceType &type) {
        if (config != nullptr && config->data_format != DATA_FORMAT_AUTO)
            return config->data_format;
        if (type == DEVICE_ARM || type == DEVICE_METAL) {
            return DATA_FORMAT_NC4HW4;
        } else if (type == DEVICE_OPENCL) {
            return DATA_FORMAT_NHC4W4;
        } else {
            return DATA_FORMAT_AUTO;
        }
    }

    static std::shared_ptr<const ImplementedLayout> GetAdaptorLayouts(const DeviceType &type) {
        auto res = std::make_shared<ImplementedLayout>();
        if (type == DEVICE_METAL) {
            res->layouts.push_back(DATA_FORMAT_NC4HW4);
        } else if (type == DEVICE_OPENCL) {
            res->layouts.push_back(DATA_FORMAT_NHC4W4);
        }
        return res;
    }

    static bool NeedDoReformat(DataFormat src_fmt, std::shared_ptr<const ImplementedLayout> dst_fmts,
                               const std::map<std::string, DataFormat> &layer_choosed_layout,
                               const std::string &layer_name) {
        // if layer's layout is already choosed
        if (layer_choosed_layout.find(layer_name) != layer_choosed_layout.end()) {
            if (src_fmt == layer_choosed_layout.at(layer_name)) {
                return false;
            } else {
                return true;
            }
        }
        for (const auto &dst_fmt : dst_fmts->layouts) {
            if (dst_fmt == src_fmt) {
                return false;
            }
        }
        return true;
    }

    // metal and opencl may use adaptor layer to fall back computing on arm
    std::shared_ptr<const ImplementedLayout> NetOptimizerInsertLayoutReformat::GetLayoutsByLayerType(LayerType type) {
        auto device_layouts = device_->GetImplementedLayout(type);
        if (!device_layouts || device_layouts->layouts.size() < 1) {
            auto adaptor_device_layouts = adaptor_device_->GetImplementedLayout(type);
            if (!adaptor_device_layouts || adaptor_device_layouts->layouts.size() < 1) {
                LOGE("NetOptimizerInsertLayoutReformat Error: empty adaptor device layouts of %d\n", type);
                return std::make_shared<ImplementedLayout>();
            } else {
                return GetAdaptorLayouts(device_->GetDeviceType());
            }
        } else {
            return device_layouts;
        }
    }

    Status NetOptimizerInsertLayoutReformat::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_orig = structure->layers;
        const int count                                     = (const int)layers_orig.size();

        // skip if network is quantized
        auto is_quantized_net = GetQuantizedInfoFromNetStructure(structure);
        if (is_quantized_net) {
            return TNN_OK;
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_modified;
        layer_choosed_layout.clear();

        const auto &constant_layers = resource->constant_layers;
        const auto &constant_blobs  = resource->constant_map;
        // reformat input layers if needed.
        // support multi inputs/outputs.
        // support multi layouts
        for (const auto &iter : structure->inputs_shape_map) {
            const auto &model_input = iter.first;
            LOGD("NetOptimizerInsertLayoutReformat::Optimize, process model input: %s\n", model_input.c_str());
            if (constant_blobs.count(model_input) > 0)
                continue;
            std::vector<DataFormat> reformat_layouts;
            DataFormat input_layout = GetInputLayout(net_config_, device_->GetDeviceType());
            for (const auto &cur_layer : layers_orig) {
                if (constant_layers.count(cur_layer->name) > 0) {
                    continue;
                }
                for (const auto &layer_input : cur_layer->inputs) {
                    if (layer_input == model_input) {
                        // get implemented layouts
                        auto implemented_layouts = GetLayoutsByLayerType(cur_layer->type);
                        if (!implemented_layouts || implemented_layouts->layouts.size() < 1) {
                            LOGE("NetOptimizerInsertLayoutReformat Error: empty implemented_layouts of layer %d\n",
                                 cur_layer->type);
                            return Status(TNNERR_LAYER_ERR,
                                          "NetOptimizerInsertLayoutReformat Error: empty implemented_layouts");
                        }
                        // If the already choosed layout is different from input_layout or input_layout is not
                        // implemented, reformat is needed.
                        if (NeedDoReformat(input_layout, implemented_layouts, layer_choosed_layout, cur_layer->name)) {
                            // use the choosed layout, or the first implemented layout
                            auto reformat_layout                  = layer_choosed_layout.count(cur_layer->name)
                                                                        ? layer_choosed_layout[cur_layer->name]
                                                                        : implemented_layouts->layouts[0];
                            layer_choosed_layout[cur_layer->name] = reformat_layout;
                            if (std::find(reformat_layouts.begin(), reformat_layouts.end(), reformat_layout) ==
                                reformat_layouts.end()) {
                                reformat_layouts.push_back(reformat_layout);
                            }
                        } else {
                            layer_choosed_layout[cur_layer->name] = input_layout;
                        }
                        break;
                    }
                }
            }
            for (const auto &reformat_layout : reformat_layouts) {
                std::vector<std::string> reformat_outs = {model_input};
                // create input_layout -> implemented_layout reformat layer
                std::shared_ptr<LayerInfo> new_layer =
                    CreateReformat(model_input + reformat_name_suffix(reformat_layout) + "__from_model_input__",
                                   input_layout, reformat_layout);

                RETURN_ON_NEQ(AdjustLayer(layers_orig, structure, constant_layers, input_layout, reformat_layout,
                                          new_layer, reformat_outs, reformat_name_suffix(reformat_layout), -1, count),
                              TNN_OK);

                LOGD("Insert layout refomat layer : src %s dst %s\n", new_layer->inputs[0].c_str(),
                     new_layer->outputs[0].c_str());
                layers_modified.push_back(new_layer);
            }
        }

        for (int index = 0; index < count; index++) {
            auto cur_layer = layers_orig[index];
            layers_modified.push_back(cur_layer);
            if (constant_layers.count(cur_layer->name) > 0) {
                continue;
            }
            if (layer_choosed_layout.find(cur_layer->name) == layer_choosed_layout.end()) {
                LOGE("NetOptimizerInsertLayoutReformat Error: layout of cur layer not choosen, index: %d, layer: %s\n",
                     index, cur_layer->name.c_str());
                return Status(TNNERR_LAYER_ERR,
                              "NetOptimizerInsertLayoutReformat Error: layout of cur layer not choosen");
            }
            auto cur_layer_layout = layer_choosed_layout[cur_layer->name];

            // find blobs need reformat
            // support multi inputs/outputs
            // support multi layouts
            std::map<DataFormat, std::vector<std::string>> layout_reformat_outs;
            for (auto cur_out : cur_layer->outputs) {
                if (constant_blobs.count(cur_out) > 0) {
                    continue;
                }
                for (int next_id = index + 1; next_id < count; next_id++) {
                    auto next_layer = layers_orig[next_id];
                    if (constant_layers.count(next_layer->name) > 0) {
                        continue;
                    }
                    auto next_layer_layouts = GetLayoutsByLayerType(next_layer->type);
                    if (!next_layer_layouts || next_layer_layouts->layouts.size() < 1) {
                        LOGE("NetOptimizerInsertLayoutReformat Error: empty implemented_layouts of layer %d\n",
                             next_layer->type);
                        return Status(TNNERR_LAYER_ERR,
                                      "NetOptimizerInsertLayoutReformat Error: empty implemented_layouts");
                    }
                    for (auto next_in : next_layer->inputs) {
                        if (next_in == cur_out) {
                            if (NeedDoReformat(cur_layer_layout, next_layer_layouts, layer_choosed_layout,
                                               next_layer->name)) {
                                // use the choosed layout, or the first implemented layout
                                auto reformat_layout                   = layer_choosed_layout.count(next_layer->name)
                                                                             ? layer_choosed_layout[next_layer->name]
                                                                             : next_layer_layouts->layouts[0];
                                layer_choosed_layout[next_layer->name] = reformat_layout;
                                auto &reformat_outs                    = layout_reformat_outs[reformat_layout];
                                if (std::find(reformat_outs.begin(), reformat_outs.end(), cur_out) ==
                                    reformat_outs.end()) {
                                    reformat_outs.push_back(cur_out);
                                }
                            } else {
                                layer_choosed_layout[next_layer->name] = cur_layer_layout;
                            }
                            break;
                        }
                    }
                }
            }

            for (const auto &iter : layout_reformat_outs) {
                auto reformat_layout = iter.first;
                auto reformat_outs   = iter.second;

                std::shared_ptr<LayerInfo> new_layer = CreateReformat(
                    cur_layer->name + reformat_name_suffix(reformat_layout), cur_layer_layout, reformat_layout);

                RETURN_ON_NEQ(
                    AdjustLayer(layers_orig, structure, constant_layers, cur_layer_layout, reformat_layout, new_layer,
                                reformat_outs, reformat_name_suffix(reformat_layout), index, count),
                    TNN_OK);

                LOGD("Insert layout refomat layer : src %s dst %s\n", new_layer->inputs[0].c_str(),
                     new_layer->outputs[0].c_str());
                layers_modified.push_back(new_layer);
            }
        }

        // reformat output layers if needed.
        // support multi outputs.
        // for (const auto &model_output : structure->outputs) {
        //     LOGD("NetOptimizerInsertLayoutReformat::Optimize, process model output: %s\n", model_output.c_str());
        //     bool need_reformat                      = false;
        //     std::shared_ptr<LayerInfo> output_layer = layers_orig[0];
        //     DataFormat output_layout                = GetInputLayout(net_config_, device_->GetDeviceType());
        //     for (const auto &layer : layers_orig) {
        //         for (const auto &output : layer->outputs) {
        //             if (output == model_output) {
        //                 if (layer_choosed_layout.find(layer->name) == layer_choosed_layout.end()) {
        //                     LOGE("NetOptimizerInsertLayoutReformat Error: layout of layer %s not choosen\n",
        //                          layer->name.c_str());
        //                     return Status(TNNERR_LAYER_ERR,
        //                                   "NetOptimizerInsertLayoutReformat Error: layout of layer not choosen");
        //                 }
        //                 if (layer_choosed_layout[layer->name] != output_layout) {
        //                     need_reformat = true;
        //                     output_layer  = layer;
        //                     break;
        //                 }
        //             }
        //         }
        //         if (need_reformat) {
        //             break;
        //         }
        //     }

        //     if (need_reformat) {
        //         auto choosed_layout = layer_choosed_layout[output_layer->name];
        //         // create choosed_layout -> output_layout reformat layer
        //         std::shared_ptr<LayerInfo> new_layer =
        //             CreateReformat(model_output + reformat_name_suffix(choosed_layout) + "__to_model_output__",
        //                            choosed_layout, output_layout);

        //         auto new_blob = model_output + reformat_name_suffix(choosed_layout);
        //         structure->blobs.insert(new_blob);
        //         output_layer->outputs = {new_blob};
        //         new_layer->inputs     = {new_blob};
        //         new_layer->outputs    = {model_output};

        //         LOGD("Insert layout refomat layer : src %s dst %s\n", new_layer->inputs[0].c_str(),
        //              new_layer->outputs[0].c_str());
        //         layers_modified.push_back(new_layer);
        //     }
        // }

        structure->layers = layers_modified;
        layer_choosed_layout.clear();

        return TNN_OK;
    }

    Status NetOptimizerInsertLayoutReformat::AdjustLayer(
        std::vector<std::shared_ptr<LayerInfo>> &layers_orig, NetStructure *structure,
        const std::set<std::string> &constant_layers, DataFormat cur_layer_layout, DataFormat reformat_layout,
        std::shared_ptr<LayerInfo> &new_layer, std::vector<std::string> &reformat_outs,
        const std::string &reformat_name_suffix, const int index, const int count) {
        // change blobs for layers to read blob data correctly
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
                auto next_layer_layouts = GetLayoutsByLayerType(next_layer->type);
                for (auto &next_in : next_layer->inputs) {
                    // only use reformat out when cur_layer_layout not supported
                    if (next_in == cur_out &&
                        NeedDoReformat(cur_layer_layout, next_layer_layouts, layer_choosed_layout, next_layer->name)) {
                        if (layer_choosed_layout.find(next_layer->name) == layer_choosed_layout.end()) {
                            LOGE("NetOptimizerInsertLayoutReformat Error: layout of next layer not choosen\n");
                            return Status(TNNERR_LAYER_ERR,
                                          "NetOptimizerInsertLayoutReformat Error: layout of next layer not choosen");
                        }
                        if (layer_choosed_layout[next_layer->name] == reformat_layout)
                            next_in = new_out;
                    }
                }
            }
        }
        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS
