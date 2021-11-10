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

#include "tnn/optimizer/net_optimizer_insert_loss_and_gradient.h"

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

    // P2 priority: reformat after all fuse, remove
    NetOptimizerRegister<NetOptimizerInsertLossAndGradient> g_net_optimizer_insert_loss_and_gradient(OptPriority::P2);
    static const std::string loss_name_suffix = "_tnn__loss";

    std::string NetOptimizerInsertLossAndGradient::Strategy() {
        return kNetOptimizerInsertLossAndGradient;
    }

    bool NetOptimizerInsertLossAndGradient::IsSupported(const NetworkConfig &net_config) {
        bool is_support = false;
#if TRAIN
        train_config = net_config.train_config;
        if (train_config.run_mode == TRAIN_MODE) {
            auto device = net_config.device_type;
            if (device == DEVICE_ARM || device == DEVICE_NAIVE) {
                is_support = true;
            }
        }
#endif
        return is_support;
    }

    /*
     * @brief deep vist the compute DAG graph, to find which layers need to be calcualted grads
     * @return if cur layer need to be calculated grad
     */
    static bool DeepVisit(const LayerInfo *layer, const std::set<std::string> &trainable_layers,
                          const std::map<std::string, LayerInfo *> &blob_to_layer,
                          std::set<std::string> &need_grad_layers, const InputShapesMap &inputs_shape_map) {
        bool need_grad = false;
        for (auto &input : layer->inputs) {
            if (inputs_shape_map.find(input) != inputs_shape_map.end()) {
                // need_grad |= false;
                continue;
            }
            auto iter = blob_to_layer.find(input);
            if (iter == blob_to_layer.end()) {
                LOGE("cann't find the layer of the blob");
                continue;
            }
            // one node may be repeatedly visited
            need_grad |= DeepVisit(iter->second, trainable_layers, blob_to_layer, need_grad_layers, inputs_shape_map);
        }
        if (trainable_layers.find(layer->name) != trainable_layers.end())
            need_grad |= true;
        if (need_grad)
            need_grad_layers.insert(layer->name);
        return need_grad;
    }

    static void BuildLayer(const std::string type_str, std::shared_ptr<LayerInfo> &layer,
                           const std::shared_ptr<LayerInfo> &last_layer, std::set<std::string> &blobs,
                           LayerParam *param, const std::string layer_name = "") {
        LOGD("Optimize, build: %s\n", type_str.c_str());
        layer->type     = GlobalConvertLayerType(type_str);
        layer->type_str = type_str;
        layer->inputs.clear();
        layer->outputs.clear();
        layer->name = layer_name != "" ? layer_name : last_layer->name + "/" + type_str;
        layer->inputs.push_back(last_layer->outputs[0]);
        layer->outputs.push_back(layer->name);  // use layer name as output blob name
        blobs.insert(last_layer->outputs[0]);
        blobs.insert(layer->name);

        param->quantized = false;
        param->type      = layer->type_str;
        param->trainable = false;
        param->name      = layer->name;
        layer->param     = std::shared_ptr<LayerParam>(param);
    }

    Status NetOptimizerInsertLossAndGradient::SetTrainLayers(NetStructure *structure, NetResource *resource,
                                                             std::set<std::string> &need_grad_layers) {
        if (train_config.trainable_layers.empty())
            return Status(TNNERR_NET_ERR, "train mode but trainable_layers is empty");

        // set loss func layers
        if (train_config.loss_func != DEFAULT_FUNC) {
            if (train_config.target_name.empty() || train_config.output_layer_name.empty() ||
                train_config.target_shape.empty() || train_config.loss_layer_name.empty())
                return Status(TNNERR_NET_ERR, "loss_func set but target_name or output_layer_name is empty");

            structure->inputs_shape_map[train_config.target_name] = train_config.target_shape;
            LayerParam *param                                     = nullptr;
            std::shared_ptr<LayerInfo> last_layer;
            std::shared_ptr<LayerInfo> cur_layer;
            for (auto &tl : structure->layers) {
                if (tl->name == train_config.output_layer_name)
                    last_layer = tl;
            }
            if (last_layer == nullptr || last_layer->outputs.size() <= 0)
                return Status(TNNERR_NET_ERR, "find output layer error");
            if (train_config.loss_func == BINARY_CROSS_ENTROPY_FUNC) {  // the output_layer is sigmoid usually
                if (train_config.auto_add_prob_layer && last_layer->type != LAYER_SIGMOID) {
                    cur_layer = std::make_shared<LayerInfo>();
                    param     = new LayerParam();
                    BuildLayer("Sigmoid", cur_layer, last_layer, structure->blobs, param);
                    structure->layers.push_back(cur_layer);
                    last_layer = cur_layer;
                }
                cur_layer = std::make_shared<LayerInfo>();
                param     = new MultidirBroadcastLayerParam();
                BuildLayer("BinaryCrossEntropy", cur_layer, last_layer, structure->blobs, param);
                cur_layer->inputs.push_back(train_config.target_name);
                structure->blobs.insert(train_config.target_name);
                structure->layers.push_back(cur_layer);
                last_layer = cur_layer;
            } else if (train_config.loss_func == CATEGORICAL_CROSS_ENTROPY_FUNC) {
                if (train_config.auto_add_prob_layer && last_layer->type != LAYER_SOFTMAX) {
                    cur_layer                                     = std::make_shared<LayerInfo>();
                    param                                         = new SoftmaxLayerParam();
                    static_cast<SoftmaxLayerParam *>(param)->axis = 1;  // defualt value is 1 in tflite converter
                    BuildLayer("Softmax", cur_layer, last_layer, structure->blobs, param);
                    structure->layers.push_back(cur_layer);
                    last_layer = cur_layer;
                }
                cur_layer = std::make_shared<LayerInfo>();
                param     = new MultidirBroadcastLayerParam();
                BuildLayer("CategoricalCrossEntropy", cur_layer, last_layer, structure->blobs, param);
                cur_layer->inputs.push_back(train_config.target_name);
                structure->blobs.insert(train_config.target_name);
                structure->layers.push_back(cur_layer);
                last_layer = cur_layer;
            } else {
                return Status(TNNERR_NET_ERR, "NOT SUPPORT LOSS FUNC");
            }

            // build loss reduce mean layer
            cur_layer  = std::make_shared<LayerInfo>();
            param      = new ReduceLayerParam();
            auto &axis = static_cast<ReduceLayerParam *>(param)->axis;
            for (int i = 0; i < train_config.target_shape.size(); ++i)
                axis.push_back(i);
            BuildLayer("ReduceMean", cur_layer, last_layer, structure->blobs, param, train_config.loss_layer_name);
            structure->layers.push_back(cur_layer);
            structure->outputs.insert(cur_layer->name);
        }
        std::map<std::string, LayerInfo *> blob_to_layer;
        for (auto &layer : structure->layers) {
            for (auto &name : layer->outputs) {
                blob_to_layer[name] = layer.get();
            }
        }

        for (auto &layer : structure->layers) {
            DeepVisit(layer.get(), train_config.trainable_layers, blob_to_layer, need_grad_layers,
                      structure->inputs_shape_map);
        }
        // set net resource trainable
        for (auto &iter : resource->resource_map) {
            if (train_config.trainable_layers.find(iter.first) != train_config.trainable_layers.end()) {
                if (iter.second)
                    iter.second->SetTrainable(true);
            }
        }
        return TNN_OK;
    }

    Status NetOptimizerInsertLossAndGradient::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }
        if (!resource) {
            LOGE("Error: empty NetResource\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetResource");
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_orig = structure->layers;
        const int count                                     = (const int)layers_orig.size();
        if (count <= 0) {
            return TNN_OK;
        }

        // skip if network is quantized
        auto is_quantized_net = GetQuantizedInfoFromNetStructure(structure);
        if (is_quantized_net) {
            return TNN_OK;
        }

        // std::vector<std::shared_ptr<LayerInfo>> layers_fused = layers_orig;

        std::set<std::string> need_grad_string;
        Status ret = SetTrainLayers(structure, resource, need_grad_string);
        RETURN_ON_NEQ(ret, TNN_OK);

        // structure->layers = layers_fused;

        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS
