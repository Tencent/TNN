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

#include "tnn/train/optimizer/net_optimizer_insert_loss_and_gradient.h"

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#include "tnn/core/layer_type.h"
#include "tnn/core/macro.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"
#include "tnn/interpreter/tnn/model_packer.h"

namespace TNN_NS {

namespace optimizer {

    // P2 priority: reformat after all fuse, remove
    NetOptimizerRegister<NetOptimizerInsertLossAndGradient> g_net_optimizer_insert_loss_and_gradient(OptPriority::P2);
    static const std::map<LossFunc, std::string> kProbSuffix{{LOSS_FUNC_BINARY_CROSS_ENTROPY, "_tnn_sigmoid"},
                                                             {LOSS_FUNC_CATEGORICAL_CROSS_ENTROPY, "_tnn_softmax"}};
    static const std::map<LossFunc, std::string> kEntropySuffix{
        {LOSS_FUNC_BINARY_CROSS_ENTROPY, "_tnn_binary_ce"},
        {LOSS_FUNC_CATEGORICAL_CROSS_ENTROPY, "_tnn_categorical_ce"}};
    static const std::string loss_suffix           = "_tnn_loss";
    static const std::string gradient_suffix       = "_tnn_grad";
    static const std::string resource_grad_suffix  = "_tnn_resource_grad_";
    static const std::string grad_update_name      = "__tnn_grad_update__";
    static const std::string global_step_name      = "__tnn_global_step__";
    static const std::string global_step_init_name = "__tnn_global_step_init__";

    std::string NetOptimizerInsertLossAndGradient::Strategy() {
        return kNetOptimizerInsertLossAndGradient;
    }

    bool NetOptimizerInsertLossAndGradient::IsSupported(const NetworkConfig &net_config) {
        bool is_support = false;
        train_config    = net_config.train_config;
        if (train_config.run_mode == TRAIN_MODE_TRAIN) {
            auto device = net_config.device_type;
            if (device == DEVICE_ARM || device == DEVICE_NAIVE) {
                is_support = true;
            }
        }
        return is_support;
    }

    Status NetOptimizerInsertLossAndGradient::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_TRAIN_ERROR, "Error: empty NetStructure");
        }
        if (structure->layers.empty()) {
            return TNN_OK;
        }
        if (!resource) {
            LOGE("Error: empty NetResource\n");
            return Status(TNNERR_TRAIN_ERROR, "Error: empty NetResource");
        }

        // skip if network is quantized
        auto is_quantized_net = GetQuantizedInfoFromNetStructure(structure);
        if (is_quantized_net) {
            return TNN_OK;
        }

        if (train_config.trainable_layers.empty() && !train_config.train_the_whole_model) {
            return Status(TNNERR_TRAIN_ERROR, "train mode but trainable_layers is empty");
        }

        resource_grads_.clear(); // clear to be able to load other model in A process.

        RETURN_ON_NEQ(InsertLossLayer(structure), TNN_OK);

        RETURN_ON_NEQ(InsertGradientLayers(structure, resource), TNN_OK);

        RETURN_ON_NEQ(InsertGradientUpdateLayer(structure), TNN_OK);

        // ModelPacker packer(structure, resource);
        // packer.Pack("pack.tnnproto", "pack.tnnmodel");

        return TNN_OK;
    }

    Status NetOptimizerInsertLossAndGradient::InsertLossLayer(NetStructure *net_structure) {
        if (train_config.loss_func == LOSS_FUNC_DEFAULT) {
            // the last layer should output loss
            auto loss_layer = net_structure->layers.back();
            loss_blob_      = loss_layer->outputs[0];
            return TNN_OK;
        }

        // target blob
        if (train_config.ground_truth_name.empty() || train_config.ground_truth_shape.empty()) {
            LOGE(
                "NetOptimizerInsertLossAndGradient::InsertLossLayer, loss func is %d, please set target name and shape "
                "to calculate loss\n",
                train_config.loss_func);
            return Status(TNNERR_TRAIN_ERROR,
                          "loss layer will be added, but target(ground truth) name and shape is empty!");
        }
        net_structure->inputs_shape_map[train_config.ground_truth_name] = train_config.ground_truth_shape;
        net_structure->blobs.insert(train_config.ground_truth_name);

        // target layer
        std::shared_ptr<LayerInfo> target_layer = GetTargetLayer(net_structure);
        if (target_layer == nullptr || target_layer->outputs.size() <= 0) {
            return Status(TNNERR_TRAIN_ERROR, "get target layer error");
        }

        // probability layer
        std::shared_ptr<LayerInfo> prob_layer = GetOrCreateProbability(target_layer);
        if (prob_layer == nullptr) {
            return Status(TNNERR_TRAIN_ERROR, "get or create prob layer error");
        }
        if (prob_layer != target_layer) {
            auto prob_input = target_layer->outputs[0];
            prob_layer->inputs.push_back(prob_input);
            auto prob_output = prob_input + kProbSuffix.at(train_config.loss_func);
            prob_layer->outputs.push_back(prob_output);
            net_structure->layers.push_back(prob_layer);
            net_structure->blobs.insert(prob_output);
        }

        // cross entropy
        std::shared_ptr<LayerInfo> entropy_layer =
            CreateCrossEntropy(prob_layer->name + kEntropySuffix.at(train_config.loss_func));
        if (entropy_layer == nullptr) {
            return Status(TNNERR_TRAIN_ERROR, "create entropy layer error");
        } else {
            auto entropy_input = prob_layer->outputs[0];
            entropy_layer->inputs.push_back(entropy_input);
            entropy_layer->inputs.push_back(train_config.ground_truth_name);
            auto entropy_output = entropy_input + kEntropySuffix.at(train_config.loss_func);
            entropy_layer->outputs.push_back(entropy_output);
            net_structure->layers.push_back(entropy_layer);
            net_structure->blobs.insert(entropy_output);
        }

        // reduce mean
        std::shared_ptr<LayerInfo> reduce_layer = CreateReduceMean(entropy_layer->name + loss_suffix);
        if (reduce_layer == nullptr) {
            return Status(TNNERR_TRAIN_ERROR, "create reduce mean layer error");
        } else {
            auto reduce_input = entropy_layer->outputs[0];
            reduce_layer->inputs.push_back(reduce_input);
            auto reduce_output = reduce_input + loss_suffix;
            reduce_layer->outputs.push_back(reduce_output);
            net_structure->layers.push_back(reduce_layer);
            net_structure->blobs.insert(reduce_output);
            net_structure->outputs.insert(reduce_output);
            loss_blob_ = reduce_output;
        }

        return TNN_OK;
    }


    bool NetOptimizerInsertLossAndGradient::LayerNameExist(
        const std::vector<std::shared_ptr<LayerInfo>>& layers, const std::string& name) {
        for (const auto& layer : layers) {
            if (layer->name == name) {
                return true;
            }
        }
        return false;
    }

    Status NetOptimizerInsertLossAndGradient::InsertGradientLayers(NetStructure *net_structure,
                                                                   NetResource *net_resource) {
        std::set<std::string> need_grad_layers;
        RETURN_ON_NEQ(GetNeedGradLayers(net_structure, net_resource, need_grad_layers), TNN_OK);

        std::map<std::string, std::string> blob_to_grad_map;
        auto ori_layers = net_structure->layers;
        for (auto iter = ori_layers.rbegin(); iter != ori_layers.rend(); ++iter) {
            auto forward_layer = *iter;
            if (need_grad_layers.find(forward_layer->name) != need_grad_layers.end()) {
                std::shared_ptr<LayerInfo> grad_layer = CreateGradient(forward_layer.get());
                if (LayerNameExist(net_structure->layers, grad_layer->name)) {
                    return Status(TNNERR_TRAIN_ERROR, "layer name already exist");
                }

                // forward blob gradients
                grad_layer->inputs.clear();
                for (auto forward_input : forward_layer->inputs) {
                    grad_layer->inputs.push_back(forward_input);
                    auto blob_grad = forward_input + gradient_suffix;
                    grad_layer->outputs.push_back(blob_grad);
                    net_structure->blobs.insert(blob_grad);
                    blob_to_grad_map[forward_input] = blob_grad;
                }

                std::vector<std::string> output_grads;
                for (auto forward_output : forward_layer->outputs) {
                    grad_layer->inputs.push_back(forward_output);
                    if (forward_output != loss_blob_) {
                        if (blob_to_grad_map.find(forward_output) != blob_to_grad_map.end()) {
                            output_grads.push_back(blob_to_grad_map[forward_output]);
                        } else {
                            LOGE(
                                "NetOptimizerInsertLossAndGradient::InsertGradientLayers ERROR, can not find blob "
                                "grad: %s\n",
                                forward_output.c_str());
                            return Status(TNNERR_TRAIN_ERROR, "can not find blob grad");
                        }
                    } else {
                        auto loss_grad = forward_output + gradient_suffix;
                        output_grads.push_back(loss_grad);
                        net_structure->blobs.insert(loss_grad);
                        net_structure->inputs_shape_map.insert({loss_grad, {1}});
                    }
                }
                grad_layer->inputs.insert(grad_layer->inputs.end(), output_grads.begin(), output_grads.end());

                // resource gradients
                if (train_config.train_the_whole_model ||
                    (train_config.trainable_layers.find(forward_layer->name) != train_config.trainable_layers.end())) {
                    const auto &resource_map = net_resource->resource_map;
                    if (resource_map.find(forward_layer->name) != resource_map.end()) {
                        auto grad_param = dynamic_cast<GradientParam *>(grad_layer->param.get());
                        if (!grad_param) {
                            LOGE(
                                "NetOptimizerInsertLossAndGradient::InsertGradientLayers ERROR, get grad param "
                                "failed\n");
                            return Status(TNNERR_TRAIN_ERROR, "get grad param failed");
                        }
                        grad_param->need_train = true;
                        auto layer_resource    = resource_map.at(forward_layer->name);
                        for (int i = 0; i < layer_resource->GetTrainable().size(); ++i) {
                            auto resource_grad = forward_layer->name + resource_grad_suffix + std::to_string(i);
                            grad_layer->outputs.push_back(resource_grad);
                            net_structure->blobs.insert(resource_grad);
                            if (layer_resource->GetTrainable()[i]->GetDataCount() > 0) {
                                resource_grads_.push_back(resource_grad);
                            }
                        }
                    }
                }

                net_structure->grad_blobs.insert(grad_layer->outputs.begin(), grad_layer->outputs.end());
                net_structure->back2forward[grad_layer->name] = forward_layer->name;
                net_structure->layers.push_back(grad_layer);
            }
        }

        return TNN_OK;
    }

    Status NetOptimizerInsertLossAndGradient::InsertGradientUpdateLayer(NetStructure *net_structure) {
        // solver
        std::shared_ptr<LayerInfo> solver_layer = CreateSolver(grad_update_name);
        if (solver_layer == nullptr) {
            return Status(TNNERR_TRAIN_ERROR, "create solver layer error");
        } else {
            solver_layer->inputs = resource_grads_;
            solver_layer->inputs.push_back(global_step_init_name);
            solver_layer->outputs.push_back(global_step_name);
            net_structure->layers.push_back(solver_layer);
            net_structure->blobs.insert(global_step_init_name);
            net_structure->blobs.insert(global_step_name);
            net_structure->inputs_shape_map.insert({global_step_init_name, {1}});
            net_structure->outputs.insert(global_step_name);
        }

        return TNN_OK;
    }

    std::shared_ptr<LayerInfo> NetOptimizerInsertLossAndGradient::GetTargetLayer(NetStructure *net_structure) {
        std::shared_ptr<LayerInfo> last_layer;
        if (train_config.target_layer.empty()) {
            LOGD("NetOptimizerInsertLossAndGradient::InsertLossLayer, target layer is empty, use the last layer\n");
            last_layer = net_structure->layers.back();
        } else {
            for (auto layer : net_structure->layers) {
                if (layer->name == train_config.target_layer) {
                    last_layer = layer;
                }
            }
        }
        return last_layer;
    }

    std::shared_ptr<LayerInfo> NetOptimizerInsertLossAndGradient::GetOrCreateProbability(
        std::shared_ptr<LayerInfo> target_layer) {
        if (train_config.loss_func == LOSS_FUNC_BINARY_CROSS_ENTROPY) {
            if (train_config.auto_add_prob_layer && target_layer->type != LAYER_SIGMOID) {
                std::shared_ptr<LayerInfo> new_layer = std::shared_ptr<LayerInfo>(new LayerInfo());
                new_layer->type                      = LAYER_SIGMOID;
                new_layer->type_str                  = "Sigmoid";
                new_layer->name                      = target_layer->name + kProbSuffix.at(train_config.loss_func);
                LayerParam *param                    = new LayerParam();
                new_layer->param                     = std::shared_ptr<LayerParam>(param);
                new_layer->param->type               = new_layer->type_str;
                new_layer->param->name               = new_layer->name;
                return new_layer;
            } else {
                LOGD(
                    "NetOptimizerInsertLossAndGradient::GetOrCreateProbability, use the target layer as probability "
                    "layer: %s\n",
                    target_layer->name.c_str());
                return target_layer;
            }
        } else if (train_config.loss_func == LOSS_FUNC_CATEGORICAL_CROSS_ENTROPY) {
            if (train_config.auto_add_prob_layer && target_layer->type != LAYER_SOFTMAX) {
                std::shared_ptr<LayerInfo> new_layer = std::shared_ptr<LayerInfo>(new LayerInfo());
                new_layer->type                      = LAYER_SOFTMAX;
                new_layer->type_str                  = "Softmax";
                new_layer->name                      = target_layer->name + kProbSuffix.at(train_config.loss_func);
                SoftmaxLayerParam *param             = new SoftmaxLayerParam();
                new_layer->param                     = std::shared_ptr<LayerParam>(param);
                new_layer->param->type               = new_layer->type_str;
                new_layer->param->name               = new_layer->name;
                // defualt value is 1 in tflite converter
                param->axis = 1;
                return new_layer;
            } else {
                LOGD(
                    "NetOptimizerInsertLossAndGradient::GetOrCreateProbability, use the target layer as probability "
                    "layer: %s\n",
                    target_layer->name.c_str());
                return target_layer;
            }
        } else {
            LOGE("NetOptimizerInsertLossAndGradient::GetOrCreateProbability, not support loss func");
            return nullptr;
        }
    }

    std::shared_ptr<LayerInfo> NetOptimizerInsertLossAndGradient::CreateCrossEntropy(const std::string &name) {
        std::shared_ptr<LayerInfo> new_layer = std::shared_ptr<LayerInfo>(new LayerInfo());
        if (train_config.loss_func == LOSS_FUNC_BINARY_CROSS_ENTROPY) {
            new_layer->type     = LAYER_BINARY_CROSSENTROPY;
            new_layer->type_str = "BinaryCrossEntropy";
        } else if (train_config.loss_func == LOSS_FUNC_CATEGORICAL_CROSS_ENTROPY) {
            new_layer->type     = LAYER_CATEGORICAL_CROSSENTROPY;
            new_layer->type_str = "CategoricalCrossEntropy";
        } else {
            LOGE("NetOptimizerInsertLossAndGradient::CreateCrossEntropy, not support loss func");
            return nullptr;
        }
        new_layer->name                    = name;
        MultidirBroadcastLayerParam *param = new MultidirBroadcastLayerParam();
        new_layer->param                   = std::shared_ptr<LayerParam>(param);
        new_layer->param->type             = new_layer->type_str;
        new_layer->param->name             = new_layer->name;
        return new_layer;
    }

    std::shared_ptr<LayerInfo> NetOptimizerInsertLossAndGradient::CreateReduceMean(const std::string &name) {
        std::shared_ptr<LayerInfo> new_layer = std::shared_ptr<LayerInfo>(new LayerInfo());
        new_layer->type                      = LAYER_REDUCE_MEAN;
        new_layer->type_str                  = "ReduceMean";
        new_layer->name                      = name;
        ReduceLayerParam *param              = new ReduceLayerParam();
        new_layer->param                     = std::shared_ptr<LayerParam>(param);
        new_layer->param->type               = new_layer->type_str;
        new_layer->param->name               = new_layer->name;
        for (int i = 0; i < train_config.ground_truth_shape.size(); ++i) {
            param->axis.push_back(i);
        }
        return new_layer;
    }

    std::shared_ptr<LayerInfo> NetOptimizerInsertLossAndGradient::CreateGradient(LayerInfo *forward_layer) {
        std::shared_ptr<LayerInfo> new_layer = std::shared_ptr<LayerInfo>(new LayerInfo());
        new_layer->type                      = LAYER_GRADIENT;
        new_layer->type_str                  = "Gradient";
        new_layer->name                      = forward_layer->name + gradient_suffix;
        GradientParam *param                 = new GradientParam();
        new_layer->param                     = std::shared_ptr<LayerParam>(param);
        new_layer->param->type               = new_layer->type_str;
        new_layer->param->name               = new_layer->name;
        param->forward_layer_type            = forward_layer->type;
        param->forward_layer_name            = forward_layer->name;
        param->forward_param                 = forward_layer->param.get();
        return new_layer;
    }

    std::shared_ptr<LayerInfo> NetOptimizerInsertLossAndGradient::CreateSolver(const std::string &name) {
        std::shared_ptr<LayerInfo> new_layer = std::shared_ptr<LayerInfo>(new LayerInfo());
        new_layer->type                      = LAYER_SOLVER;
        new_layer->type_str                  = "Solver";
        new_layer->name                      = name;
        SolverParam *param                   = new SolverParam();
        new_layer->param                     = std::shared_ptr<LayerParam>(param);
        new_layer->param->type               = new_layer->type_str;
        new_layer->param->name               = new_layer->name;
        param->type                          = train_config.solver_type;
        param->learning_rate                 = train_config.solver_params.learning_rate;
        return new_layer;
    }

    Status NetOptimizerInsertLossAndGradient::GetNeedGradLayers(NetStructure *structure,
                                                                NetResource *resource,
                                                                std::set<std::string> &need_grad_layers) {
        // check params first
        std::set<std::string> layer_names_set;
        for(auto ly : structure->layers) {
            layer_names_set.insert(ly->name);
        }

        for (auto name : train_config.trainable_layers) {
            if (layer_names_set.find(name) == layer_names_set.end()) {
                LOGE("NetOptimizerInsertLossAndGradient::GetNeedGradLayers, specified trainable layer: %s not found.\n", name.c_str());
                return Status(TNNERR_TRAIN_ERROR, "specified tranable layer not found.");
            }
        }

        std::map<std::string, LayerInfo *> blob_to_layer;
        for (auto &layer : structure->layers) {
            for (auto &name : layer->outputs) {
                blob_to_layer[name] = layer.get();
            }
        }

        for (auto &layer : structure->layers) {
            if (train_config.train_the_whole_model ||
                (train_config.trainable_layers.find(layer->name) != train_config.trainable_layers.end())) {
                need_grad_layers.insert(layer->name);
                LOGD("Layer need to calculate grad: %s\n", layer->name.c_str());
                continue;
            }

            bool need_grad = false;
            // if previous layer need grad, its succeed need too
            for (auto &input : layer->inputs) {
                if (structure->inputs_shape_map.find(input) == structure->inputs_shape_map.end()
                    && resource->constant_map.find(input) == resource->constant_map.end()) {
                    // not model input, must be output of some layer
                    LayerInfo *prev_layer = blob_to_layer[input];
                    if (prev_layer == nullptr) {
                        LOGE("NetOptimizerInsertLossAndGradient::GetNeedGradLayers, find layer by blob failed\n");
                        return Status(TNNERR_TRAIN_ERROR, "find layer by blob failed");
                    }
                    if (need_grad_layers.find(prev_layer->name) != need_grad_layers.end()) {
                        need_grad = true;
                        break;
                    }
                }
            }
            if (need_grad) {
                need_grad_layers.insert(layer->name);
                LOGD("Layer need to calculate grad: %s\n", layer->name.c_str());
            }
        }

        return TNN_OK;
    }

    /* @brief deep vist the compute DAG graph, to find which layers need to be calcualted grads
     * @return if cur layer need to be calculated grad
     */
    /*
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

    Status NetOptimizerInsertLossAndGradient::GetNeedGradLayers(NetStructure *structure,
                                                                std::set<std::string> &need_grad_layers) {
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

        return TNN_OK;
    }
    */

}  // namespace optimizer

}  // namespace TNN_NS
