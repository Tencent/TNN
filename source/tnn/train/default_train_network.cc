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

#if TRAIN

#include "tnn/train/default_train_network.h"
#include "tnn/train/solver/sgd.h"

namespace TNN_NS {

NetworkImplFactoryRegister<NetworkImplFactory<DefaultTrainNetwork>>
    g_network_impl_default_train_factory_register(NETWORK_TYPE_DEFAULT_TRAIN);

DefaultTrainNetwork::DefaultTrainNetwork() {}

DefaultTrainNetwork::~DefaultTrainNetwork() {}

Status DefaultTrainNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config,
                                 AbstractModelInterpreter *interpreter, InputShapesMap min_inputs_shape,
                                 InputShapesMap max_inputs_shape, bool enable_const_folder) {
    config_                                      = net_config;
    Status ret                                   = TNN_OK;
    DefaultModelInterpreter *default_interpreter = dynamic_cast<DefaultModelInterpreter *>(interpreter);
    CHECK_PARAM_NULL(default_interpreter);

    std::set<std::string> need_grad_string;
    ret = SetTrainLayers(default_interpreter, need_grad_string);
    RETURN_ON_NEQ(ret, TNN_OK);

    ret = CreateSolver(need_grad_string);
    RETURN_ON_NEQ(ret, TNN_OK);

    ret = DefaultNetwork::Init(net_config, model_config, interpreter, min_inputs_shape, max_inputs_shape,
                               enable_const_folder);
    RETURN_ON_NEQ(ret, TNN_OK);

    context_->SetTraining(config_.train_config.run_mode == TRAIN_MODE);
    return TNN_OK;
}

Status DefaultTrainNetwork::TrainStep() {
    if (context_->IsTraining())
        return solver_->step();
    else
        return Status(TNN_TRAIN_ERROR, "not in train mode");
};

/*
 * @brief deep vist the compute DAG graph, to find which layers need to be calcualted grads
 * @return if cur layer need to be calculated grad
 */
static bool DeepVisit(const LayerInfo *layer, const std::set<std::string> &trainable_layers,
                      const std::map<std::string, LayerInfo *> &blob_to_layer, std::set<std::string> &need_grad_layers,
                      const InputShapesMap &inputs_shape_map) {
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
                       const std::shared_ptr<LayerInfo> &last_layer, std::set<std::string> &blobs, LayerParam *param,
                       const std::string layer_name = "") {
    layer->type     = GlobalConvertLayerType(type_str);
    layer->type_str = type_str;
    layer->inputs.clear();
    layer->outputs.clear();
    layer->name = layer_name != "" ? layer_name : last_layer->name + "/" + type_str;
    layer->inputs.push_back(last_layer->outputs[0]);
    layer->outputs.push_back(layer->name); // use layer name as output blob name
    blobs.insert(last_layer->outputs[0]);
    blobs.insert(layer->name);

    param->quantized = false;
    param->type      = layer->type_str;
    param->trainable = false;
    param->name      = layer->name;
    layer->param     = std::shared_ptr<LayerParam>(param);
}

Status DefaultTrainNetwork::SetTrainLayers(DefaultModelInterpreter *interpreter,
                                           std::set<std::string> &need_grad_layers) {
    const TrainConfig &train_config = config_.train_config;
    if (train_config.run_mode != TRAIN_MODE)
        return TNN_OK;
    if (!interpreter || !interpreter->GetNetStructure())
        return Status(TNNERR_NET_ERR, "interpreter or netstructrue is null");
    if (train_config.trainable_layers.empty())
        return Status(TNNERR_NET_ERR, "train mode but trainable_layers is empty");
    auto structure = interpreter->GetNetStructure();
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
        if (train_config.loss_func == BINARY_CROSS_ENTROPY_FUNC) { // the output_layer is sigmoid usually
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
                static_cast<SoftmaxLayerParam *>(param)->axis = 1; // defualt value is 1 in tflite converter
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
    for (auto &iter : interpreter->GetNetResource()->resource_map) {
        if (train_config.trainable_layers.find(iter.first) != train_config.trainable_layers.end()) {
            if (iter.second)
                iter.second->SetTrainable(true);
        }
    }
    return TNN_OK;
}

Status DefaultTrainNetwork::CreateSolver(const std::set<std::string> &need_grad_layers) {
    if (config_.train_config.run_mode != TRAIN_MODE)
        return TNN_OK;

    if (config_.train_config.solver_type == SOLVER_SGD) {
        float learning_rate = config_.train_config.sgd_params.learning_rate;
        solver_             = std::make_shared<train::SGD>(this, &config_, learning_rate);
        solver_->SetNeedGradLayers(need_grad_layers);
    } else {
        return Status(TNNERR_NET_ERR, "not support slover type in train mode");
    }
    return TNN_OK;
}

} // namespace TNN_NS

#endif // TRAIN
