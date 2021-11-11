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

NetworkImplFactoryRegister<NetworkImplFactory<DefaultTrainNetwork>> g_network_impl_default_train_factory_register(
    NETWORK_TYPE_DEFAULT_TRAIN);

DefaultTrainNetwork::DefaultTrainNetwork() {}

DefaultTrainNetwork::~DefaultTrainNetwork() {}

Status DefaultTrainNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config,
                                 AbstractModelInterpreter *interpreter, InputShapesMap min_inputs_shape,
                                 InputShapesMap max_inputs_shape, bool enable_const_folder) {
    config_    = net_config;
    Status ret = TNN_OK;

    ret = DefaultNetwork::Init(net_config, model_config, interpreter, min_inputs_shape, max_inputs_shape,
                               enable_const_folder);
    RETURN_ON_NEQ(ret, TNN_OK);

    RETURN_ON_NEQ(UpdateNeedGradLayers(), TNN_OK);

    RETURN_ON_NEQ(UpdateSolver(), TNN_OK);

    context_->SetTraining(config_.train_config.run_mode == TRAIN_MODE_TRAIN);
    return TNN_OK;
}

Status DefaultTrainNetwork::TrainStep() {
    if (solver_) {
        return solver_->Step();
    } else {
        LOGE("ERROR: DefaultTrainNetwork::TrainStep, solver is empty\n");
        return Status(TNN_TRAIN_ERROR, "solver is empty");
    }
};

Status DefaultTrainNetwork::UpdateNeedGradLayers() {
    need_grad_layers_.clear();
    CHECK_PARAM_NULL(net_structure_);

    for (auto layer : net_structure_->layers) {
        if (layer->type == LAYER_GRADIENT) {
            GradientParam *param = dynamic_cast<GradientParam *>(layer->param.get());
            CHECK_PARAM_NULL(param);
            need_grad_layers_.insert(param->forward_layer_name);
        }
    }

    return TNN_OK;
}

static std::string GetLossBlobName(const NetStructure *structure) {
    LayerInfo *loss_layer = nullptr;
    for (auto layer : structure->layers) {
        if (layer->type == LAYER_GRADIENT) {
            break;
        } else {
            loss_layer = layer.get();
        }
    }
    return loss_layer->outputs[0];
}

Status DefaultTrainNetwork::UpdateSolver() {
    if (config_.train_config.solver_type == SOLVER_TYPE_SGD) {
        float learning_rate = config_.train_config.solver_params.learning_rate;
        if (config_.train_config.loss_name.empty()) {
            config_.train_config.loss_name = GetLossBlobName(net_structure_);
        }
        solver_             = std::make_shared<train::SGD>(this, &config_, learning_rate);
        solver_->SetNeedGradLayers(need_grad_layers_);
    } else {
        return Status(TNNERR_NET_ERR, "not support slover type in train mode");
    }

    return TNN_OK;
}

}  // namespace TNN_NS

#endif  // TRAIN
