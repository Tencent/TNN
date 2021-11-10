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

    context_->SetTraining(config_.train_config.run_mode == TRAIN_MODE);
    return TNN_OK;
}

Status DefaultTrainNetwork::TrainStep() {
    if (context_->IsTraining()) {
        if (solver_) {
            return solver_->Step();
        } else {
            LOGE("ERROR: DefaultTrainNetwork::TrainStep, solver is empty\n");
            return Status(TNN_TRAIN_ERROR, "solver is empty");
        }
    } else {
        return Status(TNN_TRAIN_ERROR, "not in train mode");
    }
};

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

}  // namespace TNN_NS

#endif  // TRAIN
