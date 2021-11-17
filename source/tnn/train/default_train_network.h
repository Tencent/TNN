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

#ifndef TNN_SOURCE_TNN_TRAIN_DEFAULT_TRAIN_NETWORK_H_
#define TNN_SOURCE_TNN_TRAIN_DEFAULT_TRAIN_NETWORK_H_

#include "tnn/core/common.h"
#include "tnn/core/default_network.h"
#include "tnn/core/macro.h"
#include "tnn/interpreter/default_model_interpreter.h"

namespace TNN_NS {

class DefaultTrainNetwork : public DefaultNetwork {
public:
    // @brief DefaultTrainNetwork Constructor
    DefaultTrainNetwork();

    // @brief DefaultTrainNetwork virtual Destructor
    virtual ~DefaultTrainNetwork();

public:
    // @brief init solver with net_config.train_config
    virtual Status Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                        InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape,
                        bool enable_const_folder = true) override;

    virtual Status TrainStep() override;
    virtual Status GetTrainingFeedback(TrainingFeedback &feed_back) override;

protected:
    virtual Status UpdateGradMap();
    virtual Status UpdateForwardLayerCount();
    virtual Status UpdateSolver();

    std::map<Blob *, Blob *> forward_blob_to_grad_map_;
    std::map<Blob *, RawBuffer *> grad_to_resource_map_;

    std::string loss_name_;
    std::string global_step_name_;

    TrainMode run_mode_ = TRAIN_MODE_TRAIN;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_TRAIN_DEFAULT_TRAIN_NETWORK_H_
