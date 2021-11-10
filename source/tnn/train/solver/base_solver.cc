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

#include "tnn/train/solver/base_solver.h"

#include "tnn/train/test_grad/test_layer_grad.h"

namespace TNN_NS {
namespace train {

    void UpdateVariable(float *dst_ptr, const float *src, int count) {
        // NOTE: don't deal with dataformat
        for (int i = 0; i < count; i++) {
            dst_ptr[i] -= src[i];
        }
    }

    BaseSolver::BaseSolver(AbstractNetwork *network, NetworkConfig *config) {
        auto &context   = grad_manager_.GetContext();
        context.network = network;
        context.config  = config;
    }

    BaseSolver::~BaseSolver() {}

    void BaseSolver::SetNeedGradLayers(const std::set<std::string> &need_grad_layers) {
        grad_manager_.SetNeedGradLayers(need_grad_layers);
        return;
    }

    Status BaseSolver::UpdateTrainableVariable(RawBuffer *resource_param,
                                               const std::shared_ptr<RawBuffer> &resource_param_grad) {
        if (resource_param->GetDataType() != resource_param_grad->GetDataType() ||
            resource_param->GetDataCount() != resource_param_grad->GetDataCount()) {
            return Status(TNN_TRAIN_ERROR, "grad data type or dims not match");
        }
        int count = resource_param->GetDataCount();
        if (count <= 0)
            return Status(TNN_TRAIN_ERROR, "grad data count error");
        if (resource_param->GetDataType() == DATA_TYPE_FLOAT) {
            UpdateVariable(resource_param->force_to<float *>(), resource_param_grad->force_to<const float *>(), count);
        } else {
            return Status(TNN_TRAIN_ERROR, "grad data type not support");
        }
        return Status(TNN_OK);
    }

    Status BaseSolver::Step() {
        Status status;
        // //TODO: temp code, need move into unit test main fuc
        // status = LayerGradTestManager::RunTestGrad();
        // if(status != TNN_OK) {
        //     LOGE("train test error: %s", status.description().c_str());
        //     return status;
        // }

        RETURN_ON_NEQ(grad_manager_.IsSupport(), TNN_OK);
        RETURN_ON_NEQ(grad_manager_.CalcuteGrads(), TNN_OK);
        auto &resource_grads = grad_manager_.GetContext().backward_grads_resource;
        for (auto iter : resource_grads) {
            if (iter.first->GetTrainable()) {
                status = ComputeUpdateValue(iter.first, iter.second);
                RETURN_ON_NEQ(status, TNN_OK);
                status = UpdateTrainableVariable(iter.first, iter.second);
                RETURN_ON_NEQ(status, TNN_OK);
            }
        }
        return TNN_OK;
    }

}  // namespace train
}  // namespace TNN_NS
