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

// author: sanerzheng@tencent.com

#include "tnn/train/solver/sgd.h"

namespace TNN_NS {
namespace train {

    void GetUpdateValue(float *ptr, int count, float learningrate) {
        // NOTE: don't deal with dataformat
        for (int i = 0; i < count; i++) {
            ptr[i] *= learningrate;
        }
    }

    SGD::SGD(AbstractNetwork *network, NetworkConfig *config, float learningrate)
        : BaseSolver(network, config), learningrate_(learningrate) {}

    SGD::~SGD() {}

    Status SGD::ComputeUpdateValue(RawBuffer *resource_param, std::shared_ptr<RawBuffer> &resource_param_grad) {
        if (resource_param->GetDataType() != resource_param_grad->GetDataType() ||
            resource_param->GetDataCount() != resource_param_grad->GetDataCount()) {
            return Status(TNN_TRAIN_ERROR, "grad data type or dims not match");
        }

        int count = resource_param->GetDataCount();
        if (count <= 0) {
            return Status(TNN_TRAIN_ERROR, "grad data count error");
        }

        if (resource_param->GetDataType() == DATA_TYPE_FLOAT) {
            GetUpdateValue(resource_param_grad->force_to<float *>(), count, learningrate_);
        } else {
            return Status(TNN_TRAIN_ERROR, "grad data type not support");
        }

        return Status(TNN_OK);
    }

}  // namespace train
}  // namespace TNN_NS
