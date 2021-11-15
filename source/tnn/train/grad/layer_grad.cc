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

#include "tnn/train/grad/layer_grad.h"
#include "tnn/train/operations/op_builder.h"

namespace TNN_NS {
namespace train {
void LayerGrad::UpdateGradValue(Blob *blob, std::shared_ptr<RawBuffer> raw_buff, TrainContext &context) {
    auto iter = context.backward_grads_blob.find(blob);
    if (iter == context.backward_grads_blob.end()) {
        context.backward_grads_blob[blob] = std::move(raw_buff);
    } else {
        // TODO: need _ADD in place
        iter->second = _Add(ParamWrapper(iter->second), ParamWrapper(raw_buff), context).GetRawbufferSharedPtr();
    }
}
void LayerGrad::UpdateGradValue(RawBuffer *resource, std::shared_ptr<RawBuffer> raw_buff, TrainContext &context) {
    auto iter = context.backward_grads_resource.find(resource);
    if (iter == context.backward_grads_resource.end()) {
        context.backward_grads_resource[resource] = std::move(raw_buff);
    } else {
        // TODO: need _ADD in place
        iter->second = _Add(ParamWrapper(resource), ParamWrapper(raw_buff), context).GetRawbufferSharedPtr();
    }
}

void PrintFloatBuffer(RawBuffer *buffer, const std::string &name) {
    auto ptr = buffer->force_to<float *>();
    printf(">> %s:\n", name.c_str());
    for (int i = 0; i < buffer->GetDataCount(); ++i) {
        printf("%f ", ptr[i]);
    }
    printf("\n");
}

} // namespace train
} // namespace TNN_NS