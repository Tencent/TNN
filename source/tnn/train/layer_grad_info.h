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

#ifndef TNN_SOURCE_TNN_TRAIN_LAYER_GRAD_INFO_H
#define TNN_SOURCE_TNN_TRAIN_LAYER_GRAD_INFO_H

#include <vector>

#include "tnn/core/blob.h"
#include "tnn/interpreter/raw_buffer.h"

namespace TNN_NS {

struct LayerGradInfo {
    // upstream grads correspond to outputs of the forward layer
    std::vector<Blob *> upstream_grads;
    // if multipy layers update the same gradient, the results will be accumulated
    std::vector<bool> accumulate_blob_grad;
    std::vector<bool> accumulate_resource_grad;
    // resource to be updated
    std::vector<RawBuffer *> trainable_resources;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_TRAIN_LAYER_GRAD_INFO_H
