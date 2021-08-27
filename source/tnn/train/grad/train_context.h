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

#ifndef TNN_SOURCE_TNN_TRAIN_GRAD_TRAIN_CONTEXT_H
#define TNN_SOURCE_TNN_TRAIN_GRAD_TRAIN_CONTEXT_H

#include <set>
#include <string>

#include "tnn/core/blob.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/raw_buffer.h"
//#include "tnn/core/abstract_network.h"

namespace TNN_NS {
class AbstractNetwork;
namespace train {
struct TrainContext {
    AbstractNetwork *network;
    NetworkConfig *config;
    std::map<Blob *, std::shared_ptr<RawBuffer>> backward_grads_blob;
    std::map<RawBuffer *, std::shared_ptr<RawBuffer>> backward_grads_resource;
};

} // namespace train
} // namespace TNN_NS
#endif // TNN_SOURCE_TNN_TRAIN_GRAD_TRAIN_CONTEXT_H