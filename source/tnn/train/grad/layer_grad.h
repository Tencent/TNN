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

#ifndef TNN_SOURCE_TNN_TRAIN_GRAD_LAYER_GRAD_H
#define TNN_SOURCE_TNN_TRAIN_GRAD_LAYER_GRAD_H

#include <string>
#include <set>

#include "tnn/core/status.h"
#include "tnn/core/blob.h"
#include "tnn/layer/base_layer.h"

namespace TNN_NS {
namespace train {
class LayerGrad {
public:
    LayerGrad();
    virtual ~LayerGrad() = default;

    //@breif calcute grads
    virtual Status OnGrad(const BaseLayer* layer, std::map<Blob*, std::shared_ptr<RawBuffer> >& backward_grads_blob, std::map<RawBuffer*, std::shared_ptr<RawBuffer> >& backward_grads_resource);

};




} // namespace train
} // namespace TNN_NS
#endif  // TNN_SOURCE_TNN_TRAIN_GRAD_LAYER_GRAD_H