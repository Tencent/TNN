//
// Created by 李烨 on 20/7/20.
//
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

#include "graph/attr_value.h"
#include "npu_base_layer_convert.h"
#include "npu_reduce_layer_convert.h"

namespace TNN_NS {

class NpuReduceProdLayer : public NpuReduceLayer {
public:
    NpuReduceProdLayer(LayerType ignore) : NpuReduceLayer(LAYER_REDUCE_PROD) {}
    ~NpuReduceProdLayer() {}

protected:
    Status Convert() {
        return NpuReduceLayer::ReduceConvertAttr<hiai::op::ReduceProdD>();
    }
};

REGISTER_NPU_LAYER(ReduceProd, LAYER_REDUCE_PROD)

}  // namespace TNN_NS