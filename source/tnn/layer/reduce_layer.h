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

#ifndef TNN_SOURCE_TNN_LAYER_REDUCE_LAYER_H_
#define TNN_SOURCE_TNN_LAYER_REDUCE_LAYER_H_

#include "tnn/layer/base_layer.h"

namespace TNN_NS {

class ReduceLayer : public BaseLayer {
public:
    explicit ReduceLayer(LayerType ignore) : BaseLayer(ignore){};
    virtual ~ReduceLayer(){};

protected:
    virtual Status InferOutputShape(bool ignore_error = false);
};

#define DECLARE_REDUCE_LAYER(type_string, layer_type)                                                                  \
    class type_string##Layer : public ReduceLayer {                                                                    \
    public:                                                                                                            \
        type_string##Layer(LayerType ignore) : ReduceLayer(layer_type){};                                              \
        virtual ~type_string##Layer(){};                                                                               \
    }

#define REGISTER_REDUCE_LAYER(type_string, layer_type) REGISTER_LAYER(type_string, layer_type)

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_LAYER_REDUCE_LAYER_H_
