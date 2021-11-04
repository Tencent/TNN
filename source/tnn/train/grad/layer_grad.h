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

#include <set>
#include <string>

#include "tnn/core/blob.h"
#include "tnn/core/status.h"
#include "tnn/layer/base_layer.h"
#include "tnn/train/grad/train_context.h"

namespace TNN_NS {
namespace train {
class LayerGrad {
public:
    LayerGrad(){};
    virtual ~LayerGrad() = default;
    // @brief calcute grads
    virtual Status OnGrad(const BaseLayer *layer, TrainContext &context) = 0;
    virtual void UpdateGradValue(Blob *blob, std::shared_ptr<RawBuffer> raw_buff, TrainContext &context);
    virtual void UpdateGradValue(RawBuffer *resource, std::shared_ptr<RawBuffer> raw_buff, TrainContext &context);
    static void RegisterLayerGrad(LayerType type, std::shared_ptr<LayerGrad> layer_grad_p) {
        GetLayerGradMap()[type] = layer_grad_p;
    };
    inline static std::map<LayerType, std::shared_ptr<LayerGrad>> &GetLayerGradMap() {
        static std::map<LayerType, std::shared_ptr<LayerGrad>> layer_2_grad_map;
        return layer_2_grad_map;
    };
};

template <typename T> class LayerGradRegister {
public:
    explicit LayerGradRegister(LayerType type) {
        LayerGrad::RegisterLayerGrad(type, std::make_shared<T>());
    }
};

#define DECLARE_LAYER_GRAD(type_string, layer_type)                                                                    \
    class type_string##LayerGrad : public LayerGrad {                                                                  \
    public:                                                                                                            \
        virtual ~type_string##LayerGrad(){};                                                                           \
        virtual Status OnGrad(const BaseLayer *layer, TrainContext &context);                                          \
    };

#define REGISTER_LAYER_GRAD(type_string, layer_type)                                                                   \
    LayerGradRegister<type_string##LayerGrad> g_##layer_type##_layer_grad_register(layer_type);

} // namespace train
} // namespace TNN_NS
#endif // TNN_SOURCE_TNN_TRAIN_GRAD_LAYER_GRAD_H