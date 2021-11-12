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

#ifndef TNN_SOURCE_TNN_TRAIN_GRADIENT_LAYER_GRAD_H
#define TNN_SOURCE_TNN_TRAIN_GRADIENT_LAYER_GRAD_H

#include <set>
#include <string>

#include "tnn/core/blob.h"
#include "tnn/core/status.h"
#include "tnn/layer/base_layer.h"

namespace TNN_NS {

class LayerGrad {
public:
    LayerGrad();

    virtual ~LayerGrad();

    virtual Status OnGrad(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                          LayerResource *resource, LayerParam *param, Context *context) = 0;

    static Status RegisterLayerGrad(DeviceType device, LayerType type, std::shared_ptr<LayerGrad> layer_grad);

    static LayerGrad *GetLayerGrad(DeviceType device, LayerType type);

private:
    static std::map<std::pair<DeviceType, LayerType>, std::shared_ptr<LayerGrad>> &GetLayerGradMap();
};

template <typename T>
class LayerGradRegister {
public:
    explicit LayerGradRegister(DeviceType device, LayerType type) {
        LayerGrad::RegisterLayerGrad(device, type, std::make_shared<T>());
    }
};

#define DECLARE_LAYER_GRAD(device_string, device, type_string, layer_type)                                             \
    class device_string##type_string##LayerGrad : public LayerGrad {                                                   \
    public:                                                                                                            \
        virtual ~device_string##type_string##LayerGrad(){};                                                            \
        virtual Status OnGrad(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,                   \
                              LayerResource *resource, LayerParam *param, Context *context);                           \
    };

#define DECLARE_ARM_LAYER_GRAD(type_string, layer_type) DECLARE_LAYER_GRAD(Arm, DEVICE_ARM, type_string, layer_type)

#define REGISTER_LAYER_GRAD(device_string, device, type_string, layer_type)                                            \
    LayerGradRegister<device_string##type_string##LayerGrad> g_##device##_##layer_type##_layer_grad_register(          \
        device, layer_type);

#define REGISTER_ARM_LAYER_GRAD(type_string, layer_type) REGISTER_LAYER_GRAD(Arm, DEVICE_ARM, type_string, layer_type)

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_TRAIN_GRADIENT_LAYER_GRAD_H
