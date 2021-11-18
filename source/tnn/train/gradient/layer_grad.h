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
#include "tnn/train/layer_grad_info.h"
#include "tnn/utils/dims_function_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

class LayerGrad {
public:
    LayerGrad();

    virtual ~LayerGrad();

    virtual Status OnGrad(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                          LayerResource *resource, LayerParam *param, Context *context, LayerGradInfo *grad_info) = 0;

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
                              LayerResource *resource, LayerParam *param, Context *context, LayerGradInfo *grad_info); \
    };

#define DECLARE_ARM_LAYER_GRAD(type_string, layer_type) DECLARE_LAYER_GRAD(Arm, DEVICE_ARM, type_string, layer_type)

#define REGISTER_LAYER_GRAD(device_string, device, type_string, layer_type)                                            \
    LayerGradRegister<device_string##type_string##LayerGrad> g_##device##_##layer_type##_layer_grad_register(          \
        device, layer_type);

#define REGISTER_ARM_LAYER_GRAD(type_string, layer_type) REGISTER_LAYER_GRAD(Arm, DEVICE_ARM, type_string, layer_type)

#define PREPARE_INPUT_AND_GRAD(I)                                                                                      \
    auto input_##I             = inputs[I];                                                                            \
    auto input_grad_##I        = outputs[I];                                                                           \
    bool acc_input_grad_##I    = grad_info->accumulate_input_grad[I];                                                  \
    auto input_##I##_dims      = input_##I->GetBlobDesc().dims;                                                        \
    auto input_grad_##I##_dims = input_grad_##I->GetBlobDesc().dims;                                                   \
    if (!DimsVectorUtils::Equal(input_##I##_dims, input_grad_##I##_dims)) {                                            \
        return Status(TNNERR_LAYER_ERR, "LayerGrad::OnGrad input and input_grad dims not match");                      \
    }
#define PREPARE_INPUT_AND_GRAD1 PREPARE_INPUT_AND_GRAD(0)
#define PREPARE_INPUT_AND_GRAD2                                                                                        \
    PREPARE_INPUT_AND_GRAD(0)                                                                                          \
    PREPARE_INPUT_AND_GRAD(1)

#define PREPARE_RESOURCE_AND_GRAD(I, J)                                                                                \
    auto resource_##I             = resource->GetTrainable()[I];                                                       \
    auto resource_grad_##I        = outputs[J];                                                                        \
    bool acc_resource_grad_##I    = grad_info->accumulate_resource_grad[I];                                            \
    auto resource_##I##_count     = resource_##I->GetDataCount();                                                      \
    auto resource_grad_##I##_dims = resource_grad_##I->GetBlobDesc().dims;                                             \
    if (DimsVectorUtils::Count(resource_grad_##I##_dims) != resource_##I##_count) {                                    \
        return Status(TNNERR_LAYER_ERR, "LayerGrad::OnGrad resource and resource_grad data count not match");          \
    }
#define PREPARE_RESOURCE_AND_GRAD0(I)
#define PREPARE_RESOURCE_AND_GRAD1(I) PREPARE_RESOURCE_AND_GRAD(0, I)
#define PREPARE_RESOURCE_AND_GRAD2(I)                                                                                  \
    PREPARE_RESOURCE_AND_GRAD(0, I)                                                                                    \
    PREPARE_RESOURCE_AND_GRAD(1, I + 1)

#define PREPARE_OUTPUT_AND_GRAD(I, J)                                                                                  \
    auto output_##I             = inputs[J];                                                                           \
    auto output_grad_##I        = inputs[J + 1];                                                                       \
    auto output_##I##_dims      = output_##I->GetBlobDesc().dims;                                                      \
    auto output_grad_##I##_dims = output_grad_##I->GetBlobDesc().dims;                                                 \
    if (!DimsVectorUtils::Equal(output_##I##_dims, output_grad_##I##_dims)) {                                          \
        return Status(TNNERR_LAYER_ERR, "LayerGrad::OnGrad output and output_grad dims not match");                    \
    }
#define PREPARE_OUTPUT_AND_GRAD1(I) PREPARE_OUTPUT_AND_GRAD(0, I)
#define PREPARE_OUTPUT_AND_GRAD2(I)                                                                                    \
    PREPARE_OUTPUT_AND_GRAD(0, I)                                                                                      \
    PREPARE_OUTPUT_AND_GRAD(1, I + 2)

// IOR: input, output and resource counts
#define ON_GRAD_PREPARATION_IOR(I, O, R)                                                                               \
    CHECK_PARAM_NULL(grad_info);                                                                                       \
    if (inputs.size() != (I + O * 2) || outputs.size() != I + R) {                                                     \
        LOGE(                                                                                                          \
            "LayerGrad::OnGrad, input or output size error, input %d vs expected %d + %d, output %d vs expected %d + " \
            "%d\n",                                                                                                    \
            int(inputs.size()), I, O * 2, int(outputs.size()), I, R);                                                  \
        return Status(TNNERR_TRAIN_ERROR, "input or output size error");                                               \
    }                                                                                                                  \
    if (R > 0 && resource->GetTrainable().size() != R) {                                                               \
        LOGE("LayerGrad::OnGrad, trainable size error\n");                                                             \
        return Status(TNNERR_TRAIN_ERROR, "trainable size error");                                                     \
    }                                                                                                                  \
    if (grad_info->accumulate_input_grad.size() != I) {                                                                \
        LOGE("LayerGrad::OnGrad, accumulate_input_grad size error\n");                                                 \
        return Status(TNNERR_TRAIN_ERROR, "accumulate_input_grad size error");                                         \
    }                                                                                                                  \
    if (grad_info->accumulate_resource_grad.size() != R) {                                                             \
        LOGE("LayerGrad::OnGrad, accumulate_resource_grad size error\n");                                              \
        return Status(TNNERR_TRAIN_ERROR, "accumulate_resource_grad size error");                                      \
    }                                                                                                                  \
    PREPARE_INPUT_AND_GRAD##I;                                                                                         \
    PREPARE_OUTPUT_AND_GRAD##O(I);                                                                                     \
    PREPARE_RESOURCE_AND_GRAD##R(I);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_TRAIN_GRADIENT_LAYER_GRAD_H
