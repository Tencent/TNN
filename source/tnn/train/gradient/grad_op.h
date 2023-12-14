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

#include <map>
#include <string>
#include <functional>

#include "tnn/core/blob.h"
#include "tnn/core/status.h"
#include "tnn/layer/base_layer.h"
#include "tnn/train/training_info.h"
#include "tnn/utils/dims_function_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

class GradOp;
using GradOpPtr = std::unique_ptr<GradOp>;
using GradOpCreator = std::function<GradOpPtr()>;

class GradOp {
public:
    GradOp();

    virtual ~GradOp();

    virtual Status OnGrad(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                          LayerResource *resource, GradientParam *grad_param, Context *context,
                          const GradOpInfo &grad_info) = 0;

    static Status RegisterGradOpCreator(DeviceType device, LayerType type, GradOpCreator grad_op_creator);

    static GradOpPtr CreateGradOp(DeviceType device, LayerType type);

private:
    static std::map<std::pair<DeviceType, LayerType>, std::function<GradOpPtr()>> &GetGradOpCreatorMap();
};

template <typename T>
class GradOpRegister {
public:
    explicit GradOpRegister(DeviceType device, LayerType type) {
        GradOp::RegisterGradOpCreator(device, type, []() { return GradOpPtr(new T()); });
    }
};

#define DECLARE_GRAD_OP(device_string, device, type_string, layer_type)                                          \
    class device_string##type_string##GradOp : public GradOp {                                                   \
    public:                                                                                                      \
        virtual ~device_string##type_string##GradOp(){};                                                         \
        virtual Status OnGrad(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,             \
                              LayerResource *resource, GradientParam *grad_param, Context *context,              \
                              const GradOpInfo &grad_info);                                                      \
    };

#define REGISTER_GRAD_OP(device_string, device, type_string, layer_type)                                         \
    GradOpRegister<device_string##type_string##GradOp> g_##device##_##layer_type##_layer_grad_register(          \
        device, layer_type);

#define PREPARE_INPUT_AND_GRAD(I)                                                                                      \
    std::vector<Blob *> fw_inputs(inputs.begin(), inputs.begin() + I);                                                 \
    std::vector<Blob *> input_grads(outputs.begin(), outputs.begin() + I);                                             \
    const std::vector<bool> &acc_input_grads = grad_info.accumulate_input_grad;                                        \
    std::vector<DimsVector> input_dims;                                                                                \
    for (int i = 0; i < I; ++i) {                                                                                      \
        input_dims.push_back(fw_inputs[i]->GetBlobDesc().dims);                                                        \
        auto input_grad_dims = input_grads[i]->GetBlobDesc().dims;                                                     \
        if (!DimsVectorUtils::Equal(input_dims[i], input_grad_dims)) {                                                 \
            LOGE("GradOp::OnGrad %s vs %s: , dims not match\n", fw_inputs[i]->GetBlobDesc().description().c_str(),  \
                 input_grads[i]->GetBlobDesc().description().c_str());                                                 \
            return Status(TNNERR_LAYER_ERR, "GradOp::OnGrad input and input_grad dims not match");                  \
        }                                                                                                              \
    }

#define PREPARE_OUTPUT_AND_GRAD(I, O)                                                                                  \
    std::vector<Blob *> fw_outputs(inputs.begin() + I, inputs.begin() + I + O);                                        \
    std::vector<Blob *> output_grads(inputs.begin() + I + O, inputs.begin() + I + O * 2);                              \
    std::vector<DimsVector> output_dims;                                                                               \
    for (int i = 0; i < O; ++i) {                                                                                      \
        output_dims.push_back(fw_outputs[i]->GetBlobDesc().dims);                                                      \
        auto output_grad_dims = output_grads[i]->GetBlobDesc().dims;                                                   \
        if (!DimsVectorUtils::Equal(output_dims[i], output_grad_dims)) {                                               \
            LOGE("GradOp::OnGrad %s vs %s: , dims not match\n", fw_outputs[i]->GetBlobDesc().description().c_str(), \
                 output_grads[i]->GetBlobDesc().description().c_str());                                                \
            return Status(TNNERR_LAYER_ERR, "GradOp::OnGrad output and output_grad dims not match");                \
        }                                                                                                              \
    }

#define PREPARE_RESOURCE_AND_GRAD(I, R)                                                                                \
    std::vector<RawBuffer *> fw_resources;                                                                             \
    if (R > 0) {                                                                                                       \
        fw_resources = resource->GetTrainable();                                                                       \
    }                                                                                                                  \
    std::vector<Blob *> resource_grads(outputs.begin() + I, outputs.begin() + I + R);                                  \
    const std::vector<bool> &acc_resource_grads = grad_info.accumulate_resource_grad;                                  \
    std::vector<DimsVector> resource_dims;                                                                             \
    for (int i = 0; i < R; ++i) {                                                                                      \
        resource_dims.push_back(resource_grads[i]->GetBlobDesc().dims);                                                \
        auto resource_count = fw_resources[i]->GetDataCount();                                                         \
        if (resource_count > 0 && DimsVectorUtils::Count(resource_dims[i]) != resource_count) {                        \
            LOGE("GradOp::OnGrad %d vs %s: , dims not match\n", resource_count,                                     \
                 resource_grads[i]->GetBlobDesc().description().c_str());                                              \
            return Status(TNNERR_LAYER_ERR, "GradOp::OnGrad resource and resource_grad data count not match");      \
        }                                                                                                              \
    }

// IOR: input, output and resource counts
#define ON_GRAD_PREPARATION_IOR(I, O, R)                                                                               \
    if (inputs.size() != ((I) + (O)*2)) {                                                                              \
        LOGE("GradOp::OnGrad, input size error, input %d vs expected %d + %d\n", int(inputs.size()), (I), (O)*2);   \
        return Status(TNNERR_TRAIN_ERROR, "input size error");                                                         \
    }                                                                                                                  \
    int resource_need_train = 0;                                                                                       \
    if (outputs.size() == (I)) {                                                                                       \
        LOGD("GradOp::OnGrad, resource do not need to train, skip calculating resource grads\n");                   \
    } else {                                                                                                           \
        resource_need_train = (R);                                                                                     \
        if (outputs.size() != (I) + resource_need_train) {                                                             \
            LOGE("GradOp::OnGrad, output size error, output %d vs expected %d + %d\n", int(outputs.size()), (I),    \
                 resource_need_train);                                                                                 \
            return Status(TNNERR_TRAIN_ERROR, "output size error");                                                    \
        }                                                                                                              \
    }                                                                                                                  \
    if (resource_need_train > 0 && resource->GetTrainable().size() != resource_need_train) {                           \
        LOGE("GradOp::OnGrad, trainable size error\n");                                                             \
        return Status(TNNERR_TRAIN_ERROR, "trainable size error");                                                     \
    }                                                                                                                  \
    if (grad_info.accumulate_input_grad.size() != (I)) {                                                               \
        LOGE("GradOp::OnGrad, accumulate_input_grad size error\n");                                                 \
        return Status(TNNERR_TRAIN_ERROR, "accumulate_input_grad size error");                                         \
    }                                                                                                                  \
    if (grad_info.accumulate_resource_grad.size() != resource_need_train) {                                            \
        LOGE("GradOp::OnGrad, accumulate_resource_grad size error\n");                                              \
        return Status(TNNERR_TRAIN_ERROR, "accumulate_resource_grad size error");                                      \
    }                                                                                                                  \
    PREPARE_INPUT_AND_GRAD((I));                                                                                       \
    PREPARE_OUTPUT_AND_GRAD((I), (O));                                                                                 \
    PREPARE_RESOURCE_AND_GRAD((I), resource_need_train);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_TRAIN_GRADIENT_LAYER_GRAD_H
