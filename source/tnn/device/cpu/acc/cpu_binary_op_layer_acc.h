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

#ifndef TNN_SOURCE_TNN_DEVICE_CPU_ACC_CPU_BINARY_OP_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_CPU_ACC_CPU_BINARY_OP_LAYER_ACC_H_

#include <vector>

#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/device/cpu/cpu_device.h"

namespace TNN_NS {

class CpuBinaryOpLayerAcc : public CpuLayerAcc {
public:
    virtual ~CpuBinaryOpLayerAcc();

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

private:
    virtual Status Calculate(const std::vector<Blob *> &input_blobs, const std::vector<void *> &input_ptrs,
                             const std::vector<DimsVector> &input_shapes, Blob *output) = 0;
};

#define DECLARE_CPU_BINARY_OP_ACC(type_string, layer_type)                                                             \
    class Cpu##type_string##LayerAcc : public CpuBinaryOpLayerAcc {                                                    \
    public:                                                                                                            \
        virtual ~Cpu##type_string##LayerAcc(){};                                                                       \
        virtual Status Init(Context *context, LayerParam *param, LayerResource *resource,                              \
                            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {                   \
            if (inputs.size() == 1) {                                                                                  \
                CPU_CONVERT_HALF_RESOURCE(layer_type);                                                                 \
            } else {                                                                                                   \
                RETURN_ON_NEQ(CpuLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);                   \
            }                                                                                                          \
            return TNN_OK;                                                                                             \
        }                                                                                                              \
                                                                                                                       \
    protected:                                                                                                         \
        std::shared_ptr<LayerResource> fp32_resource_ = nullptr;                                                       \
                                                                                                                       \
    private:                                                                                                           \
        virtual Status Calculate(const std::vector<Blob *> &input_blobs, const std::vector<void *> &input_ptrs,        \
                                 const std::vector<DimsVector> &input_shapes, Blob *output);                           \
    }
}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_CPU_ACC_CPU_BINARY_OP_LAYER_ACC_H_
