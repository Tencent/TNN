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

#ifndef TNN_SOURCE_TNN_DEVICE_CPU_CPU_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_CPU_CPU_LAYER_ACC_H_

#include <memory>
#include <vector>

#include "tnn/core/abstract_layer_acc.h"
#include "tnn/device/cpu/acc/compute/compute_elewise.h"
#include "tnn/device/cpu/acc/compute/compute_int8.h"
#include "tnn/device/cpu/cpu_device.h"
#include "tnn/interpreter/layer_resource_generator.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/bfp16_utils.h"

namespace TNN_NS {

// @brief cpu layer acc
class CpuLayerAcc : public AbstractLayerAcc {
public:
    // @brief virtual destrcutor
    virtual ~CpuLayerAcc();

    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs);

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) = 0;

    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) = 0;

protected:
    LayerParam *param_       = nullptr;
    LayerResource *resource_ = nullptr;

private:
    // @brief return device layer acc support data format
    virtual std::vector<DataFormat> SupportDataFormat(DataType data_type, int dims_size);
};

#define DECLARE_CPU_ACC(type_string, layer_type)                                                                       \
    class Cpu##type_string##LayerAcc : public CpuLayerAcc {                                                            \
    public:                                                                                                            \
        virtual ~Cpu##type_string##LayerAcc(){};                                                                       \
        virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);                 \
        virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);                 \
    }

#define CPU_CONVERT_HALF_RESOURCE(layer_type)                                                                          \
    {                                                                                                                  \
        LayerResource *fp32_res = nullptr;                                                                             \
        RETURN_ON_NEQ(ConvertHalfResource(layer_type, resource, &fp32_res), TNN_OK);                                   \
        fp32_resource_ = std::shared_ptr<LayerResource>(fp32_res);                                                     \
        RETURN_ON_NEQ(CpuLayerAcc::Init(context, param, fp32_resource_.get(), inputs, outputs), TNN_OK);               \
    }

#define DECLARE_CPU_ACC_WITH_FP32_RESOURCE(type_string, layer_type)                                                    \
    class Cpu##type_string##LayerAcc : public CpuLayerAcc {                                                            \
    public:                                                                                                            \
        virtual ~Cpu##type_string##LayerAcc(){};                                                                       \
        virtual Status Init(Context *context, LayerParam *param, LayerResource *resource,                              \
                            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {                   \
            CPU_CONVERT_HALF_RESOURCE(layer_type);                                                                     \
            return TNN_OK;                                                                                             \
        }                                                                                                              \
        virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);                 \
        virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);                 \
                                                                                                                       \
    protected:                                                                                                         \
        std::shared_ptr<LayerResource> fp32_resource_ = nullptr;                                                       \
    }

#define REGISTER_CPU_ACC(type_string, layer_type)                                                                      \
    CpuTypeLayerAccRegister<TypeLayerAccCreator<Cpu##type_string##LayerAcc>> g_cpu_##layer_type##_acc_register(        \
        layer_type);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_CPU_CPU_LAYER_ACC_H_
