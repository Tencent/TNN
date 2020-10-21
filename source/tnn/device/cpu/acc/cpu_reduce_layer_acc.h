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

#ifndef TNN_SOURCE_TNN_DEVICE_CPU_CPU_REDUCE_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_CPU_CPU_REDUCE_LAYER_ACC_H_

#include <vector>

#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/device/cpu/cpu_device.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/bfp16_utils.h"

namespace TNN_NS {

// @brief reduce layer acc
class CpuReduceLayerAcc : public CpuLayerAcc {
public:
    // @brief virtual destrcutor
    virtual ~CpuReduceLayerAcc();

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

protected:
    virtual Status PreCalculateReduce(float *dst, float *src, int count);
    virtual Status PostCalculateReduce(float *dst, float *src, int count);

private:
    virtual Status CalculateReduce(float *output_data, float *input_data, int outer_dim, int channels,
                                   int inner_dim) = 0;
};

#define DECLARE_CPU_REDUCE_ACC(type_string, layer_type)                                                                \
    class Cpu##type_string##LayerAcc : public CpuReduceLayerAcc {                                                      \
    public:                                                                                                            \
        virtual ~Cpu##type_string##LayerAcc(){};                                                                       \
                                                                                                                       \
    private:                                                                                                           \
        virtual Status CalculateReduce(float *output_data, float *input_data, int outer_dim, int channels,             \
                                       int inner_dim);                                                                 \
    }

#define DECLARE_CPU_PRE_REDUCE_POST_ACC(type_string, layer_type)                                                       \
    class Cpu##type_string##LayerAcc : public CpuReduceLayerAcc {                                                      \
    public:                                                                                                            \
        virtual ~Cpu##type_string##LayerAcc(){};                                                                       \
                                                                                                                       \
    protected:                                                                                                         \
        virtual Status PreCalculateReduce(float *dst, float *src, int count);                                          \
        virtual Status PostCalculateReduce(float *dst, float *src, int count);                                         \
                                                                                                                       \
    private:                                                                                                           \
        virtual Status CalculateReduce(float *output_data, float *input_data, int outer_dim, int channels,             \
                                       int inner_dim);                                                                 \
    }

#define REGISTER_CPU_REDUCE_ACC(type_string, layer_type)                                                               \
    CpuTypeLayerAccRegister<TypeLayerAccCreator<Cpu##type_string##LayerAcc>> g_cpu_##layer_type##_acc_register(        \
        layer_type);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_CPU_CPU_REDUCE_LAYER_ACC_H_
