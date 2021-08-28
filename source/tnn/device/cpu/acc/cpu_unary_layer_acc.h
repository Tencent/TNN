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

#ifndef TNN_SOURCE_TNN_DEVICE_CPU_CPU_UNARY_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_CPU_CPU_UNARY_LAYER_ACC_H_

#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "tnn/core/abstract_layer_acc.h"
#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/device/cpu/cpu_device.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/bfp16_utils.h"

namespace TNN_NS {

typedef struct unary_operator {
public:
    virtual Status Init(LayerParam *param = NULL) {
        param_ = param;
        return TNN_OK;
    }
    virtual float operator()(float in) {
        return in;
    }
    virtual int operator()(int in) {
        return in;
    }
    virtual bfp16_t operator()(bfp16_t in) {
        return in;
    }
    virtual int8_t operator()(int8_t in) {
        return in;
    }

protected:
    LayerParam *param_ = NULL;
} UNARY_OP;

// @brief cpu unary layer acc
class CpuUnaryLayerAcc : public CpuLayerAcc {
public:
    CpuUnaryLayerAcc() : op_(NULL) {}

    // @brief virtual destrcutor
    virtual ~CpuUnaryLayerAcc(){};

    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs);

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

protected:
    std::shared_ptr<UNARY_OP> op_;
};

#define DECLARE_UNARY_ACC(type_string, layer_type, OP_TYPE)                                                            \
    class Cpu##type_string##LayerAcc : public CpuUnaryLayerAcc {                                                       \
    public:                                                                                                            \
        Cpu##type_string##LayerAcc() {                                                                                 \
            CpuUnaryLayerAcc::op_ = std::make_shared<OP_TYPE>();                                                       \
        };                                                                                                             \
        virtual ~Cpu##type_string##LayerAcc(){};                                                                       \
    }

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_CPU_CPU_UNARY_LAYER_ACC_H_
