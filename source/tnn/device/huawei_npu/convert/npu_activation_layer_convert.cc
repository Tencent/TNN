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

#include "npu_base_layer_convert.h"
#include "tnn/core/layer_type.h"

#ifndef TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_CONVERT_NPU_ACTIVATION_LAYER_CONVERT_H_
#define TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_CONVERT_NPU_ACTIVATION_LAYER_CONVERT_H_

namespace TNN_NS {

class NpuActivationLayerConvert : public NpuBaseLayer {
protected:
    int mode = 0;
    virtual Status Convert() {
        auto output = std::make_shared<ge::op::Activation>(outputs_name_[0]);
        output->set_input_x(*input_ops_[0]->GetOperator());

        switch (type_) {
            case LAYER_SIGMOID:
                mode = 0;
                break;
            case LAYER_RELU:
                mode = 1;
                break;
            case LAYER_TANH:
                mode = 2;
                break;
            case LAYER_ELU: {
                mode       = 4;
                auto param = dynamic_cast<EluLayerParam *>(param_);
                CHECK_PARAM_NULL(param);
                output->set_attr_coef(param->alpha);
            } break;
            case LAYER_ABS:
                mode = 6;
                break;
            case LAYER_SOFTSIGN:
                mode = 8;
                break;
            case LAYER_SOFTPLUS:
                mode = 9;
                break;
            case LAYER_HARDSIGMOID: {
                auto param = dynamic_cast<HardSigmoidLayerParam *>(param_);
                CHECK_PARAM_NULL(param);
                if (!(param->alpha >= 0.1666f && param->alpha <= 0.1667f && param->beta >= 0.4999f &&
                      param->beta <= 0.5001f)) {
                    LOGE("hardsigmoid only support alpha=1/6 beta=0.5, but in fact, alpha=%f beta=%f\n", param->alpha,
                         param->beta);
                    return Status(TNNERR_LAYER_ERR,
                                  "Error: Npu currently only supports hardsigmoid (alpha=1/6, beta=0.5)");
                }
                mode = 10;
            } break;
            case LAYER_SELU:
                mode = 12;
                break;
            case LAYER_RELU6:
                mode = 14;
                break;
            case LAYER_GELU:
                mode = 15;
                break;
            default:
                return Status(TNNERR_UNKNOWN_LAYER, "This activation is not defined in NPU");
        }

        output->set_attr_mode(mode);
        ADD_OUTPUT_OP(output)
    }

public:
    NpuActivationLayerConvert(LayerType layer_type) : NpuBaseLayer(layer_type) {}
    ~NpuActivationLayerConvert() {}
};

#define DECLARE_NPU_ACTIVATION_LAYER(type_string, layer_type)                                                          \
    class Npu##type_string##Layer : public NpuActivationLayerConvert {                                                 \
    public:                                                                                                            \
        Npu##type_string##Layer(LayerType ignore) : NpuActivationLayerConvert(layer_type){};                           \
        ~Npu##type_string##Layer(){};                                                                                  \
    };

DECLARE_NPU_ACTIVATION_LAYER(Sigmoid, LAYER_SIGMOID)
REGISTER_NPU_LAYER(Sigmoid, LAYER_SIGMOID)
DECLARE_NPU_ACTIVATION_LAYER(Relu, LAYER_RELU)
REGISTER_NPU_LAYER(Relu, LAYER_RELU)
DECLARE_NPU_ACTIVATION_LAYER(Tanh, LAYER_TANH)
REGISTER_NPU_LAYER(Tanh, LAYER_TANH)
DECLARE_NPU_ACTIVATION_LAYER(Elu, LAYER_ELU)
REGISTER_NPU_LAYER(Elu, LAYER_ELU)
DECLARE_NPU_ACTIVATION_LAYER(Abs, LAYER_ABS)
REGISTER_NPU_LAYER(Abs, LAYER_ABS)
DECLARE_NPU_ACTIVATION_LAYER(Softplus, LAYER_SOFTPLUS)
REGISTER_NPU_LAYER(Softplus, LAYER_SOFTPLUS)
DECLARE_NPU_ACTIVATION_LAYER(Softsign, LAYER_SOFTSIGN)
REGISTER_NPU_LAYER(Softsign, LAYER_SOFTSIGN)
DECLARE_NPU_ACTIVATION_LAYER(HardSigmoid, LAYER_HARDSIGMOID)
REGISTER_NPU_LAYER(HardSigmoid, LAYER_HARDSIGMOID)
DECLARE_NPU_ACTIVATION_LAYER(Selu, LAYER_SELU)
REGISTER_NPU_LAYER(Selu, LAYER_SELU)
DECLARE_NPU_ACTIVATION_LAYER(Relu6, LAYER_RELU6)
REGISTER_NPU_LAYER(Relu6, LAYER_RELU6)
DECLARE_NPU_ACTIVATION_LAYER(Gelu, LAYER_GELU)
REGISTER_NPU_LAYER(Gelu, LAYER_GELU)

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_CONVERT_NPU_ACTIVATION_LAYER_CONVERT_H_
