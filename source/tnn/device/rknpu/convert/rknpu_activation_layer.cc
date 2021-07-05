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

#include "rknpu_base_layer.h"
#include "rknpu_utils.h"
#include "tnn/core/layer_type.h"

#ifndef TNN_SOURCE_TNN_DEVICE_RK_NPU_CONVERT_RKNPU_ACTIVATION_LAYER_H_
#define TNN_SOURCE_TNN_DEVICE_RK_NPU_CONVERT_RKNPU_ACTIVATION_LAYER_H_

namespace TNN_NS {

class RknpuActivationLayer : public RknpuBaseLayer {
protected:
    rk::nn::OperatorType type = rk::nn::OperatorType::RELU;

    virtual Status Convert() {
        Status ret = TNN_OK;
        std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;

        // input
        inputs.push_back(input_ops_[0]);

        // output
        ADD_OUTPUT_OP();

        switch (type_) {
            case LAYER_SIGMOID:
                type = rk::nn::OperatorType::SIGMOID;
                break;
            case LAYER_RELU:
                type = rk::nn::OperatorType::RELU;
                break;
            case LAYER_TANH:
                type = rk::nn::OperatorType::TANH;
                break;
            case LAYER_ELU: {
                type       = rk::nn::OperatorType::ELU;
                auto param = dynamic_cast<EluLayerParam *>(param_);
                CHECK_PARAM_NULL(param);
                // output->set_attr_coef(param->alpha);
            } break;
            case LAYER_PRELU: {
                type          = rk::nn::OperatorType::PRELU;
                auto param    = dynamic_cast<PReluLayerParam *>(param_);
                auto resource = dynamic_cast<PReluLayerResource *>(resource_);
                CHECK_PARAM_NULL(param);
                if (!resource) {
                    return Status(TNNERR_MODEL_ERR, "Error: prelu layer resource is nil");
                }
                const float *slope_data = resource->slope_handle.force_to<float *>();
                if (param->channel_shared) {
                    rk::nn::LeakyReluAttr attr;
                    attr.alpha = slope_data[0];
                    graph_->AddOperator(rk::nn::OperatorType::LEAKY_RELU, inputs, output_ops_, (void *)&attr);

                    return TNN_OK;
                } else {
                    // std::vector<int> slope_shape = {1, output_shapes[0][1], 1, 1};
                    std::vector<int> slope_shape = {output_shapes[0][1]};
                    auto slope_const =
                        RknpuUtils::CreateRknnTensor(graph_, layer_name_ + "_slope", slope_shape, slope_data,
                                                     rk::nn::TensorRole::CONST, resource->slope_handle.GetDataType());
                    inputs.push_back(slope_const);
                }
            } break;
            case LAYER_ABS:
                type = rk::nn::OperatorType::ABS;
                break;
            case LAYER_RELU6:
                type = rk::nn::OperatorType::RELU6;
                break;
            case LAYER_LEAKY_RELU:
                type = rk::nn::OperatorType::LEAKY_RELU;
                // 没有 LeakyreluLayerParam 定义, rknpu有 alpha 参数 (coefficient of leakage)
                break;
            default:
                return Status(TNNERR_UNKNOWN_LAYER, "This activation is not defined in NPU");
        }

        graph_->AddOperator(type, inputs, output_ops_, nullptr);

        return ret;
    }

public:
    RknpuActivationLayer(LayerType layer_type) : RknpuBaseLayer(layer_type) {}
    ~RknpuActivationLayer() {}
};

#define DECLARE_RKNPU_ACTIVATION_LAYER(type_string, layer_type)                                                        \
    class Rknpu##type_string##Layer : public RknpuActivationLayer {                                                    \
    public:                                                                                                            \
        Rknpu##type_string##Layer(LayerType ignore) : RknpuActivationLayer(layer_type){};                              \
        ~Rknpu##type_string##Layer(){};                                                                                \
    };

DECLARE_RKNPU_ACTIVATION_LAYER(Sigmoid, LAYER_SIGMOID)
REGISTER_RKNPU_LAYER(Sigmoid, LAYER_SIGMOID)
DECLARE_RKNPU_ACTIVATION_LAYER(Relu, LAYER_RELU)
REGISTER_RKNPU_LAYER(Relu, LAYER_RELU)
// DECLARE_RKNPU_ACTIVATION_LAYER(Tanh, LAYER_TANH)
// REGISTER_RKNPU_LAYER(Tanh, LAYER_TANH)
// DECLARE_RKNPU_ACTIVATION_LAYER(Elu, LAYER_ELU)
// REGISTER_RKNPU_LAYER(Elu, LAYER_ELU)
DECLARE_RKNPU_ACTIVATION_LAYER(Prelu, LAYER_PRELU)
REGISTER_RKNPU_LAYER(Prelu, LAYER_PRELU)
// DECLARE_RKNPU_ACTIVATION_LAYER(Abs, LAYER_ABS)
// REGISTER_RKNPU_LAYER(Abs, LAYER_ABS)
DECLARE_RKNPU_ACTIVATION_LAYER(Relu6, LAYER_RELU6)
REGISTER_RKNPU_LAYER(Relu6, LAYER_RELU6)
DECLARE_RKNPU_ACTIVATION_LAYER(Leakyrelu, LAYER_LEAKY_RELU)
REGISTER_RKNPU_LAYER(Leakyrelu, LAYER_LEAKY_RELU)

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_RK_NPU_CONVERT_RKNPU_ACTIVATION_LAYER_H_
