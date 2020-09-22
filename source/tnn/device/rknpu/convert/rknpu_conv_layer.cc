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
#include "rknpu_conv_layer_impl.h"
#include "rknpu_utils.h"

namespace TNN_NS {

class RknpuConvLayer : public RknpuConvImplLayer {
public:
    RknpuConvLayer(LayerType ignore) : RknpuConvImplLayer(LAYER_CONVOLUTION){};
    virtual ~RknpuConvLayer() {}

protected:
    virtual Status Convert() {
        Status ret    = ObtainParam();
        auto resource = dynamic_cast<ConvLayerResource *>(resource_);
        if (ret != TNN_OK) {
            return ret;
        }
        if (!resource) {
            return Status(TNNERR_MODEL_ERR, "Error: ConvLayerResource is empty");
        }

        // pad type
        rk::nn::PadType rk_pad_type = rk::nn::PadType::AUTO;
        ret                         = RknpuUtils::GetPadType(rk_pad_type, pad_type);
        if (ret != TNN_OK)
            return ret;

        // fuse relu
        if (activation_type > ActivationType_ReLU) {
            return Status(TNNERR_PARAM_ERR, "Error: ConvLayer dont support fuse ActivationType except ReLu");
        }

        std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;

        // input
        inputs.push_back(input_ops_[0]);

        // weight
        int total_data_size           = resource->filter_handle.GetDataCount();
        int in_group                  = total_data_size / (kernel_h * kernel_w * output_channel);
        std::vector<int> weight_shape = {output_channel, in_group, kernel_h, kernel_w};
        auto weight_const             = RknpuUtils::CreateRknnTensor(
            graph_, layer_name_ + "_weight", weight_shape, resource->filter_handle.force_to<void *>(),
            rk::nn::TensorRole::CONST, resource->filter_handle.GetDataType());
        inputs.push_back(weight_const);

        // bias
        int bias_count = resource->bias_handle.GetDataCount();
        if (bias_count != 0) {
            std::vector<int> bias_shape = {1, bias_count, 1, 1};
            auto bias_const             = RknpuUtils::CreateRknnTensor(
                graph_, layer_name_ + "_bias", bias_shape, resource->bias_handle.force_to<void *>(),
                rk::nn::TensorRole::CONST, resource->bias_handle.GetDataType());
            inputs.push_back(bias_const);
        } else {
            void *ptr = (void *)malloc(sizeof(float) * output_channel);
            memset(ptr, 0, sizeof(float) * output_channel);
            free_list.push_back(ptr);

            std::vector<int> dims = {output_channel};
            auto rk_bias          = RknpuUtils::CreateRknnTensor(graph_, layer_name_ + "_bias", dims, ptr,
                                                        rk::nn::TensorRole::CONST, DATA_TYPE_FLOAT);
            inputs.push_back(rk_bias);
        }

        // output
        ADD_OUTPUT_OP();

        int32_t multiplier = 0;
        if (group == 1) {
            multiplier = 0;
        } else if (in_group == 1) {
            multiplier = output_channel / group;
        } else {
            multiplier = 0;
        }

        rk::nn::Conv2DAttr attr;
        attr.ksize[0]    = kernel_w;
        attr.ksize[1]    = kernel_h;
        attr.stride[0]   = stride_w;
        attr.stride[1]   = stride_h;
        attr.pad[0]      = pad_w_begin;
        attr.pad[1]      = pad_w_end;
        attr.pad[2]      = pad_h_begin;
        attr.pad[3]      = pad_h_end;
        attr.group       = group;
        attr.weights     = output_channel;  // TODO
        attr.dilation[0] = 1;
        attr.dilation[1] = 1;
        attr.pad_type    = rk_pad_type;
        attr.multiplier  = multiplier;  // TODO
        attr.has_relu    = (activation_type == ActivationType_ReLU) ? true : false;

        graph_->AddOperator(rk::nn::OperatorType::CONV2D, inputs, output_ops_, (void *)&attr);

        return ret;
    }
};

REGISTER_RKNPU_LAYER(Conv, LAYER_CONVOLUTION)

}  // namespace TNN_NS
