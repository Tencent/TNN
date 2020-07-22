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

#include "graph/attr_value.h"
#include "graph/op/all_ops.h"
#include "graph/op/nn_defs.h"
#include "npu_base_layer_convert.h"
#include "npu_conv_layer_convert_impl.h"
#include "npu_utils.h"

namespace TNN_NS {

class NpuConvLayer : public NpuConvImplLayer {
public:
    NpuConvLayer(LayerType ignore) : NpuConvImplLayer(LAYER_CONVOLUTION){};
    virtual ~NpuConvLayer() {}

protected:
    virtual Status Convert() {
        Status ret    = ObtainParam();
        auto resource = dynamic_cast<ConvLayerResource *>(resource_);
        if (ret != TNN_OK){
           return ret;
        }
        if (!resource) {
            return Status(TNNERR_MODEL_ERR, "Error: ConvLayerResource is empty");
        }

        // pad mode
        int pad_mode = 0;
        ret          = NpuUtils::GetPadMode(pad_mode, pad_type);
        if (ret != TNN_OK)
            return ret;

        // weight
        int total_data_size = resource->filter_handle.GetDataCount();
        int in_group        = total_data_size / (kernel_h * kernel_w * output_channel);
        ge::Shape weight_shape({output_channel, in_group, kernel_h, kernel_w});
        auto weight_const = std::make_shared<ge::op::Const>(layer_name_ + "_weight");
        NpuUtils::CreateAttrValue(weight_const, weight_shape, resource->filter_handle);
        weight_ops_.push_back(weight_const);

        auto output = std::make_shared<ge::op::Convolution>(outputs_name_[0]);
        output->set_input_x(*input_ops_[0]->GetOperator());
        output->set_input_w(*weight_const);
        // Init bias
        int bias_count = resource->bias_handle.GetDataCount();
        // check bias
        if (bias_count != 0) {
            // bias
            ge::Shape bias_shape({1, bias_count, 1, 1});
            auto bias_const = std::make_shared<ge::op::Const>(layer_name_ + "_bias");
            NpuUtils::CreateAttrValue(bias_const, bias_shape, resource->bias_handle);
            weight_ops_.push_back(bias_const);
            output->set_input_b(*bias_const);
        }
        output->set_attr_kernel(ge::AttrValue::LIST_INT({kernel_h, kernel_w}));
        output->set_attr_stride(ge::AttrValue::LIST_INT({stride_h, stride_w}));
        output->set_attr_dilation(ge::AttrValue::LIST_INT({dilation_h, dilation_w}));
        output->set_attr_group(group);
        output->set_attr_pad(ge::AttrValue::LIST_INT({pad_h_begin, pad_h_end, pad_w_begin, pad_w_end}));
        output->set_attr_pad_mode(pad_mode);
        output->set_attr_num_output(output_channel);

        ADD_OUTPUT_OP(output)
    }
};

REGISTER_NPU_LAYER(Conv, LAYER_CONVOLUTION)

}  // namespace TNN_NS
