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

class NpuDeconvLayer : public NpuConvImplLayer {
public:
    NpuDeconvLayer(LayerType ignore) : NpuConvImplLayer(LAYER_DECONVOLUTION){};
    virtual ~NpuDeconvLayer() {}

protected:
    virtual Status Convert() {
        Status ret    = ObtainParam();
        auto resource = dynamic_cast<ConvLayerResource *>(resource_);
        if (ret != TNN_OK || !resource) {
            return Status(TNNERR_MODEL_ERR, "Error: ConvLayerParam or ConvLayerResource is empty");
        }
        auto input_data         = (input_ops_[0]);
        const int input_channel = input_data->GetShape()[1];

        int pad_mode = 0;
        ret          = NpuUtils::GetPadMode(pad_mode, pad_type, false);
        if (ret != TNN_OK)
            return ret;

        // filter
        int filter_channel = (resource->filter_handle.GetDataCount() / (kernel_h * kernel_w * input_channel));
        ge::Shape filter_shape({input_channel, filter_channel, kernel_h, kernel_w});
        auto filter_const = std::make_shared<ge::op::Const>(layer_name_ + "filter");
        NpuUtils::CreateAttrValue(filter_const, filter_shape, resource->filter_handle);
        weight_ops_.push_back(filter_const);

        std::vector<int> calculate_shape = NpuBaseLayer::GetOutputShape(0);

        // input size
        std::shared_ptr<ge::op::Const> input_size_const = std::make_shared<ge::op::Const>(layer_name_ + "_input_size");
        ge::TensorDesc desc(ge::Shape({4}), ge::FORMAT_NCHW, ge::DT_UINT8);
        NpuUtils::CreateAttrArray(input_size_const, calculate_shape, desc);
        weight_ops_.push_back(input_size_const);

        auto output = std::make_shared<ge::op::Deconvolution>(outputs_[0]);
        output->set_input_filter(*filter_const);
        output->set_input_input_sizes(*input_size_const);
        output->set_input_x(*input_data->GetOperator());
        output->set_attr_group(group);
        output->set_attr_num_output(output_channel);
        output->set_attr_pad(ge::AttrValue::LIST_INT({
            pad_h_begin,
            pad_h_end,
            pad_w_begin,
            pad_w_end,
        }));
        output->set_attr_pad_mode(pad_mode);
        output->set_attr_stride(ge::AttrValue::LIST_INT({stride_h, stride_w}));
        output->set_attr_dilation(ge::AttrValue::LIST_INT({dilation_h, dilation_w}));
        output->set_attr_kernel(ge::AttrValue::LIST_INT({kernel_h, kernel_w}));

        std::shared_ptr<OperatorInfo> output_op = std::make_shared<OperatorInfo>(output, calculate_shape);
        output_ops_.push_back(output_op);
        return TNN_OK;
    }
};

REGISTER_NPU_LAYER(Deconv, LAYER_DECONVOLUTION);

}  // namespace TNN_NS
