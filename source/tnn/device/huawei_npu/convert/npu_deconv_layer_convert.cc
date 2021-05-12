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
        Status ret = ObtainParam();
        if (ret != TNN_OK) {
            return ret;
        }
        auto resource = dynamic_cast<ConvLayerResource *>(resource_);
        if (!resource) {
            return Status(TNNERR_MODEL_ERR, "Error: DeConvLayerResource is empty");
        }

        if (!(NpuUtils::VersionCompare(npu_version_, "100.320.xxx.xxx", VCT_BIGEQUAL) &&
              ((NpuUtils::VersionCompare(npu_version_, "100.320.010.023", VCT_BIGEQUAL) &&
                NpuUtils::VersionCompare(npu_version_, "100.320.010.999", VCT_SMALLER)) ||
               (NpuUtils::VersionCompare(npu_version_, "100.320.011.019", VCT_BIGEQUAL) &&
                NpuUtils::VersionCompare(npu_version_, "100.320.011.999", VCT_SMALLER)) ||
               (NpuUtils::VersionCompare(npu_version_, "100.320.012.011", VCT_BIGEQUAL) &&
                NpuUtils::VersionCompare(npu_version_, "100.320.012.999", VCT_SMALLER))))) {
            if (resource->bias_handle.GetDataCount() > 0) {
                LOGE("Current IR deconv does not support bias (npu version: %s)\n", npu_version_.c_str());
                return Status(TNNERR_LAYER_ERR, "Error: Current IR deconv does not support bias");
            }
        }

        const int input_channel = input_ops_[0]->GetShape()[1];

        // filter
        int filter_channel = (resource->filter_handle.GetDataCount() / (kernel_h_ * kernel_w_ * input_channel));
        ge::Shape filter_shape({input_channel, filter_channel, kernel_h_, kernel_w_});
        auto filter_const = std::make_shared<ge::op::Const>(layer_name_ + "filter");
        NpuUtils::CreateAttrValue(filter_const, filter_shape, resource->filter_handle);
        weight_ops_.push_back(filter_const);

        // bias
        int bias_count  = resource->bias_handle.GetDataCount();
        auto bias_const = std::make_shared<ge::op::Const>(layer_name_ + "_bias");
        if (bias_count != 0) {
            // bias
            ge::Shape bias_shape({1, bias_count, 1, 1});
            NpuUtils::CreateAttrValue(bias_const, bias_shape, resource->bias_handle);
            weight_ops_.push_back(bias_const);
        }

        std::string deconv_op_name = outputs_name_[0];
        if (activation_type_ != ActivationType_None) {
            deconv_op_name = layer_name_ + "_deconv_op";
        }
        auto deconv_op = std::make_shared<hiai::op::ConvTranspose>(deconv_op_name);
        deconv_op->set_input_filter(*filter_const);
        if (bias_count != 0) {
            deconv_op->set_input_bias(*bias_const);
        }
        deconv_op->set_input_x(*input_ops_[0]->GetOperator());
        deconv_op->set_attr_groups(group_);
        if (0 == pad_type_ || 3 == pad_type_) {
            deconv_op->set_attr_pad_mode("SAME");
            deconv_op->set_attr_pads(ge::AttrValue::LIST_INT({0, 0, 0, 0}));
        } else if (1 == pad_type_) {
            deconv_op->set_attr_pad_mode("VALID");
            deconv_op->set_attr_pads(ge::AttrValue::LIST_INT({0, 0, 0, 0}));
        } else {
            deconv_op->set_attr_pad_mode("SPECIFIC");
            deconv_op->set_attr_pads(ge::AttrValue::LIST_INT({pad_h_begin_, pad_h_end_, pad_w_begin_, pad_w_end_}));
        }
        deconv_op->set_attr_strides(ge::AttrValue::LIST_INT({stride_h_, stride_w_}));
        deconv_op->set_attr_dilations(ge::AttrValue::LIST_INT({dilation_h_, dilation_w_}));

        if (activation_type_ == ActivationType_None) {
            ADD_OUTPUT_OP(deconv_op)
        } else {
            weight_ops_.push_back(deconv_op);

            if (activation_type_ == ActivationType_ReLU) {
                auto activation_op = std::make_shared<hiai::op::Activation>(outputs_name_[0]);
                activation_op->set_input_x(*deconv_op);
                activation_op->set_attr_mode(1);
                ADD_OUTPUT_OP(activation_op)
            } else if (activation_type_ == ActivationType_ReLU6) {
                auto activation_op = std::make_shared<hiai::op::Activation>(outputs_name_[0]);
                activation_op->set_input_x(*deconv_op);
                activation_op->set_attr_mode(14);
                ADD_OUTPUT_OP(activation_op)
            } else if (activation_type_ == ActivationType_SIGMOID_MUL) {
                auto activation_op = std::make_shared<hiai::op::Activation>(layer_name_ + "_sigmoid");
                activation_op->set_input_x(*deconv_op);
                activation_op->set_attr_mode(0);
                weight_ops_.push_back(activation_op);

                auto mul_op = std::make_shared<hiai::op::Mul>(outputs_name_[0]);
                mul_op->set_input_x1(*deconv_op);
                mul_op->set_input_x2(*activation_op);
                ADD_OUTPUT_OP(mul_op)
            } else {
                LOGE("the deconvolution activation type (%d) is not support in HUAWEI_NPU yet\n", activation_type_);
                return Status(TNNERR_PARAM_ERR, "the deconvolution activation type is not support in HUAWEI_NPU yet");
            }
        }
    }
};
REGISTER_NPU_LAYER(Deconv, LAYER_DECONVOLUTION)

}  // namespace TNN_NS
