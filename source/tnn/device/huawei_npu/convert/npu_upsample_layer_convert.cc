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
#include "npu_utils.h"

namespace TNN_NS {

DECLARE_NPU_LAYER_WEIGHT(Upsample, LAYER_UPSAMPLE)

Status NpuUpsampleLayer::Convert() {
    auto param = dynamic_cast<UpsampleLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    float scale_w     = param->scales[0];
    float scale_h     = param->scales[1];
    int output_width  = (int)round(input_ops_[0]->GetShape()[3] * scale_w);
    int output_height = (int)round(input_ops_[0]->GetShape()[2] * scale_h);

    if (param->dims.size() >= 2) {
        output_width  = param->dims[0];
        output_height = param->dims[1];
    }

    const int resize_mode     = param->mode;
    const bool align_corners  = param->align_corners == 0 ? false : true;
    std::vector<int> dims_vec = param->dims;

    std::shared_ptr<ge::op::Const> input_size_const = std::make_shared<ge::op::Const>(layer_name_ + "_input_size");
    ge::TensorDesc desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
    NpuUtils::CreateAttrArray(input_size_const, std::vector<int>{output_height, output_width}, desc, 2);
    weight_ops_.push_back(input_size_const);
    if (resize_mode == 1) {
        // nereast
        auto output = std::make_shared<hiai::op::ResizeNearestNeighbor>(outputs_name_[0]);
        output->set_input_x(*input_ops_[0]->GetOperator());
        output->set_input_size(*input_size_const);
        output->set_attr_align_corners(false);
        ADD_OUTPUT_OP(output)

    } else if (resize_mode == 2) {
        // bilinear/linear
        auto output = std::make_shared<hiai::op::ResizeBilinearV2>(outputs_name_[0]);
        output->set_input_x(*input_ops_[0]->GetOperator());
        output->set_input_size(*input_size_const);
        output->set_attr_align_corners(align_corners);
        output->set_attr_half_pixel_centers(!align_corners);
        ADD_OUTPUT_OP(output)
    } else {
        LOGE("the upsample type is not support in huawei NPU\n");
        return Status(TNNERR_NPU_UNSUPPORT_ERROR, "the upsample scale is not support in huawei NPU");
    }
}

REGISTER_NPU_LAYER(Upsample, LAYER_UPSAMPLE)

}  // namespace TNN_NS
