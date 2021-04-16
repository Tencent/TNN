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

DECLARE_NPU_LAYER_WEIGHT(Reshape, LAYER_RESHAPE)

static Status ConvertShapeFromTNNToTFLite(std::vector<int>& shape) {
    if (shape.empty()) {
        LOGE("(Reshape) param->shape is empty\n");
        return Status(TNNERR_PARAM_ERR, "(Reshape) param->shape is empty");
    }

    if (shape.size() < 3) {
        return TNN_OK;
    } else if (shape.size() == 3) {
        auto c   = shape[1];
        auto h   = shape[2];
        shape[1] = h;
        shape[2] = c;
    } else if (shape.size() == 4) {
        auto c   = shape[1];
        auto h   = shape[2];
        auto w   = shape[3];
        shape[1] = h;
        shape[2] = w;
        shape[3] = c;
    } else {
        LOGE("(Reshape) param->shape is invalid for HUAWEI_NPU\n");
        return Status(TNNERR_PARAM_ERR, "(Reshape) param->shape is invalid for HUAWEI_NPU");
    }

    return TNN_OK;
}

static Status GetPermuteOrder(std::vector<int64_t>& order, int dims_size, bool to_tflite) {
    order.clear();
    if (dims_size < 3) {
        for (int i = 0; i < dims_size; ++i) {
            order.push_back(i);
        }
    } else if (dims_size == 3) {
        if (to_tflite) {
            // from nch to nhc
            order = {0, 2, 1};
        } else {
            // from nhc to nch
            order = {0, 2, 1};
        }
    } else if (dims_size == 4) {
        if (to_tflite) {
            // from nchw to nhwc
            order = {0, 2, 3, 1};
        } else {
            // from nhwc to nchw
            order = {0, 3, 1, 2};
        }
    } else {
        LOGE("(Reshape) is not support input dims > 4 HUAWEI_NPU\n");
        return Status(TNNERR_PARAM_ERR, "(Reshape) is not support input dims > 4 HUAWEI_NPU");
    }

    return TNN_OK;
}

static Status InferShapeFromZero(std::vector<int> input_dims, std::vector<int>& shape) {
    int infer_max_idx = (int)input_dims.size() - 1;

    for (int i = 0; i < shape.size(); ++i) {
        if (shape[i] == 0) {
            if (i > infer_max_idx) {
                LOGE("Reshape param is invalid, 0 in shape (pos: %d) can't be infered\n", i);
                return Status(TNNERR_PARAM_ERR, "Reshape param is invalid, 0 in shape can't be infered");
            }
            if (input_dims[i] != 0) {
                shape[i] = input_dims[i];
            }
        }
    }

    return TNN_OK;
}

Status NpuReshapeLayer::Convert() {
    auto param = dynamic_cast<ReshapeLayerParam*>(param_);
    CHECK_PARAM_NULL(param);

    auto shape = param->shape;
    if (shape.size() > 4) {
        LOGE("(Reshape) dims size bigger than 4 is not support in HUAWEI_NPU\n");
        return Status(TNNERR_MODEL_ERR, "(Reshape) dims size bigger than 4 is not support in HUAWEI_NPU");
    }

    // infer shape to avoid the suitation: 0 exist in the shape position which is bigger than input.size(),
    // this suitation exists in TFLite type Reshape
    RETURN_ON_NEQ(InferShapeFromZero(input_ops_[0]->GetShape(), shape), TNN_OK);

    if (param->reshape_type == 1) {
        RETURN_ON_NEQ(ConvertShapeFromTNNToTFLite(shape), TNN_OK);
    }

    std::shared_ptr<ge::op::Const> shape_const = std::make_shared<ge::op::Const>(layer_name_ + "_shape");
    ge::TensorDesc shape_desc(ge::Shape({(int64_t)shape.size()}), ge::FORMAT_NCHW, ge::DT_INT32);
    NpuUtils::CreateAttrArray(shape_const, shape, shape_desc, (int)shape.size());
    weight_ops_.push_back(shape_const);

    if (param->reshape_type == 0) {
        // onnx caffe reshape(nchw): 0
        auto output = std::make_shared<hiai::op::Reshape>(outputs_name_[0]);
        output->set_input_x(*input_ops_[0]->GetOperator());
        output->set_input_shape(*shape_const);
        output->set_attr_axis(param->axis);
        ADD_OUTPUT_OP(output)
    } else {
        LOGE("TFLite type Reshape is not support in HUAWEI_NPU\n");
        return Status(TNNERR_MODEL_ERR, "TFLite type Reshape is not support in HUAWEI_NPU");

        //std::vector<int64_t> order;
        //// Tensorflow TFLite reshape(nhwc): 1
        //// convert input form nchw to nhwc first
        //auto permute_op = std::make_shared<hiai::op::Permute>(layer_name_ + "permute");
        //permute_op->set_input_x(*input_ops_[0]->GetOperator());
        //RETURN_ON_NEQ(GetPermuteOrder(order, input_ops_[0]->GetShape().size(), true), TNN_OK);
        //permute_op->set_attr_order(order);
        //weight_ops_.push_back(permute_op);

        //// do reshape
        //auto reshape_op = std::make_shared<hiai::op::Reshape>(layer_name_ + "reshape");
        //reshape_op->set_input_x(*permute_op);
        //reshape_op->set_input_shape(*shape_const);
        //reshape_op->set_attr_axis(param->axis);
        //weight_ops_.push_back(reshape_op);

        //// convert input form nhwc to nchw
        //auto output = std::make_shared<hiai::op::Permute>(outputs_name_[0]);
        //output->set_input_x(*reshape_op);
        //RETURN_ON_NEQ(GetPermuteOrder(order, shape.size(), false), TNN_OK);
        //output->set_attr_order(order);

        //ADD_OUTPUT_OP(output)
    }
}

REGISTER_NPU_LAYER(Reshape, LAYER_RESHAPE)

}  // namespace TNN_NS
