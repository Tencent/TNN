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

#include "tnn/layer/multidir_broadcast_layer.h"

#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

static Status GetBroadcastType(DimsVector input, DimsVector output, int &type) {
    int input_count = DimsVectorUtils::Count(input, 1);
    output          = DimsVectorUtils::Max(input, output);

    if (DimsVectorUtils::Count(input) == DimsVectorUtils::Count(output)) {
        type = BroadcastTypeNormal;
    } else {
        if (input_count == 1) {
            type = BroadcastTypeSingle;
        } else if (input_count == output[1]) {
            type = BroadcastTypeChannel;
        } else if (input_count == DimsVectorUtils::Count(output, 1)) {
            type = BroadcastTypeElement;
        } else if (input_count == DimsVectorUtils::Count(output, 2)) {
            type = BroadcastTypeHeightWidth;
        } else if (input_count == DimsVectorUtils::Count(output, 3)) {
            type = BroadcastTypeWidth;
        } else {
            type = BroadcastTypeGeneral;
        }
    }
    return TNN_OK;
}

void EXPAND(DimsVector &dim0, DimsVector &dim1) {
    if (dim0.size() < dim1.size()) {
        // dim0 < dim1
        size_t diff = dim1.size() - dim0.size();
        for (int i = 0; i < diff; ++i) {
            dim0.insert(dim0.begin(), 1);
        }
    } else {
        // dim0 > dim1
        size_t diff = dim0.size() - dim1.size();
        for (int i = 0; i < diff; ++i) {
            dim1.insert(dim1.begin(), 1);
        }
    }
}

bool SupportBroadcast(DimsVector dim0, DimsVector dim1) {
    if (dim0.size() != dim1.size()) {
        EXPAND(dim0, dim1);
    }
    ASSERT(dim0.size() == dim1.size());

    for (int i = 0; i < dim0.size(); ++i) {
        // the "!" is key point
        if (!(dim0[i] == 1 || dim1[i] == 1 || dim1[i] == dim0[i])) {
            return false;
        }
    }
    return true;
}

Status MultidirBroadcastLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);

    auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    auto layer_res = dynamic_cast<EltwiseLayerResource *>(resource_);

    if (layer_res) {
        const int weight_input_index = layer_param->weight_input_index;
        if (weight_input_index != 0 && weight_input_index != 1) {
            LOGE_IF(!ignore_error, "Error: unsupported weight_input_index\n");
            return Status(TNNERR_LAYER_ERR, "Error: unsupported weight_input_index");
        }

        DimsVector input_shape = input_blobs_[0]->GetBlobDesc().dims;
        int input_count        = DimsVectorUtils::Count(input_shape,1);// 这个地方好像有问题

        DimsVector weight_shape = layer_res->element_handle.GetBufferDims();
        if (input_shape.empty() && weight_shape.empty()) {
            output_blobs_[0]->GetBlobDesc().dims = input_shape;
            return TNN_OK;
        }
        if (weight_shape.size() <= 0) {
            weight_shape       = DimsVector(input_shape.size(), 1);
            int layer_res_size = layer_res->element_handle.GetDataCount();
            if (layer_res_size == 1) {
                // single element
                weight_shape[1] = layer_res_size;
            } else if (layer_res_size == input_shape[1]) {
                // channel broadcast
                weight_shape[1] = layer_res_size;
            } else if (layer_res_size == input_count) {
                // element broadcast
                weight_shape[1] = input_shape[1];
                weight_shape[2] = input_shape[2];
                weight_shape[3] = input_shape[3];
            } else if (layer_res_size == input_shape[3]) {
                weight_shape[3] = input_shape[3];
            } else if (layer_res_size == DimsVectorUtils::Count(input_shape, 2)) {
                for (int i = 2; i < input_shape.size(); ++i) {
                    weight_shape[i] = input_shape[i];
                }
            } else {
                LOGE_IF(!ignore_error, "Error: unsupported broadcast type\n");
                return Status(TNNERR_LAYER_ERR, "Error: unsupported broadcast type");
            }
            layer_res->element_shape = weight_shape;
        } else {
            layer_res->element_shape = weight_shape;
        }
        EXPAND(input_shape, weight_shape);
        DimsVector dims_output               = DimsVectorUtils::Max(input_shape, weight_shape);
        output_blobs_[0]->GetBlobDesc().dims = dims_output;

        int input_broadcast_type  = BroadcastTypeNormal;
        int weight_broadcast_type = BroadcastTypeNormal;
        auto status               = GetBroadcastType(input_shape, dims_output, input_broadcast_type);
        if (status != TNN_OK) {
            return status;
        }
        status = GetBroadcastType(weight_shape, dims_output, weight_broadcast_type);
        if (status != TNN_OK) {
            return status;
        }

        if (weight_input_index == 0) {
            layer_param->input0_broadcast_type = weight_broadcast_type;
            layer_param->input1_broadcast_type = input_broadcast_type;
        } else {
            layer_param->input0_broadcast_type = input_broadcast_type;
            layer_param->input1_broadcast_type = weight_broadcast_type;
        }
    } else {
        DimsVector dim0 = input_blobs_[0]->GetBlobDesc().dims;
        DimsVector dim1 = dim0;
        if (input_blobs_.size() > 1) {
            dim1 = input_blobs_[1]->GetBlobDesc().dims;
        }

        if (!SupportBroadcast(dim0, dim1)) {
            LOGE_IF(!ignore_error, 
                "Error: operands could not be broadcast together with wrong "
                "shape (name: %s)\n", layer_param->name.c_str());
            return Status(TNNERR_LAYER_ERR,
                          "Error: operands could not be broadcast together "
                          "with wrong shape");
        }
        auto dims_output = dim0;
        for (auto iter : input_blobs_) {
            auto tmp    = iter->GetBlobDesc().dims;
            dims_output = DimsVectorUtils::Max(tmp, dims_output);
        }
        output_blobs_[0]->GetBlobDesc().dims = dims_output;

        int input0_broadcast_type = BroadcastTypeNormal;
        int input1_broadcast_type = BroadcastTypeNormal;
        auto status               = GetBroadcastType(dim0, dims_output, input0_broadcast_type);
        if (status != TNN_OK) {
            return status;
        }
        status = GetBroadcastType(dim1, dims_output, input1_broadcast_type);
        if (status != TNN_OK) {
            return status;
        }

        layer_param->input0_broadcast_type = input0_broadcast_type;
        layer_param->input1_broadcast_type = input1_broadcast_type;
    }

//    LOGD("broadcast_type: input0(%d) input1(%d)\n", layer_param->input0_broadcast_type,
//         layer_param->input1_broadcast_type);

    return TNN_OK;
}

}  // namespace TNN_NS
