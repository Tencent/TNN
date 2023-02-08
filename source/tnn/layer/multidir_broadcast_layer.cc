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

#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

void BroadCastTypeFilter(const DimsVector &dims_output, const DimsVector &dims_input, int &type) {

    if (DimsVectorUtils::Equal(dims_output, dims_input)) {
        type = BroadcastTypeNormal;
        return;
    }
    if (DimsVectorUtils::Equal(dims_output, dims_input, 1)) {
        type = BroadcastTypeElement;
        return;
    }
    if (DimsVectorUtils::Equal(dims_output, dims_input, 2)) {
        type = BroadcastTypeHeightWidth;
        return;
    }
    if (DimsVectorUtils::Equal(dims_output, dims_input, 3)) {
        type = BroadcastTypeWidth;
        if (dims_input[1] != 1) {
            type = BroadcastTypeChannelWidth;
        }
        return;
    }
    int broadcast_count = DimsVectorUtils::Count(dims_input);
    if (broadcast_count == 1) {
        type = BroadcastTypeSingle;
    } else if (broadcast_count == dims_output[1]) {
        // broadcast dim = [1, channel, 1...]
        if (dims_input[1] == dims_output[1]) {
            type = BroadcastTypeChannel;
        } else {
            type = BroadcastTypeGeneral;
        }
    } else {
        type = BroadcastTypeGeneral;
    }
    
    return;
}

static Status GetBroadcastType(DimsVector input, DimsVector output, int &type) {
    DimsVector input_pad_shape;
    // support input dims size diff with output dims
    int pad_size = output.size() - input.size();
    while (pad_size-- != 0) {
        input_pad_shape.push_back(1);
    }
    input_pad_shape.insert(input_pad_shape.end(), input.begin(), input.end());

    BroadCastTypeFilter(output, input_pad_shape, type);
    return TNN_OK;
}


// @brief expand 1 at the begin of dim0 or dim1 to the same size, for example [1, 256, 128] and [128] will be expanded to [1, 256, 128] and [1, 1, 128]
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
                for (int i = 1; i < input_shape.size(); ++i) {
                    weight_shape[i] = input_shape[i];
                }
            } else if ((input_shape.size() >= 4) && (layer_res_size == input_shape[3])) {
                weight_shape[3] = input_shape[3];
            } else if (layer_res_size == DimsVectorUtils::Count(input_shape, 2)) {
                for (int i = 2; i < input_shape.size(); ++i) {
                    weight_shape[i] = input_shape[i];
                }
            } else {
                LOGE_IF(!ignore_error, "Error: unsupported broadcast type\n");
                return Status(TNNERR_LAYER_ERR, "Error: unsupported broadcast type");
            }
        }
        EXPAND(input_shape, weight_shape);
        layer_res->element_shape = weight_shape;
        layer_res->element_handle.SetBufferDims(weight_shape);
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
            auto tmp = iter->GetBlobDesc().dims;
            if (dims_output.size() != tmp.size()) {
                EXPAND(dims_output, tmp);
            }
            dims_output = DimsVectorUtils::Max(dims_output, tmp);
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
