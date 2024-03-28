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

#include "base_layer.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {
DECLARE_LAYER_WITH_FUNC(Linspace, LAYER_LINSPACE,
                        virtual Status FillLayerParamWithConstantResource(););

Status LinspaceLayer::InferOutputDataType() {
    auto status = BaseLayer::InferOutputDataType();
    if (runtime_model_ != RUNTIME_MODE_CONST_FOLD) {
        return status;
    }
    
    auto *layer_param = dynamic_cast<LinspaceLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);

    // Used In TNN torch, when 1-3 of start, end, end is in input_blobs_
    // start_index, end_index, end_index represent their index in input_blobs.
    if (layer_param->start_index != -1 || layer_param->end_index != -1 ||
        layer_param->steps_index != -1) {
        output_blobs_[0]->SetFlag(output_blobs_[0]->GetFlag() | DATA_FLAG_ALLOCATE_IN_FORWARD);
        return status;
    }

    const auto& input_name = input_blobs_[0]->GetBlobDesc().name;
    const auto& const_res  = const_resource_;
    if (const_res != nullptr && const_res->find(input_name) != const_res->end()) {
        output_blobs_[0]->SetFlag(output_blobs_[0]->GetFlag() | DATA_FLAG_ALLOCATE_IN_FORWARD);
    }
    return status;
}

Status LinspaceLayer::InferOutputShape(bool ignore_error) {
    //NOTE: This layer should not be excuted on device which is not NAIVE. see ConstantOfShapeLayer
    
    auto status = BaseLayer::InferOutputShape(ignore_error);
    RETURN_ON_NEQ(status, TNN_OK);
    
    auto *layer_param = dynamic_cast<LinspaceLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    if (input_blobs_[0]->GetBlobDesc().device_type != DEVICE_NAIVE) {
        return Status(TNNERR_MODEL_ERR, "RangeLayer input blob has invalid device type");
    }
    
    auto output_dims = {layer_param->steps.i};
    
    output_blobs_[0]->GetBlobDesc().dims = output_dims;
    
    return TNN_OK;
}

Status LinspaceLayer::FillLayerParamWithConstantResource() {
    Status status = TNN_OK;
    auto *layer_param = dynamic_cast<LinspaceLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    if (input_blobs_.size() != 3) {
        return Status(TNNERR_PARAM_ERR, "RangeLayer has invalid layer param");
    }
    
    // start
    {
        const auto start_name = input_blobs_[0]->GetBlobDesc().name;
        if (const_resource_ != nullptr && const_resource_->find(start_name) != const_resource_->end()) {
            auto start_buffer = (*const_resource_)[start_name];
            layer_param->data_type = start_buffer->GetDataType();
            auto start_data   = start_buffer->force_to<float *>();
            auto start = layer_param->start;
            if (start_buffer->GetDataType() == DATA_TYPE_FLOAT) {
                start.f = *start_data;
            } else if (start_buffer->GetDataType() == DATA_TYPE_INT32) {
                start.i = *((int *)start_data);
            } else {
                return Status(TNNERR_PARAM_ERR, "RangeLayer has invalid start data type");
            }
            layer_param->start = start;
        }
    }
    
    // end
    {
        const auto end_name = input_blobs_[1]->GetBlobDesc().name;
        if (const_resource_ != nullptr && const_resource_->find(end_name) != const_resource_->end()) {
            auto end_buffer = (*const_resource_)[end_name];
            layer_param->data_type = end_buffer->GetDataType();
            auto end_data   = end_buffer->force_to<float *>();
            auto end = layer_param->end;
            if (end_buffer->GetDataType() == DATA_TYPE_FLOAT) {
                end.f = *end_data;
            } else if (end_buffer->GetDataType() == DATA_TYPE_INT32) {
                end.i = *((int *)end_data);
            } else {
                return Status(TNNERR_PARAM_ERR, "RangeLayer has invalid end data type");
            }
            layer_param->end = end;
        }
    }
    
    // steps
    {
        const auto steps_name = input_blobs_[2]->GetBlobDesc().name;
        if (const_resource_ != nullptr && const_resource_->find(steps_name) != const_resource_->end()) {
            auto steps_buffer = (*const_resource_)[steps_name];
            auto steps_data   = steps_buffer->force_to<float *>();
            auto steps = layer_param->steps;
            if (steps_buffer->GetDataType() == DATA_TYPE_INT32) {
                steps.i = *((int *)steps_data);
            } else {
                return Status(TNNERR_PARAM_ERR, "RangeLayer has invalid end data type");
            }
            layer_param->steps = steps;
        }
    }
    return status;
}

REGISTER_LAYER(Linspace, LAYER_LINSPACE);

}  // namespace TNN_NS
