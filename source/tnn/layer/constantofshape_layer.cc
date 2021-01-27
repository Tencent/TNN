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
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {
DECLARE_LAYER(ConstantOfShape, LAYER_CONSTANT_OF_SHAPE);

Status ConstantOfShapeLayer::InferOutputDataType() {
    BaseLayer::InferOutputDataType();
    
    auto layer_resource = dynamic_cast<ConstantOfShapeLayerResource*>(resource_);
    CHECK_PARAM_NULL(layer_resource);
    
    for (auto& iter : output_blobs_) {
        int allocate_status = DATA_FLAG_ALLOCATE_IN_FORWARD;
        if (runtime_model_ == RUNTIME_MODE_NORMAL &&
            const_resource_ != nullptr && const_resource_->find(iter->GetBlobDesc().name) != const_resource_->end()) {
            allocate_status = 0;
        }
        iter->flag = DATA_FLAG_CHANGE_IF_SHAPE_DIFFER | allocate_status;
        iter->GetBlobDesc().data_type = layer_resource->value.GetDataType();
    }
    
    return TNN_OK;
}

Status ConstantOfShapeLayer::InferOutputShape(bool ignore_error) {
    //NOTE: This layer should not be excuted on device which is not NAIVE. see RangeLayer
    
    auto status = BaseLayer::InferOutputShape(ignore_error);
    RETURN_ON_NEQ(status, TNN_OK);
    
    auto input_dims = input_blobs_[0]->GetBlobDesc().dims;
    auto data_type = input_blobs_[0]->GetBlobDesc().data_type;
    if (data_type != DATA_TYPE_INT32) {
        return Status(TNNERR_MODEL_ERR, "ConstantOfShapeLayer input blob has invalid data type");
    }
    if (input_blobs_[0]->GetBlobDesc().device_type != DEVICE_NAIVE) {
        return Status(TNNERR_MODEL_ERR, "ConstantOfShapeLayer input blob has invalid device type");
    }
    
    //runtime infer output shape
    {
        auto input_data = (int *)input_blobs_[0]->GetHandle().base;
        auto count = DimsVectorUtils::Count(input_dims);
        if (input_dims.size() <= 0 || input_data==nullptr || count <= 0) {
            return Status(TNNERR_LAYER_ERR, "ConstantOfShape has invalid output dims");
        }
        
        DimsVector output_dims;
        for (int i=0; i<count; i++) {
            output_dims.push_back(input_data[i]);
        }
        output_blobs_[0]->GetBlobDesc().dims = output_dims;
    }
    return TNN_OK;
}

REGISTER_LAYER(ConstantOfShape, LAYER_CONSTANT_OF_SHAPE);

}  // namespace TNN_NS
