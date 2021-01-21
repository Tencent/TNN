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

#include <cmath>
#include "tnn/layer/base_layer.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_LAYER_WITH_FUNC(Upsample, LAYER_UPSAMPLE,
                        virtual Status FillLayerParamWithConstantResource(););

Status UpsampleLayer::InferOutputDataType() {
    BaseLayer::InferOutputDataType();
    auto layer_param = dynamic_cast<UpsampleLayerParam *>(param_);

    if (layer_param->scales.empty() && runtime_model_ == RUNTIME_MODE_CONST_FOLD) {
        for (auto &iter : output_blobs_) {
            int allocat_status = DATA_FLAG_ALLOCATE_IN_FORWARD;
            iter->flag         = iter->flag | allocat_status;
        }
    }
    return TNN_OK;
}

Status UpsampleLayer::InferOutputShape(bool ignore_error) {
    auto status = BaseLayer::InferOutputShape(ignore_error);
    RETURN_ON_NEQ(status, TNN_OK);
    
    auto layer_param = dynamic_cast<UpsampleLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    auto scales = layer_param->scales;
    auto sizes = layer_param->dims;
    
    if (sizes.size() <= 0 && scales.size() >= 2) {
        // width_scale height_scale
        float w_scale = scales[scales.size() - 1];
        float h_scale = scales[scales.size() - 2];
        
        if (layer_param->align_corners < 0) {
            if (w_scale >= 1.0f && h_scale >= 1.0f) {
                layer_param->align_corners = 0;
            } else {
                layer_param->align_corners = 1;
            }
        }
    }
    
    auto input_dims = input_blobs_[0]->GetBlobDesc().dims;
    auto output_dims = DimsVectorUtils::Upsample(input_dims, scales, sizes, layer_param->mode, &status);
    RETURN_ON_NEQ(status, TNN_OK);
    
    output_blobs_[0]->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

Status UpsampleLayer::FillLayerParamWithConstantResource() {
    Status status = TNN_OK;
    auto *layer_param = dynamic_cast<UpsampleLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    if (input_blobs_.size() > 1) {
        //fill param with inputs
        std::vector<float> scales;
        std::vector<int> sizes;
        std::shared_ptr<RawBuffer> scales_buffer = nullptr;
        std::shared_ptr<RawBuffer> sizes_buffer = nullptr;
        if (input_blobs_.size() == 2) {
            const auto scales_name = input_blobs_[1]->GetBlobDesc().name;
            if (const_resource_ != nullptr && const_resource_->find(scales_name) != const_resource_->end()) {
                scales_buffer = (*const_resource_)[scales_name];
            }
        } else if (input_blobs_.size() == 3) {
            const auto scales_name = input_blobs_[2]->GetBlobDesc().name;
            if (const_resource_ != nullptr && const_resource_->find(scales_name) != const_resource_->end()) {
                scales_buffer = (*const_resource_)[scales_name];
            }
        } else if (input_blobs_.size() == 4) {
            const auto sizes_name = input_blobs_[3]->GetBlobDesc().name;
            if (const_resource_ != nullptr && const_resource_->find(sizes_name) != const_resource_->end()) {
                sizes_buffer = (*const_resource_)[sizes_name];
            }
        }
        
        if (scales_buffer && scales_buffer->GetBytesSize() > 0) {
            auto scales_data   = scales_buffer->force_to<float *>();
            auto scales_count  = scales_buffer->GetDataCount();
            if (scales_count < 2) {
                LOGE("Error: Upsample has invalid scales count:%d", scales_count);
                return Status(TNNERR_PARAM_ERR, "Error: Upsample has invalid scales count");
            }
            for (int i = 0; i < scales_count; ++i) {
                scales.push_back(scales_data[i]);
            }
            // width_scale height_scale
            float w_scale = scales[scales.size() - 1];
            float h_scale = scales[scales.size() - 2];
            scales = {w_scale, h_scale};
            layer_param->scales = scales;
        }
        
        if (sizes_buffer && sizes_buffer->GetBytesSize() > 0) {
            auto sizes_data   = sizes_buffer->force_to<int *>();
            auto sizes_count  = sizes_buffer->GetDataCount();
            if (sizes_count < 2) {
                LOGE("Error: Upsample has invalid sizes count:%d", sizes_count);
                return Status(TNNERR_PARAM_ERR, "Error: Upsample has invalid scales count");
            }
            for (int i = 0; i < sizes_count; ++i) {
                sizes.push_back(sizes_data[i]);
            }
            // width_scale height_scale
            int w_size = sizes[sizes.size() - 1];
            int h_size = sizes[sizes.size() - 2];
            sizes = {w_size, h_size};
            layer_param->dims = sizes;
        }
    }
    return status;
}

REGISTER_LAYER(Upsample, LAYER_UPSAMPLE);

}  // namespace TNN_NS
