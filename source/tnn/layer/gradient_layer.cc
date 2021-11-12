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

#include "tnn/layer/gradient_layer.h"

#include <cmath>

#include "tnn/layer/base_layer.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

GradientLayer::GradientLayer(LayerType ignore) : BaseLayer(LAYER_GRADIENT) {}

GradientLayer::~GradientLayer() {}

const std::vector<std::pair<Blob*, Blob*>>& GradientLayer::GetBlobGradPairs() {
    return forward_blob_to_grad_;
}

const std::vector<std::pair<RawBuffer*, Blob*>>& GradientLayer::GetResourceGradPairs() {
    return resource_to_grad_;
}

Status GradientLayer::SetAccumulateBlobGradFlag(int index, bool cond) {
    if (index >= accumulate_blob_grad_.size()) {
        LOGE("Error, blob index exceeds %d\n", index);
        return Status(TNNERR_LAYER_ERR, "set blob accumulate flag error");
    }
    accumulate_blob_grad_[index] = cond;

    return TNN_OK;
}

Status GradientLayer::SetAccumulateResourceGradFlag(int index, bool cond) {
    if (index >= accumulate_resource_grad_.size()) {
        LOGE("Error, resource index exceeds %d\n", index);
        return Status(TNNERR_LAYER_ERR, "set resource accumulate flag error");
    }
    accumulate_resource_grad_[index] = cond;

    return TNN_OK;
}

Status GradientLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);

    GradientParam* grad_param = dynamic_cast<GradientParam*>(param_);
    CHECK_PARAM_NULL(grad_param);

    int resource_grad_count = resource_ ? resource_->GetTrainable().size() : 0;

    int blob_grad_count = output_blobs_.size() - resource_grad_count;
    if (blob_grad_count < 0) {
        LOGE("GradientLayer::InferOutputShape, output blob should not be less than resource grad\n");
        return Status(TNNERR_LAYER_ERR, "output blob less than resource grad");
    }

    int grad_index = input_blobs_.size() - blob_grad_count;
    if (grad_index < 0) {
        LOGE("GradientLayer::InferOutputShape, input blob should not be less than blob grad\n");
        return Status(TNNERR_LAYER_ERR, "input blob less than output blob");
    }

    for (int i = 0; i < blob_grad_count; ++i) {
        Blob* forward_input_blob             = input_blobs_[i + grad_index];
        output_blobs_[i]->GetBlobDesc().dims = forward_input_blob->GetBlobDesc().dims;
        forward_blob_to_grad_.push_back({forward_input_blob, output_blobs_[i]});
        accumulate_blob_grad_.push_back(false);
    }

    for (int i = blob_grad_count; i < output_blobs_.size(); ++i) {
        auto trainable_buffer = resource_->GetTrainable()[i - blob_grad_count];
        // resouce buffer dims is empty, use data count
        output_blobs_[i]->GetBlobDesc().dims = {1, trainable_buffer->GetDataCount()};
        resource_to_grad_.push_back({trainable_buffer, output_blobs_[i]});
        accumulate_resource_grad_.push_back(false);
    }

    return TNN_OK;
}

REGISTER_LAYER(Gradient, LAYER_GRADIENT);

}  // namespace TNN_NS
