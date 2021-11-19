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

#include "tnn/train/gradient/gradient_layer.h"

#include <cmath>

#include "tnn/layer/base_layer.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

GradientLayer::GradientLayer(LayerType ignore) : BaseLayer(LAYER_GRADIENT) {}

GradientLayer::~GradientLayer() {}

Status GradientLayer::Init(Context* context, LayerParam* param, LayerResource* resource,
                           std::vector<Blob*>& input_blobs, std::vector<Blob*>& output_blobs, AbstractDevice* device,
                           bool enable_const_folder) {
    RETURN_ON_NEQ(BaseLayer::Init(context, param, resource, input_blobs, output_blobs, device, enable_const_folder),
                  TNN_OK);

    RETURN_ON_NEQ(InitGradInfo(), TNN_OK);

    if (!layer_acc_) {
        LOGE("GradientLayer::Init ERROR, layer acc is nil\n");
        return Status(TNNERR_LAYER_ERR, "layer acc is nil");
    }

    layer_acc_->SetRuntimeTrainingInfo(&training_info_);

    return TNN_OK;
}

const std::vector<std::pair<Blob*, Blob*>>& GradientLayer::GetInputGradPairs() {
    return input_to_grad_;
}

const std::vector<std::pair<Blob*, RawBuffer*>>& GradientLayer::GetGradResourcePairs() {
    return grad_to_resource_;
}

Status GradientLayer::SetAccumulateInputGradFlag(int index, bool cond) {
    if (index >= training_info_.grad_info.accumulate_input_grad.size()) {
        LOGE("Error, input index exceeds %d\n", index);
        return Status(TNNERR_LAYER_ERR, "set input accumulate flag error");
    }
    training_info_.grad_info.accumulate_input_grad[index] = cond;

    return TNN_OK;
}

Status GradientLayer::SetAccumulateResourceGradFlag(int index, bool cond) {
    if (index >= training_info_.grad_info.accumulate_resource_grad.size()) {
        LOGE("Error, resource index exceeds %d\n", index);
        return Status(TNNERR_LAYER_ERR, "set resource accumulate flag error");
    }
    training_info_.grad_info.accumulate_resource_grad[index] = cond;

    return TNN_OK;
}

/*
input_blobs_:  [inputs, outputs, output_grads]
output_blobs_: [input_grads, resource_grads]
*/
Status GradientLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);

    resource_grad_count_ = resource_ ? resource_->GetTrainable().size() : 0;

    input_grad_count_ = output_blobs_.size() - resource_grad_count_;
    if (input_grad_count_ < 0) {
        LOGE("GradientLayer::InferOutputShape, empty input grad to calculate\n");
        return Status(TNNERR_LAYER_ERR, "empty input grad to calculate");
    }

    for (int i = 0; i < input_grad_count_; ++i) {
        Blob* forward_input_blob             = input_blobs_[i];
        output_blobs_[i]->GetBlobDesc().dims = forward_input_blob->GetBlobDesc().dims;
    }

    for (int i = input_grad_count_; i < output_blobs_.size(); ++i) {
        auto trainable_buffer = resource_->GetTrainable()[i - input_grad_count_];
        // resouce buffer dims is empty, use data count
        output_blobs_[i]->GetBlobDesc().dims = {
            1, trainable_buffer->GetDataCount() > 0 ? trainable_buffer->GetDataCount() : 1};
    }

    return TNN_OK;
}

Status GradientLayer::InitGradInfo() {
    input_to_grad_.clear();
    training_info_.grad_info.accumulate_input_grad.clear();

    for (int i = 0; i < input_grad_count_; ++i) {
        input_to_grad_.push_back({input_blobs_[i], output_blobs_[i]});
        training_info_.grad_info.accumulate_input_grad.push_back(false);
    }

    grad_to_resource_.clear();
    training_info_.grad_info.accumulate_resource_grad.clear();
    for (int i = input_grad_count_; i < output_blobs_.size(); ++i) {
        auto trainable_buffer = resource_->GetTrainable()[i - input_grad_count_];
        grad_to_resource_.push_back({output_blobs_[i], trainable_buffer});
        training_info_.grad_info.accumulate_resource_grad.push_back(false);
    }

    return TNN_OK;
}

REGISTER_LAYER(Gradient, LAYER_GRADIENT);

}  // namespace TNN_NS
