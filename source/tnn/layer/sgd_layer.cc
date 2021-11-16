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

#include "tnn/layer/sgd_layer.h"

#include <cmath>

#include "tnn/layer/base_layer.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

SGDLayer::SGDLayer(LayerType ignore) : BaseLayer(LAYER_SGD) {}

SGDLayer::~SGDLayer() {}

Status SGDLayer::Init(Context* context, LayerParam* param, LayerResource* resource, std::vector<Blob*>& input_blobs,
                      std::vector<Blob*>& output_blobs, AbstractDevice* device, bool enable_const_folder) {
    RETURN_ON_NEQ(BaseLayer::Init(context, param, resource, input_blobs, output_blobs, device, enable_const_folder),
                  TNN_OK);

    if (!layer_acc_) {
        LOGE("SGDLayer::Init ERROR, layer acc is nil\n");
        return Status(TNNERR_LAYER_ERR, "layer acc is nil");
    }

    layer_acc_->SetLayerGradInfo(&grad_info_);

    return TNN_OK;
}

Status SGDLayer::SetTrainableResources(std::vector<RawBuffer*> trainable) {
    grad_info_.trainable_resources = trainable;

    return TNN_OK;
}

Status SGDLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);

    if (output_blobs_.size() != 1) {
        LOGE("SGDLayer::InferOutputShape, output_blobs size error\n");
        return Status(TNNERR_LAYER_ERR, "output_blobs size error");
    }

    // global step
    output_blobs_[0]->GetBlobDesc().dims = {1};

    return TNN_OK;
}

REGISTER_LAYER(SGD, LAYER_SGD);

}  // namespace TNN_NS
