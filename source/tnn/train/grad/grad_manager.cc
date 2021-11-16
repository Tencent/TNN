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

// author: sanerzheng@tencent.com

#include "tnn/train/grad/grad_manager.h"
#include "tnn/core/default_network.h"
#include "tnn/train/grad/layer_grad.h"
#include "tnn/train/grad/utils.h"
#include "tnn/utils/blob_transfer_utils.h"

namespace TNN_NS {
namespace train {
GradManager::GradManager(AbstractNetwork *network, NetworkConfig *config) {
    context_.network = network;
    context_.config  = config;
};
void GradManager::SetNeedGradLayers(const std::set<std::string> &need_grad_layers) {
    need_grad_layers_ = need_grad_layers;
}
void GradManager::SetLossName(const std::string &loss_name) {
    loss_name_ = loss_name;
}

Status GradManager::IsSupport() {
    if (context_.config->device_type != DEVICE_ARM && context_.config->device_type != DEVICE_NAIVE)
        return Status(TNN_TRAIN_ERROR, "only support device arm or device naive for now");
    if (context_.config->train_config.run_mode != TRAIN_MODE_TRAIN)
        return Status(TNN_TRAIN_ERROR, "not in train mode");
    if (context_.config->precision != PRECISION_HIGH)
        return Status(TNN_TRAIN_ERROR, "only support high precision in train mode");
    return TNN_OK;
}
// store grads with Rawbuffer not blob to avoid device difference
// now trainable varibles are always resource, maybe blob input can be a trainable in future
Status GradManager::CalcuteGrads() {
    DefaultNetwork *network = dynamic_cast<DefaultNetwork *>(context_.network);
    if (network == nullptr) {
        return Status(TNNERR_UNSUPPORT_NET, "grad module only support default net work");
    }
    Blob *loss = network->GetBlob(loss_name_);
    if (!loss) {
        return Status(TNN_TRAIN_ERROR, "can't find loss blob");
    }
    auto loss_dims = loss->GetBlobDesc().dims;
    int loss_count = DimsVectorUtils::Count(loss_dims);
    if (loss_dims.empty() || loss_count != 1 || loss->GetBlobDesc().data_type != DATA_TYPE_FLOAT) {
        return Status(TNNERR_INVALID_DATA, "dims size of loss must 1 and data_type must be float");
    }
    // TODO:set grads to 0 with memset to avoid memory release and alloc when resize every train step
    context_.backward_grads_blob.clear();
    context_.backward_grads_resource.clear();
    auto loss_raw_buffer = std::make_shared<RawBuffer>(CalculateElementCount(loss->GetBlobDesc()), loss_dims);
    loss_raw_buffer->force_to<float *>()[0] = 1.f;
    loss_raw_buffer->SetTrainable(false);
    loss_raw_buffer->SetDataFormat(loss->GetBlobDesc().data_format);
    loss_raw_buffer->SetDataType(loss->GetBlobDesc().data_type);
    context_.backward_grads_blob[loss] = loss_raw_buffer;
    auto &layers                       = network->GetLayers();
    Status status;
    for (auto iter = layers.rbegin(); iter != layers.rend(); iter++) {
        if (need_grad_layers_.find((*iter)->layer_name_) == need_grad_layers_.end())
            continue;
        // LOGI("calcute layer grad: %s, layer_type: %d", (*iter)->layer_name_.c_str(), (*iter)->type_);
        status = LayerGrad::GetLayerGradMap()[(*iter)->GetLayerType()]->OnGrad(*iter, context_);
        RETURN_ON_NEQ(status, TNN_OK);
    }
    return status;
}

} // namespace train
} // namespace TNN_NS