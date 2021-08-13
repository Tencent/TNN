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
#include "tnn/utils/blob_transfer_utils.h"

namespace TNN_NS {
namespace train {
GradManager::GradManager(AbstractNetwork* network, NetworkConfig* config) {
    context_.network = network;
    context_.config = config;
};
inline TrainContext& GradManager::GetContext() {
    return context_;
}
void GradManager::SetNeedGradLayers(const std::set<std::string>& need_grad_layers){
    need_grad_layers_ = need_grad_layers;
}

// store grads with Rawbuffer not blob to avoid device difference 
// now trainable varibles are always resource, maybe blob input can be a trainable in future
Status GradManager::CalcuteGrads() {
    DefaultNetwork* network = dynamic_cast<DefaultNetwork* >(context_.network);
    if(network == nullptr) {
        return Status(TNNERR_UNSUPPORT_NET, "grad module only support default net work");
    }
    Blob* loss = network->GetBlob(context_.config->train_config.loss_layer_name);
    if(loss->GetBlobDesc().dims.empty() || loss->GetBlobDesc().data_type != DATA_TYPE_FLOAT) {
        return Status(TNNERR_INVALID_DATA, "dims size of loss must 1 and data_type must be float");
    }
    // TODO:set grads to 0 with memset to avoid memory release and alloc when resize every train step 
    // TODO:check loss size
    
    context_.backward_grads_blob.clear();
    context_.backward_grads_resource.clear();
    Status status = Blob2RawBuffer(loss, context_.backward_grads_blob[loss]);
    RETURN_ON_NEQ(status, TNN_OK);
    auto& loss_raw_buffer = context_.backward_grads_blob[loss];
    loss_raw_buffer->force_to<float *>()[0] = 1.f;
    loss_raw_buffer->SetTrainable(false);
    loss_raw_buffer->SetDataFormat(loss->GetBlobDesc().data_format);
    auto layers = network->GetLayers();
    for(auto iter = layers.rbegin(); iter != layers.rend(); iter++) {
        if(need_grad_layers_.find((*iter)->layer_name_) == need_grad_layers_.end())
            continue;
        status = LayerGrad::GetLayerGradMap()[(*iter)->GetLayerType()]->OnGrad(*iter, context_);
        RETURN_ON_NEQ(status, TNN_OK);
    }
    return status;
} 
 
}
} //namspace TNN_NS