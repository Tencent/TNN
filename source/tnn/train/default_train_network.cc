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

#include "tnn/train/default_train_network.h"

#include "tnn/train/gradient/gradient_layer.h"
#include "tnn/train/solver/sgd_layer.h"

namespace TNN_NS {

NetworkImplFactoryRegister<NetworkImplFactory<DefaultTrainNetwork>> g_network_impl_default_train_factory_register(
    NETWORK_TYPE_DEFAULT_TRAIN);

DefaultTrainNetwork::DefaultTrainNetwork() {}

DefaultTrainNetwork::~DefaultTrainNetwork() {}

Status DefaultTrainNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config,
                                 AbstractModelInterpreter *interpreter, InputShapesMap min_inputs_shape,
                                 InputShapesMap max_inputs_shape, bool enable_const_folder) {
    config_    = net_config;
    run_mode_  = config_.train_config.run_mode;
    Status ret = TNN_OK;

    ret = DefaultNetwork::Init(net_config, model_config, interpreter, min_inputs_shape, max_inputs_shape,
                               enable_const_folder);
    RETURN_ON_NEQ(ret, TNN_OK);

    RETURN_ON_NEQ(UpdateGradMap(), TNN_OK);

    return TNN_OK;
}

Status DefaultTrainNetwork::TrainStep() {
    if (run_mode_ != TRAIN_MODE_TRAIN) {
        return TNN_OK;
    }

    Status ret = TNN_OK;
    for (auto layer : layers_) {
        ret = layer->RefreshBuffers();
        if (ret != TNN_OK) {
            LOGE("%s layer RefreshBuffers error %s, exit\n", layer->GetLayerName().c_str(), ret.description().c_str());
            return ret;
        }
    }

    return ret;
}

Status DefaultTrainNetwork::GetTrainingFeedback(TrainingFeedback &feed_back) {
    feed_back.loss_name        = GetLossBlobName();
    feed_back.global_step_name = GetGlobalStepBlobName();
    return TNN_OK;
}

Status DefaultTrainNetwork::UpdateGradMap() {
    forward_blob_to_grad_map_.clear();
    grad_to_resource_map_.clear();

    for (auto layer : layers_) {
        auto sgd_layer = dynamic_cast<SGDLayer *>(layer);
        if (sgd_layer) {
            std::vector<RawBuffer *> trainable_resources;
            for (auto input : sgd_layer->GetInputBlobs()) {
                if (grad_to_resource_map_.find(input) == grad_to_resource_map_.end()) {
                    LOGD("DefaultTrainNetwork::UpdateGradMap, sgd layer find update resource error\n");
                    return Status(TNNERR_NET_ERR, "sgd layer find update resource error");
                }
                trainable_resources.push_back(grad_to_resource_map_.at(input));
            }
            sgd_layer->SetTrainableResources(trainable_resources);
            continue;
        }

        auto grad_layer = dynamic_cast<GradientLayer *>(layer);
        if (!grad_layer) {
            continue;
        }
        int index = 0;
        for (auto pair : grad_layer->GetBlobGradPairs()) {
            // if blob appears more than once, set accumulate flag
            if (forward_blob_to_grad_map_.find(pair.first) != forward_blob_to_grad_map_.end()) {
                RETURN_ON_NEQ(grad_layer->SetAccumulateBlobGradFlag(index, true), TNN_OK);
                LOGD("layer %s accumulate %d's blob grad\n", layer->GetLayerName().c_str(), index);
            }
            forward_blob_to_grad_map_.insert(pair);
            ++index;
        }
        index = 0;
        for (auto pair : grad_layer->GetGradResourcePairs()) {
            // if resource appears more than once, set accumulate flag
            if (grad_to_resource_map_.find(pair.first) != grad_to_resource_map_.end()) {
                RETURN_ON_NEQ(grad_layer->SetAccumulateResourceGradFlag(index, true), TNN_OK);
                LOGD("layer %s accumulate %d's resource grad\n", layer->GetLayerName().c_str(), index);
            }
            grad_to_resource_map_.insert(pair);
            ++index;
        }
        for (index = 0; index < grad_layer->GetUpstreamGradCount(); ++index) {
            auto offset         = grad_layer->GetInputBlobs().size() - grad_layer->GetUpstreamGradCount();
            auto forward_output = grad_layer->GetInputBlobs().at(index + offset);
            if (forward_blob_to_grad_map_.find(forward_output) != forward_blob_to_grad_map_.end()) {
                RETURN_ON_NEQ(grad_layer->SetUpstreamGrad(index, forward_blob_to_grad_map_.at(forward_output)), TNN_OK);
            } else {
                LOGD("Dont get %d's upstream grad of layer %s, assume all 1s\n", index, layer->GetLayerName().c_str());
            }
        }
    }

    LOGD("Blob to grad map:\n");
    for (auto iter : forward_blob_to_grad_map_) {
        LOGD("%s -> %s\n", iter.first->GetBlobDesc().description().c_str(),
             iter.second->GetBlobDesc().description().c_str());
    }

    LOGD("Grad to resource map:\n");
    for (auto iter : grad_to_resource_map_) {
        LOGD("%s -> %d\n", iter.first->GetBlobDesc().description().c_str(), iter.second->GetDataCount());
    }

    return TNN_OK;
}

std::string DefaultTrainNetwork::GetLossBlobName() {
    LayerInfo *loss_layer = nullptr;
    for (auto layer : net_structure_->layers) {
        if (layer->type == LAYER_GRADIENT) {
            break;
        } else {
            loss_layer = layer.get();
        }
    }
    if (!loss_layer) {
        LOGE("DefaultTrainNetwork::GetLossBlobName ERROR, cannot get loss name\n");
        return "";
    }
    return loss_layer->outputs[0];
}

std::string DefaultTrainNetwork::GetGlobalStepBlobName() {
    LayerInfo *solver_layer = net_structure_->layers.back().get();
    if (!solver_layer) {
        LOGE("DefaultTrainNetwork::GetGlobalStepBlobName ERROR, cannot get global_step name\n");
        return "";
    }
    return solver_layer->outputs[0];
}

}  // namespace TNN_NS
