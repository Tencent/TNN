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

    RETURN_ON_NEQ(UpdateForwardLayerCount(), TNN_OK);

    RETURN_ON_NEQ(UpdateSolver(), TNN_OK);

    return TNN_OK;
}

Status DefaultTrainNetwork::TrainStep() {
    if (run_mode_ != TRAIN_MODE_TRAIN) {
        return TNN_OK;
    }

    Status ret = TNN_OK;

    RuntimeMode prev_mode = runtime_model_;
    runtime_model_        = RUNTIME_MODE_BACKWARD;
    ret                   = Forward();
    if (ret != TNN_OK) {
        LOGE("DefaultTrainNetwork::TrainStep, backward pass failed\n");
        return ret;
    }
    runtime_model_ = prev_mode;

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
    feed_back.loss_name        = loss_name_;
    feed_back.global_step_name = global_step_name_;
    return TNN_OK;
}

Status DefaultTrainNetwork::UpdateGradMap() {
    forward_blob_to_grad_map_.clear();
    grad_to_resource_map_.clear();
    std::set<RawBuffer *> resource_visited;

    for (auto layer : layers_) {
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
            if (resource_visited.find(pair.second) != resource_visited.end()) {
                RETURN_ON_NEQ(grad_layer->SetAccumulateResourceGradFlag(index, true), TNN_OK);
                LOGD("layer %s accumulate %d's resource grad\n", layer->GetLayerName().c_str(), index);
            }
            grad_to_resource_map_.insert(pair);
            resource_visited.insert(pair.second);
            ++index;
        }
    }

    /*
        LOGD("Blob to grad map:\n");
        for (auto iter : forward_blob_to_grad_map_) {
            LOGD("%s -> %s\n", iter.first->GetBlobDesc().description().c_str(),
                 iter.second->GetBlobDesc().description().c_str());
        }

        LOGD("Grad to resource map:\n");
        for (auto iter : grad_to_resource_map_) {
            LOGD("%s -> %d\n", iter.first->GetBlobDesc().description().c_str(), iter.second->GetDataCount());
        }
    */

    return TNN_OK;
}

Status DefaultTrainNetwork::UpdateForwardLayerCount() {
    LayerInfo *loss_layer = nullptr;
    int cnt               = 0;
    for (auto layer : net_structure_->layers) {
        if (layer->type == LAYER_GRADIENT) {
            break;
        }
        loss_layer = layer.get();
        cnt++;
    }
    if (!loss_layer) {
        LOGE("DefaultTrainNetwork::UpdateForwardLayerCount ERROR, cannot get loss layer\n");
        return Status(TNNERR_TRAIN_ERROR, "cannot get loss layer");
    }
    loss_name_           = loss_layer->outputs[0];
    forward_layer_count_ = cnt;

    return TNN_OK;
}

Status DefaultTrainNetwork::UpdateSolver() {
    LayerInfo *solver_layer = net_structure_->layers.back().get();
    if (!solver_layer) {
        LOGE("DefaultTrainNetwork::UpdateSolver ERROR, layers is empty\n");
        return Status(TNNERR_TRAIN_ERROR, "layers is empty");
    }
    global_step_name_ = solver_layer->outputs[0];

    // only support sgd now
    auto sgd_layer = dynamic_cast<SGDLayer *>(layers_.back());
    if (!sgd_layer) {
        LOGE("DefaultTrainNetwork::UpdateSolver ERROR, sgd_layer is empty\n");
        return Status(TNNERR_TRAIN_ERROR, "sgd_layer is empty");
    }

    std::vector<RawBuffer *> trainable_resources;
    for (auto input : sgd_layer->GetInputBlobs()) {
        if (grad_to_resource_map_.find(input) == grad_to_resource_map_.end()) {
            LOGD("DefaultTrainNetwork::UpdateSolver, sgd layer find resource error\n");
            return Status(TNNERR_NET_ERR, "sgd layer find resource error");
        }
        trainable_resources.push_back(grad_to_resource_map_.at(input));
    }
    return sgd_layer->SetTrainableResources(trainable_resources);
}

}  // namespace TNN_NS
