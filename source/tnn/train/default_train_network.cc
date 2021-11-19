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
#include "tnn/train/solver/solver_layer.h"

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

    RETURN_ON_NEQ(InitTrainingStatus(), TNN_OK);

    RETURN_ON_NEQ(InitRuntimeInfo(), TNN_OK);

    return TNN_OK;
}

Status DefaultTrainNetwork::GetAllInputBlobs(BlobMap &blobs) {
    blob_manager_->GetAllInputBlobs(blobs);
    // loss grad is assumed to be one
    blobs.erase(loss_grad_name_);
    // global step init value is assumed to be zero
    blobs.erase(global_step_init_name_);
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

    for (auto layer : need_refresh_layers_) {
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

Status DefaultTrainNetwork::InitTrainingStatus() {
    LayerInfo *loss_layer      = nullptr;
    LayerInfo *loss_grad_layer = nullptr;
    int cnt                    = 0;
    for (auto layer : net_structure_->layers) {
        if (layer->type == LAYER_GRADIENT) {
            loss_grad_layer = layer.get();
            break;
        }
        loss_layer = layer.get();
        cnt++;
    }
    forward_layer_count_ = cnt;
    if (!loss_layer) {
        LOGE("DefaultTrainNetwork::InitTrainingStatus ERROR, cannot get loss layer\n");
        return Status(TNNERR_TRAIN_ERROR, "cannot get loss layer");
    }
    if (!loss_grad_layer) {
        LOGE("DefaultTrainNetwork::InitTrainingStatus ERROR, cannot get loss grad layer\n");
        return Status(TNNERR_TRAIN_ERROR, "cannot get loss grad layer");
    }
    loss_name_      = loss_layer->outputs[0];
    loss_grad_name_ = loss_grad_layer->inputs.back();

    LayerInfo *solver_layer_info = net_structure_->layers.back().get();
    if (!solver_layer_info) {
        LOGE("DefaultTrainNetwork::InitTrainingStatus ERROR, solver layer is empty\n");
        return Status(TNNERR_TRAIN_ERROR, "solver layer is empty");
    }
    global_step_name_      = solver_layer_info->outputs[0];
    global_step_init_name_ = solver_layer_info->inputs.back();

    RETURN_ON_NEQ(SetLossGrad(), TNN_OK);
    RETURN_ON_NEQ(SetGlobalStep(), TNN_OK);

    return TNN_OK;
}

Status DefaultTrainNetwork::InitRuntimeInfo() {
    RETURN_ON_NEQ(SetGradientLayerRuntimeInfo(), TNN_OK);
    RETURN_ON_NEQ(SetSolverLayerRuntimeInfo(), TNN_OK);
    return TNN_OK;
}

Status DefaultTrainNetwork::SetLossGrad() {
    Blob *loss_blob = blob_manager_->GetBlob(loss_name_);
    if (!loss_blob) {
        LOGE("DefaultTrainNetwork::SetLossGrad get loss_blob failed\n");
        return Status(TNNERR_TRAIN_ERROR, "get loss_blob failed!");
    }
    auto loss_data_count = DimsVectorUtils::Count(loss_blob->GetBlobDesc().dims);
    if (loss_data_count != 1) {
        LOGE(
            "DefaultTrainNetwork::SetLossGrad only support loss data count = 1 now, got %d. Try to change loss "
            "function type or loss target layer!\n",
            loss_data_count);
        return Status(TNNERR_TRAIN_ERROR,
                      "loss data count not supported, try to change loss function type or loss target layer!");
    }

    std::shared_ptr<Mat> mat(new Mat(DEVICE_ARM, NCHW_FLOAT, {loss_data_count}));
    if (!mat || !mat->GetData()) {
        LOGE("DefaultTrainNetwork::SetLossGrad create mat failed\n");
        return Status(TNNERR_TRAIN_ERROR, "create mat failed");
    }

    // init loss grad as one
    auto ptr = reinterpret_cast<float *>(mat->GetData());
    for (int i = 0; i < loss_data_count; ++i) {
        ptr[i] = 1.0;
    }

    Blob *loss_grad = blob_manager_->GetBlob(loss_grad_name_);
    if (!loss_grad) {
        LOGE("DefaultTrainNetwork::SetLossGrad get loss_grad failed\n");
        return Status(TNNERR_TRAIN_ERROR, "get loss_grad failed!");
    }

    // create blob convert
    std::shared_ptr<BlobConverter> blob_converter = std::make_shared<BlobConverter>(loss_grad);

    // get command queue
    void *command_queue = nullptr;
    RETURN_ON_NEQ(GetCommandQueue(&command_queue), TNN_OK);

    Status status = blob_converter->ConvertFromMatAsync(*(mat.get()), MatConvertParam(), command_queue);
    if (status != TNN_OK) {
        LOGE("DefaultTrainNetwork::SetLossGrad, ConvertFromMatAsync Error: %s\n", status.description().c_str());
        return status;
    }

    return TNN_OK;
}

Status DefaultTrainNetwork::SetGlobalStep() {
    std::shared_ptr<Mat> mat(new Mat(DEVICE_ARM, NCHW_FLOAT, {1}));
    if (!mat || !mat->GetData()) {
        LOGE("DefaultTrainNetwork::SetGlobalStep create mat failed\n");
        return Status(TNNERR_TRAIN_ERROR, "create mat failed");
    }

    // init global step as zero
    auto ptr = reinterpret_cast<float *>(mat->GetData());
    *ptr     = 0.0;

    Blob *global_step_init = blob_manager_->GetBlob(global_step_init_name_);
    if (!global_step_init) {
        LOGE("DefaultTrainNetwork::SetGlobalStep get global_step_init failed\n");
        return Status(TNNERR_TRAIN_ERROR, "get global_step_init failed!");
    }

    // create blob convert
    std::shared_ptr<BlobConverter> blob_converter = std::make_shared<BlobConverter>(global_step_init);

    // get command queue
    void *command_queue = nullptr;
    RETURN_ON_NEQ(GetCommandQueue(&command_queue), TNN_OK);

    Status status = blob_converter->ConvertFromMatAsync(*(mat.get()), MatConvertParam(), command_queue);
    if (status != TNN_OK) {
        LOGE("DefaultTrainNetwork::SetGlobalStep, ConvertFromMatAsync Error: %s\n", status.description().c_str());
        return status;
    }

    return TNN_OK;
}

Status DefaultTrainNetwork::SetGradientLayerRuntimeInfo() {
    input_to_grad_map_.clear();
    grad_to_resource_map_.clear();
    need_refresh_layers_.clear();
    std::set<RawBuffer *> resource_visited;

    auto &trainable_layers = config_.train_config.trainable_layers;
    for (auto layer : layers_) {
        if (config_.train_config.train_the_whole_model ||
            (trainable_layers.find(layer->GetLayerName()) != trainable_layers.end())) {
            need_refresh_layers_.push_back(layer);
        }
        auto grad_layer = dynamic_cast<GradientLayer *>(layer);
        if (!grad_layer) {
            continue;
        }
        int index = 0;
        for (auto pair : grad_layer->GetInputGradPairs()) {
            // if blob appears more than once, set accumulate flag
            if (input_to_grad_map_.find(pair.first) != input_to_grad_map_.end()) {
                RETURN_ON_NEQ(grad_layer->SetAccumulateInputGradFlag(index, true), TNN_OK);
                LOGD("layer %s accumulate %d's input grad\n", layer->GetLayerName().c_str(), index);
            }
            input_to_grad_map_.insert(pair);
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

Status DefaultTrainNetwork::SetSolverLayerRuntimeInfo() {
    auto solver_layer = dynamic_cast<SolverLayer *>(layers_.back());
    if (!solver_layer) {
        LOGE("DefaultTrainNetwork::SetSolverLayerRuntimeInfo ERROR, solver_layer is empty\n");
        return Status(TNNERR_TRAIN_ERROR, "solver_layer is empty");
    }

    std::vector<RawBuffer *> trainable_resources;
    for (auto input : solver_layer->GetInputBlobs()) {
        if (input == solver_layer->GetInputBlobs().back()) {
            // global step init value
            break;
        }
        if (grad_to_resource_map_.find(input) == grad_to_resource_map_.end()) {
            LOGE("DefaultTrainNetwork::SetSolverLayerRuntimeInfo, solver layer find resource error, %s\n",
                 input->GetBlobDesc().description().c_str());
            return Status(TNNERR_TRAIN_ERROR, "solver layer find resource error");
        }
        trainable_resources.push_back(grad_to_resource_map_.at(input));
    }
    return solver_layer->SetTrainableResources(trainable_resources);
}

}  // namespace TNN_NS
