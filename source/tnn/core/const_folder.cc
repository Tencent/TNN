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

#include "tnn/core/const_folder.h"

#include <string.h>
#include <sstream>

#include "tnn/core/blob_int8.h"
#include "tnn/core/profile.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource_generator.h"
#include "tnn/memory_manager/blob_memory_pool_factory.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/utils/blob_dump_utils.h"
#include "tnn/utils/blob_transfer_utils.h"
#include "tnn/utils/data_flag_utils.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

ConstFolder::ConstFolder() {
    runtime_model_ = RUNTIME_MODE_CONST_FOLD;
}

ConstFolder::~ConstFolder() {
}

/*
 * The Network holds blob, blobmanager, layers etc.
 * Those object is initialized in this function.
 */
Status ConstFolder::Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                            InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape) {
    config_ = net_config;
    config_.device_type = DEVICE_NAIVE;
    runtime_blob_pool_ = BlobMemoryPoolFactory::CreateBlobMemoryPool(GetDevice(DEVICE_NAIVE));
    
    runtime_model_ = RUNTIME_MODE_CONST_FOLD;
    
    return DefaultNetwork::Init(config_, model_config, interpreter, min_inputs_shape, max_inputs_shape);
}

Status ConstFolder::AllocateBlobMemory() {
    return blob_manager_->AllocateBlobMemory(DATA_FLAG_CHANGE_IF_SHAPE_DIFFER);
}

Status ConstFolder::Reshape(const InputShapesMap &inputs) {
    return DefaultNetwork::Reshape(inputs);
}

Status ConstFolder::DeInit() {
    return DefaultNetwork::DeInit();
}

Status ConstFolder::Forward() {
    auto status = DefaultNetwork::Forward();
    RETURN_ON_NEQ(status, TNN_OK);
    
    BlobShapesMap shapes_map;
    //save all input and output blob shapes, better for cuda device
    for (auto layer : layers_){
        auto inputs = layer->GetInputBlobs();
        for (auto blob : inputs) {
            shapes_map[blob->GetBlobDesc().name]  = blob->GetBlobDesc().dims;
        }
        auto outputs = layer->GetOutputBlobs();
        for (auto blob : outputs) {
            shapes_map[blob->GetBlobDesc().name]  = blob->GetBlobDesc().dims;
        }
    }
    
    std::set<std::string> constant_layers;
    std::set<std::string> shape_differ_layers;
    //In Forword, keep old const resource for reuse, save new const blobs to ConstantResource
    //In GetOptimizeNetStructure, remove redundant constants of layer NEVER CHANGE
    ConstantResource constant_map = net_resource_->constant_map;

    for (auto layer : layers_) {
        auto layer_flag = layer->GetLayerChangeFlag();
        if (layer_flag == DATA_FLAG_CHANGE_NEVER) {
            constant_layers.insert(layer->GetLayerName());
            if (layer->GetLayerChangeFlag() == DATA_FLAG_CHANGE_IF_SHAPE_DIFFER) {
                shape_differ_layers.insert(layer->GetLayerName());
                continue;
            }
            continue;
        } else if (layer_flag == DATA_FLAG_CHANGE_IF_SHAPE_DIFFER) {
            constant_layers.insert(layer->GetLayerName());
            // never change input of layers SHAPE_DIFFER must be saved
            shape_differ_layers.insert(layer->GetLayerName());
        }

        //save const input blob
        auto inputs = layer->GetInputBlobs();
        for (auto blob : inputs) {
            auto blob_flag = DataFlagUtils::ChangeStatus(blob->flag);
            if ((layer_flag == DATA_FLAG_CHANGE_ALWAYS && blob_flag > 0) ||
                (layer_flag == DATA_FLAG_CHANGE_IF_SHAPE_DIFFER && blob_flag == DATA_FLAG_CHANGE_NEVER)) {
                //save constant resource
                std::shared_ptr<RawBuffer> buffer = nullptr;
                status= Blob2RawBuffer(blob, buffer);
                RETURN_ON_NEQ(status, TNN_OK);
                
                {
                    std::stringstream ss;
                    ss << "<" << blob->GetBlobDesc().name << "> shape:[";
                    for(int i: blob->GetBlobDesc().dims) {ss <<  i << ","; } ss << "]";
                    LOGD("ConstFolder save const with name: %s\n", ss.str().c_str());
                }
                
                constant_map[blob->GetBlobDesc().name] = buffer;
            }
        }
    }
    net_resource_->constant_layers = constant_layers;
    net_resource_->shape_differ_layers = shape_differ_layers;
    net_resource_->constant_map = constant_map;
    net_resource_->blob_shapes_map = shapes_map;
    
    return TNN_OK;
}

Status ConstFolder::GetOptimizedNet(std::shared_ptr<NetStructure> &const_fold_struct,
                               std::shared_ptr<NetResource> &const_fold_resource,
                               int  target_flag) {
    target_flag = DataFlagUtils::ChangeStatus(target_flag);
    
    auto net_structure = net_structure_;
    auto net_resource = net_resource_;
    
    auto constant_layers = net_resource_->constant_layers;
    auto shape_differ_layers = net_resource_->shape_differ_layers;
    
    //optimized layers, remove redundant layer
    std::vector<std::shared_ptr<LayerInfo>> optmized_layers;
    
    const_fold_struct = std::make_shared<NetStructure>();
    *const_fold_struct = *net_structure;
    {
        
        for (auto iter : net_structure->layers) {
            if (target_flag == DATA_FLAG_CHANGE_IF_SHAPE_DIFFER) {
                //layers with output flag DATA_FLAG_CHANGE_NEVER or DATA_FLAG_CHANGE_IF_SHAPE_DIFFER will be removed
                if (constant_layers.find(iter->name) == constant_layers.end()) {
                    optmized_layers.push_back(iter);
                }
            } else {
                //only layers with output flag DATA_FLAG_CHANGE_NEVER will be removed
                if (!(constant_layers.find(iter->name) != constant_layers.end() && shape_differ_layers.find(iter->name) == shape_differ_layers.end())) {
                    optmized_layers.push_back(iter);
                }
            }
        }
        const_fold_struct->layers = optmized_layers;
    }
    
    const_fold_resource = std::make_shared<NetResource>();
    *const_fold_resource = *net_resource;
    {
        //In GetOptimizeNetStructure,  remove redundant constants of layer NEVER CHANGE
        std::map<std::string, std::shared_ptr<LayerResource>> optmized_resource_map;
        ConstantResource optmized_constant_map;
        
        auto resource_map = net_resource->resource_map;
        auto constant_map = net_resource->constant_map;
        
        for (auto layer_info : optmized_layers) {
            BaseLayer * layer = nullptr;
            for (auto item : layers_) {
                if (item->GetLayerName() == layer_info->name) {
                    layer = item;
                    break;
                }
            }
            RETURN_VALUE_ON_NEQ(!layer, false, Status(TNNERR_LAYER_ERR, "layer is nil, internal error"));
            
            if (resource_map.find(layer_info->name) != resource_map.end()) {
                optmized_resource_map[layer_info->name] = resource_map[layer_info->name];
            }
            
            auto layer_flag = layer->GetLayerChangeFlag();
            for (auto blob : layer->GetInputBlobs()) {
                auto blob_name = blob->GetBlobDesc().name;
                if (constant_map.find(blob_name) == constant_map.end()) {
                    continue;
                }
                
                auto blob_flag = DataFlagUtils::ChangeStatus(blob->flag);
   
                if ((target_flag == DATA_FLAG_CHANGE_IF_SHAPE_DIFFER && blob_flag > 0) ||
                    (target_flag == DATA_FLAG_CHANGE_NEVER && blob_flag == DATA_FLAG_CHANGE_NEVER)) {
                    optmized_constant_map[blob_name] = constant_map[blob_name];
                    LOGD("GetOptimizedNet save const with name: %s\n", blob_name.c_str());
                }
            }
        }
        
        const_fold_resource->resource_map = optmized_resource_map;
        const_fold_resource->constant_map = optmized_constant_map;
    }

    
    return TNN_OK;
}

}  // namespace TNN_NS
