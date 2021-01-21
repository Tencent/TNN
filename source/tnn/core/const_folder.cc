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
    
    std::set<std::string> constant_layers;
    //将计算好的常量放入NetResource中，保留模型原有的常量
    ConstantResource constant_map = net_resource_->constant_map;
    BlobShapesMap shapes_map;
    
    for (auto layer : layers_) {
        if (layer->IsOutputConstant()) {
            constant_layers.insert(layer->GetLayerName());
            continue;
        }
        
        //save const input blob
        auto inputs = layer->GetInputBlobs();
        for (auto blob : inputs) {
            if (!blob->IsConstant()) {
                continue;
            }
            
            //save constant resource
            std::shared_ptr<RawBuffer> buffer = nullptr;
            status= Blob2RawBuffer(blob, buffer);
            RETURN_ON_NEQ(status, TNN_OK);
            
            LOGD("ConstFolder save const with name: %s\n", blob->GetBlobDesc().name.c_str());
            
            constant_map[blob->GetBlobDesc().name] = buffer;
        }
        
        //save all input and output blob shapes
        {
            auto inputs = layer->GetInputBlobs();
            for (auto blob : inputs) {
                shapes_map[blob->GetBlobDesc().name]  = blob->GetBlobDesc().dims;
            }
            auto outputs = layer->GetOutputBlobs();
            for (auto blob : outputs) {
                shapes_map[blob->GetBlobDesc().name]  = blob->GetBlobDesc().dims;
            }
        }

    }
    net_resource_->constant_layers = constant_layers;
    net_resource_->constant_map = constant_map;
    net_resource_->blob_shapes_map = shapes_map;
    
    return TNN_OK;
}

}  // namespace TNN_NS
