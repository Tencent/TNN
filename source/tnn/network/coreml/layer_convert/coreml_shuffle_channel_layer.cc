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

#include "coreml_base_layer.h"

namespace TNN_NS {

DECLARE_COREML_LAYER_WITH_FUNC_DATA(ShuffleChannel, LAYER_SHUFFLE_CHANNEL,
                                     virtual std::vector<CoreML__Specification__NeuralNetworkLayer*> GetCoreMLLayerPtrs();,
                                     std::shared_ptr<CoreMLBaseLayer> coreml_layer_reshape0_;
                                     std::shared_ptr<LayerInfo> reshape0_layer_info_;
                                     std::shared_ptr<CoreMLBaseLayer> coreml_layer_permute_;
                                     std::shared_ptr<LayerInfo> permute_layer_info_;
                                     std::shared_ptr<CoreMLBaseLayer> coreml_layer_reshape1_;
                                     std::shared_ptr<LayerInfo> reshape1_layer_info_;);

std::vector<CoreML__Specification__NeuralNetworkLayer*> CoreMLShuffleChannelLayer::GetCoreMLLayerPtrs() {
    auto all_ptrs = CoreMLBaseLayer::GetCoreMLLayerPtrs();
    if (coreml_layer_reshape0_) {
        auto ptrs = coreml_layer_reshape0_->GetCoreMLLayerPtrs();
        all_ptrs.insert(all_ptrs.end(), ptrs.begin(), ptrs.end());
    }
    if (coreml_layer_permute_) {
        auto ptrs = coreml_layer_permute_->GetCoreMLLayerPtrs();
        all_ptrs.insert(all_ptrs.end(), ptrs.begin(), ptrs.end());
    }
    if (coreml_layer_reshape1_) {
        auto ptrs = coreml_layer_reshape1_->GetCoreMLLayerPtrs();
        all_ptrs.insert(all_ptrs.end(), ptrs.begin(), ptrs.end());
    }
    return all_ptrs;
}

Status CoreMLShuffleChannelLayer::BuildLayerType() {
    //nullfy coreml_layer_, ortherwise GetCoreMLLayerPtrs will get wrong result
    coreml_layer_ = nullptr;
    return TNN_OK;
}

Status CoreMLShuffleChannelLayer::BuildLayerParam() {
    auto param = dynamic_cast<ShuffleLayerParam *>(layer_info_->param.get());
    CHECK_PARAM_NULL(param);
    
    //get input and output shape
    DimsVector input_shape, output_shape;
    if (net_resource_ && layer_info_->inputs.size()>0) {
        if (net_resource_->blob_shapes_map.find(layer_info_->inputs[0]) != net_resource_->blob_shapes_map.end()) {
            input_shape = net_resource_->blob_shapes_map[layer_info_->inputs[0]];
        }
    }
    
    if (net_resource_ && layer_info_->outputs.size()>0) {
        if (net_resource_->blob_shapes_map.find(layer_info_->outputs[0]) != net_resource_->blob_shapes_map.end()) {
            output_shape = net_resource_->blob_shapes_map[layer_info_->outputs[0]];
        }
    }
    
    if (input_shape.size() <= 0 || output_shape.size() <= 0) {
        LOGE("ShuffleChannelLayer has no fixed input or output shape\n");
        return Status(TNNERR_MODEL_ERR, "ShuffleChannelLayer has no fixed input or output shape");
    }
    
    DimsVector reshape0_output_shape(4);
    reshape0_output_shape[0] = input_shape[0];
    reshape0_output_shape[1] = param->group;
    reshape0_output_shape[2] = input_shape[1] / param->group;
    reshape0_output_shape[3] = DimsVectorUtils::Count(input_shape, 2);
    
    DimsVector permute_order = {0, 2, 1, 3};
    DimsVector permute_output_shape = reshape0_output_shape;
    permute_output_shape[1] = reshape0_output_shape[2];
    permute_output_shape[2] = reshape0_output_shape[1];
    
    DimsVector reshape1_output_shape = input_shape;
    
    coreml_layer_reshape0_ = CreateCoreMLBaseLayer(LAYER_RESHAPE);
    coreml_layer_permute_ = CreateCoreMLBaseLayer(LAYER_PERMUTE);
    coreml_layer_reshape1_ = CreateCoreMLBaseLayer(LAYER_RESHAPE);
    if (coreml_layer_reshape0_ == nullptr || coreml_layer_permute_ == nullptr || coreml_layer_reshape1_ == nullptr) {
        LOGE("Error: CreateCoreMLBaseLayer failed, dont support type:ShuffleChannel\n");
        return Status(TNNERR_PARAM_ERR, "CreateCoreMLBaseLayer failed, dont support op");
    }
    coreml_layer_reshape0_->SetNetResource(net_resource_);
    coreml_layer_permute_->SetNetResource(net_resource_);
    coreml_layer_reshape1_->SetNetResource(net_resource_);
    
    
    //build reshape0
    {
        reshape0_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
        auto reshape0_param = std::shared_ptr<ReshapeLayerParam>(new ReshapeLayerParam);
        {
            reshape0_layer_info_->type = LAYER_RESHAPE;
            reshape0_layer_info_->name = layer_info_->name + "-shuffle-reshape0";
            reshape0_layer_info_->inputs = layer_info_->inputs;
            reshape0_layer_info_->outputs = {reshape0_layer_info_->name + "-output"};
            reshape0_layer_info_->param = reshape0_param;
            {
                reshape0_param->type = reshape0_layer_info_->type;
                reshape0_param->name = reshape0_layer_info_->name;
                reshape0_param->shape = reshape0_output_shape;
                reshape0_param->num_axes = (int)reshape0_output_shape.size();
            }
        }
        //put reshape0 output shape to net resource
        net_resource_->blob_shapes_map[reshape0_layer_info_->outputs[0]] = reshape0_output_shape;
        
        RETURN_ON_NEQ(coreml_layer_reshape0_->Init(reshape0_layer_info_.get(), nullptr),  TNN_OK);
    }
    
    //build permute
    {
        permute_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
        auto permute_param = std::shared_ptr<PermuteLayerParam>(new PermuteLayerParam);
        {
            permute_layer_info_->type = LAYER_PERMUTE;
            permute_layer_info_->name = layer_info_->name + "-shuffle-permute";
            permute_layer_info_->inputs = reshape0_layer_info_->outputs;
            permute_layer_info_->outputs = {permute_layer_info_->name + "-output"};
            permute_layer_info_->param = permute_param;
            {
                permute_param->type = permute_layer_info_->type;
                permute_param->name = permute_layer_info_->name;
                permute_param->orders = permute_order;
            }
        }
        
        //put permute output shape to net resource
        net_resource_->blob_shapes_map[permute_layer_info_->outputs[0]] = permute_output_shape;
        RETURN_ON_NEQ(coreml_layer_permute_->Init(permute_layer_info_.get(), nullptr),  TNN_OK);
    }
    
    //build reshape1
    {
        reshape1_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
        auto reshape1_param = std::shared_ptr<ReshapeLayerParam>(new ReshapeLayerParam);
        {
            reshape1_layer_info_->type = LAYER_RESHAPE;
            reshape1_layer_info_->name = layer_info_->name + "-shuffle-reshape1";
            reshape1_layer_info_->inputs = permute_layer_info_->outputs;
            reshape1_layer_info_->outputs = layer_info_->outputs;
            reshape1_layer_info_->param = reshape1_param;
            {
                reshape1_param->type = reshape1_layer_info_->type;
                reshape1_param->name = reshape1_layer_info_->name;
                reshape1_param->shape = reshape1_output_shape;
                reshape1_param->num_axes = (int)reshape1_output_shape.size();
            }
        }
        RETURN_ON_NEQ(coreml_layer_reshape1_->Init(reshape1_layer_info_.get(), nullptr),  TNN_OK);
    }
    
    return TNN_OK;
}

Status CoreMLShuffleChannelLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLShuffleChannelLayer::BuildLayerInputs() {
    return std::vector<std::string>();
}

std::vector<std::string> CoreMLShuffleChannelLayer::BuildLayerOutputs() {
    return std::vector<std::string>();
}

REGISTER_COREML_LAYER(ShuffleChannel, LAYER_SHUFFLE_CHANNEL);

}  // namespace TNN_NS
