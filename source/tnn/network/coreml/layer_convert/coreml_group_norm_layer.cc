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

DECLARE_COREML_LAYER_WITH_FUNC_DATA(Groupnorm, LAYER_GROUP_NORM,
                                     virtual std::vector<CoreML__Specification__NeuralNetworkLayer*> GetCoreMLLayerPtrs();,
                                     std::shared_ptr<CoreMLBaseLayer> coreml_layer_reshape0_;
                                     std::shared_ptr<LayerInfo> reshape0_layer_info_;
                                     std::shared_ptr<CoreMLBaseLayer> coreml_layer_instancenorm_;
                                     std::shared_ptr<LayerInfo> instancenorm_layer_info_;
                                     std::shared_ptr<InstanceNormLayerResource> instancenorm_resource_;
                                     std::shared_ptr<EltwiseLayerResource> mul_resource_;
                                     std::shared_ptr<EltwiseLayerResource> add_resource_;
                                     std::shared_ptr<float> instancenorm_scale_;
                                     std::shared_ptr<float> instancenorm_bias_;
                                     std::shared_ptr<CoreMLBaseLayer> coreml_layer_reshape1_;
                                     std::shared_ptr<LayerInfo> reshape1_layer_info_;
                                     std::shared_ptr<CoreMLBaseLayer> coreml_layer_mul_;
                                     std::shared_ptr<LayerInfo> mul_layer_info_;
                                     std::shared_ptr<CoreMLBaseLayer> coreml_layer_add_;
                                     std::shared_ptr<LayerInfo> add_layer_info_;);

std::vector<CoreML__Specification__NeuralNetworkLayer*> CoreMLGroupnormLayer::GetCoreMLLayerPtrs() {
    auto all_ptrs = CoreMLBaseLayer::GetCoreMLLayerPtrs();
    if (coreml_layer_reshape0_) {
        auto ptrs = coreml_layer_reshape0_->GetCoreMLLayerPtrs();
        all_ptrs.insert(all_ptrs.end(), ptrs.begin(), ptrs.end());
    }
    if (coreml_layer_instancenorm_) {
        auto ptrs = coreml_layer_instancenorm_->GetCoreMLLayerPtrs();
        all_ptrs.insert(all_ptrs.end(), ptrs.begin(), ptrs.end());
    }
    if (coreml_layer_reshape1_) {
        auto ptrs = coreml_layer_reshape1_->GetCoreMLLayerPtrs();
        all_ptrs.insert(all_ptrs.end(), ptrs.begin(), ptrs.end());
    }
    if (coreml_layer_mul_) {
        auto ptrs = coreml_layer_mul_->GetCoreMLLayerPtrs();
        all_ptrs.insert(all_ptrs.end(), ptrs.begin(), ptrs.end());
    }
    if (coreml_layer_add_) {
        auto ptrs = coreml_layer_add_->GetCoreMLLayerPtrs();
        all_ptrs.insert(all_ptrs.end(), ptrs.begin(), ptrs.end());
    }
    return all_ptrs;
}

Status CoreMLGroupnormLayer::BuildLayerType() {
    //nullfy coreml_layer_, ortherwise GetCoreMLLayerPtrs will get wrong result
    coreml_layer_ = nullptr;
    return TNN_OK;
}

Status CoreMLGroupnormLayer::BuildLayerParam() {
    auto param = dynamic_cast<GroupNormLayerParam *>(layer_info_->param.get());
    CHECK_PARAM_NULL(param);
    
    auto blob_name_scale = layer_info_->inputs[1];
    auto blob_name_bias = layer_info_->inputs[2];

    std::shared_ptr<RawBuffer> buffer_scale = nullptr;
    if (net_resource_->constant_map.find(blob_name_scale) != net_resource_->constant_map.end()) {
        buffer_scale = net_resource_->constant_map[blob_name_scale];
    }
    std::shared_ptr<RawBuffer> buffer_bias = nullptr;
    if (net_resource_->constant_map.find(blob_name_bias) != net_resource_->constant_map.end()) {
        buffer_bias = net_resource_->constant_map[blob_name_bias];
    }
    
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
    reshape0_output_shape[0] = 1;
    reshape0_output_shape[1] = DimsFunctionUtils::GetDim(input_shape, 0);
    reshape0_output_shape[2] = param->group;
    reshape0_output_shape[3] = DimsVectorUtils::Count(input_shape, 1) / param->group;

    DimsVector reshape1_output_shape = input_shape;
    
    coreml_layer_reshape0_ = CreateCoreMLBaseLayer(LAYER_RESHAPE);
    coreml_layer_instancenorm_ = CreateCoreMLBaseLayer(LAYER_INST_BATCH_NORM);
    coreml_layer_reshape1_ = CreateCoreMLBaseLayer(LAYER_RESHAPE);
    coreml_layer_mul_ = CreateCoreMLBaseLayer(LAYER_MUL);
    coreml_layer_add_ = CreateCoreMLBaseLayer(LAYER_ADD);

    if (coreml_layer_reshape0_ == nullptr || coreml_layer_instancenorm_ == nullptr || coreml_layer_reshape1_ == nullptr || coreml_layer_mul_ == nullptr || coreml_layer_add_ == nullptr) {
        LOGE("Error: CreateCoreMLBaseLayer failed, dont support type:GrouoNorm\n");
        return Status(TNNERR_PARAM_ERR, "CreateCoreMLBaseLayer failed, dont support op");
    }
    coreml_layer_reshape0_->SetNetResource(net_resource_);
    coreml_layer_instancenorm_->SetNetResource(net_resource_);
    coreml_layer_reshape1_->SetNetResource(net_resource_);
    coreml_layer_mul_->SetNetResource(net_resource_);
    coreml_layer_add_->SetNetResource(net_resource_);

    
    //build reshape0
    {
        reshape0_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
        auto reshape0_param = std::shared_ptr<ReshapeLayerParam>(new ReshapeLayerParam);
        {
            reshape0_layer_info_->type = LAYER_RESHAPE;
            reshape0_layer_info_->name = layer_info_->name + "-groupnrom-reshape0";
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
    
    //build instancenorm
    {
        instancenorm_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
        auto instancenorm_param = std::shared_ptr<InstanceNormLayerParam>(new InstanceNormLayerParam);
        {
            instancenorm_layer_info_->type = LAYER_INST_BATCH_NORM;
            instancenorm_layer_info_->name = layer_info_->name + "-groupnrom-instancenorm";
            instancenorm_layer_info_->inputs = reshape0_layer_info_->outputs;
            instancenorm_layer_info_->outputs = {instancenorm_layer_info_->name + "-output"};
            instancenorm_layer_info_->param = instancenorm_param;
            {
                instancenorm_param->type = instancenorm_layer_info_->type;
                instancenorm_param->name = instancenorm_layer_info_->name;
                instancenorm_param->channels = param->group;
                instancenorm_param->eps = param->eps;
            }
        }
        
//        instancenorm_resource_ = std::shared_ptr<InstanceNormLayerResource>(new InstanceNormLayerResource);
//        instancenorm_scale_ = std::shared_ptr<float>(new float);
//        instancenorm_scale_.get()[0] = 1.0;
//        RawBuffer scale_handle(sizeof(float),(char*)instancenorm_scale_.get(),{1});
//        instancenorm_resource_->scale_handle = scale_handle;
//
//        instancenorm_bias_ = std::shared_ptr<float>(new float(0.0));
//        RawBuffer bias_handle(sizeof(float),(char*)instancenorm_bias_.get(),{1});
//        instancenorm_resource_->bias_handle = bias_handle;

        //put permute output shape to net resource
        net_resource_->blob_shapes_map[instancenorm_layer_info_->outputs[0]] = reshape0_output_shape;
        RETURN_ON_NEQ(coreml_layer_instancenorm_->Init(instancenorm_layer_info_.get(), nullptr),  TNN_OK);
//        RETURN_ON_NEQ(coreml_layer_instancenorm_->Init(instancenorm_layer_info_.get(), instancenorm_resource_.get()),  TNN_OK);
    }
    
    //build reshape1
    {
        reshape1_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
        auto reshape1_param = std::shared_ptr<ReshapeLayerParam>(new ReshapeLayerParam);
        {
            reshape1_layer_info_->type = LAYER_RESHAPE;
            reshape1_layer_info_->name = layer_info_->name + "-groupnrom-reshape1";
            reshape1_layer_info_->inputs = instancenorm_layer_info_->outputs;
            reshape1_layer_info_->outputs = {reshape1_layer_info_->name + "output"};
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
    
    //build Mul
    {
        mul_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
        auto mul_param = std::shared_ptr<MultidirBroadcastLayerParam>(new MultidirBroadcastLayerParam);
        {
            mul_layer_info_->type = LAYER_MUL;
            mul_layer_info_->name = layer_info_->name + "-groupnrom-mul";
            mul_layer_info_->inputs = reshape1_layer_info_->outputs;
            mul_layer_info_->outputs = {mul_layer_info_->name + "output"};
            mul_layer_info_->param = mul_param;
            {
                
            }
        }
        //put Mul output shape to net resource
        net_resource_->blob_shapes_map[mul_layer_info_->outputs[0]] = reshape1_output_shape;
        
        mul_resource_ = std::shared_ptr<EltwiseLayerResource>(new EltwiseLayerResource);
        mul_resource_->element_handle = *buffer_scale;
        mul_resource_->element_shape = buffer_scale->GetBufferDims();
        
        RETURN_ON_NEQ(coreml_layer_mul_->Init(mul_layer_info_.get(), mul_resource_.get()),  TNN_OK);
    }
    
    
    //build Mul
    {
        add_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
        auto add_param = std::shared_ptr<MultidirBroadcastLayerParam>(new MultidirBroadcastLayerParam);
        {
            add_layer_info_->type = LAYER_ADD;
            add_layer_info_->name = layer_info_->name + "-groupnrom-add";
            add_layer_info_->inputs = mul_layer_info_->outputs;
            add_layer_info_->outputs = layer_info_->outputs;
            add_layer_info_->param = add_param;
            {
                
            }
        }
        //put Mul output shape to net resource
        net_resource_->blob_shapes_map[add_layer_info_->outputs[0]] = reshape1_output_shape;
        
        add_resource_ = std::shared_ptr<EltwiseLayerResource>(new EltwiseLayerResource);
        add_resource_->element_handle = *buffer_bias;
        add_resource_->element_shape = buffer_bias->GetBufferDims();
        
        RETURN_ON_NEQ(coreml_layer_add_->Init(add_layer_info_.get(), add_resource_.get()),  TNN_OK);
    }
    
    return TNN_OK;
}

Status CoreMLGroupnormLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLGroupnormLayer::BuildLayerInputs() {
    return std::vector<std::string>();
}

std::vector<std::string> CoreMLGroupnormLayer::BuildLayerOutputs() {
    return std::vector<std::string>();
}

REGISTER_COREML_LAYER(Groupnorm, LAYER_GROUP_NORM);

}  // namespace TNN_NS
