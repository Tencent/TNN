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

DECLARE_COREML_LAYER_WITH_FUNC_DATA(Swish, LAYER_SWISH,
                                     virtual std::vector<CoreML__Specification__NeuralNetworkLayer*> GetCoreMLLayerPtrs();,
                                     std::shared_ptr<CoreMLBaseLayer> coreml_layer_sigmoid_;
                                     std::shared_ptr<LayerInfo> sigmoid_layer_info_;
                                     std::shared_ptr<CoreMLBaseLayer> coreml_layer_mul_;
                                     std::shared_ptr<LayerInfo> mul_layer_info_;);

std::vector<CoreML__Specification__NeuralNetworkLayer*> CoreMLSwishLayer::GetCoreMLLayerPtrs() {
    auto all_ptrs = CoreMLBaseLayer::GetCoreMLLayerPtrs();
    if (coreml_layer_sigmoid_) {
        auto ptrs = coreml_layer_sigmoid_->GetCoreMLLayerPtrs();
        all_ptrs.insert(all_ptrs.end(), ptrs.begin(), ptrs.end());
    }
    if (coreml_layer_mul_) {
        auto ptrs = coreml_layer_mul_->GetCoreMLLayerPtrs();
        all_ptrs.insert(all_ptrs.end(), ptrs.begin(), ptrs.end());
    }
    return all_ptrs;
}

Status CoreMLSwishLayer::BuildLayerType() {
    //nullfy coreml_layer_, ortherwise GetCoreMLLayerPtrs will get wrong result
    coreml_layer_ = nullptr;
    return TNN_OK;
}

Status CoreMLSwishLayer::BuildLayerParam() {
//    auto param = dynamic_cast<SwishLayerParam *>(layer_info_->param.get());
//    CHECK_PARAM_NULL(param);
    
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
        LOGE("CoreMLSwishLayer has no fixed input or output shape\n");
        return Status(TNNERR_MODEL_ERR, "CoreMLSwishLayer has no fixed input or output shape");
    }
    
    // use Sigmoid & Mul to instead of Swish
    coreml_layer_sigmoid_ = CreateCoreMLBaseLayer(LAYER_SIGMOID);
    coreml_layer_sigmoid_->SetNetResource(net_resource_);

    coreml_layer_mul_ = CreateCoreMLBaseLayer(LAYER_MUL);
    coreml_layer_mul_->SetNetResource(net_resource_);
    
    //build Sigmoid
    {
        sigmoid_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
//        auto sigmoid_param = std::shared_ptr<SigmoidLayerParam>(new SigmoidLayerParam);
        {
            sigmoid_layer_info_->type = LAYER_SIGMOID;
            sigmoid_layer_info_->name = layer_info_->name + "-sigmoid";
            if (layer_info_->inputs.size() == 1) {
                sigmoid_layer_info_->inputs = layer_info_->inputs;
            } else if (layer_info_->inputs.size() == 2) {
                sigmoid_layer_info_->inputs = {layer_info_->inputs[1]};
            } else {
                LOGE("CoreMLSwishLayer can not support input shape size: %d\n", (int) layer_info_->inputs.size());
                return Status(TNNERR_MODEL_ERR, "CoreMLSwishLayer can not support this input shape size");
            }
            sigmoid_layer_info_->outputs = {sigmoid_layer_info_->name + "-out"};
//            sigmoid_layer_info_->param = sigmoid_param;
//            {
//
//            }
        }
        //put Sigmoid output shape to net resource
        net_resource_->blob_shapes_map[sigmoid_layer_info_->outputs[0]] = output_shape;
        
        RETURN_ON_NEQ(coreml_layer_sigmoid_->Init(sigmoid_layer_info_.get(), nullptr),  TNN_OK);
    }
    
    //build Mul
    {
        mul_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
        auto mul_param = std::shared_ptr<MultidirBroadcastLayerParam>(new MultidirBroadcastLayerParam);
        {
            mul_layer_info_->type = LAYER_MUL;
            mul_layer_info_->name = layer_info_->name + "-mul";
            std::vector<std::string> inputs_;
            if (layer_info_->inputs.size() == 1) {
                inputs_ = layer_info_->inputs;
            } else if (layer_info_->inputs.size() == 2) {
                inputs_ = {layer_info_->inputs[0]};
            } else {
                LOGE("CoreMLSwishLayer can not support input shape size: %d\n", (int) layer_info_->inputs.size());
                return Status(TNNERR_MODEL_ERR, "CoreMLSwishLayer can not support this input shape size");
            }
            inputs_.push_back(layer_info_->name + "-sigmoid-out");
            mul_layer_info_->inputs = inputs_;
            mul_layer_info_->outputs = layer_info_->outputs;
            mul_layer_info_->param = mul_param;
            {
                
            }
        }
        //put Mul output shape to net resource
        net_resource_->blob_shapes_map[mul_layer_info_->outputs[0]] = output_shape;
        
        RETURN_ON_NEQ(coreml_layer_mul_->Init(mul_layer_info_.get(), nullptr),  TNN_OK);
    }
    
    
    return TNN_OK;
}

Status CoreMLSwishLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLSwishLayer::BuildLayerInputs() {
    return std::vector<std::string>();
}

std::vector<std::string> CoreMLSwishLayer::BuildLayerOutputs() {
    return std::vector<std::string>();
}

REGISTER_COREML_LAYER(Swish, LAYER_SWISH);

}  // namespace TNN_NS
