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

DECLARE_COREML_LAYER_WITH_FUNC_DATA(Relu6, LAYER_RELU6,
                                    virtual std::vector<CoreML__Specification__NeuralNetworkLayer*> GetCoreMLLayerPtrs();,
                                    std::shared_ptr<CoreMLBaseLayer> coreml_layer_clip_;
                                    std::shared_ptr<LayerInfo> clip_layer_info_;);

std::vector<CoreML__Specification__NeuralNetworkLayer*> CoreMLRelu6Layer::GetCoreMLLayerPtrs() {
    auto all_ptrs = CoreMLBaseLayer::GetCoreMLLayerPtrs();
    if (coreml_layer_clip_) {
        auto ptrs = coreml_layer_clip_->GetCoreMLLayerPtrs();
        all_ptrs.insert(all_ptrs.end(), ptrs.begin(), ptrs.end());
    }
    return all_ptrs;
}

Status CoreMLRelu6Layer::BuildLayerType() {
    //nullfy coreml_layer_, ortherwise GetCoreMLLayerPtrs will get wrong result
    coreml_layer_ = nullptr;
    return TNN_OK;
}

Status CoreMLRelu6Layer::BuildLayerParam() {
    //layer param
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
        LOGE("CoreMLRelu6Layer has no fixed input or output shape\n");
        return Status(TNNERR_MODEL_ERR, "CoreMLRelu6Layer has no fixed input or output shape");
    }
    
    // use Clip(min=0, max=6) to instead of Relu6
    coreml_layer_clip_ = CreateCoreMLBaseLayer(LAYER_CLIP);
    coreml_layer_clip_->SetNetResource(net_resource_);

    //build clip
    {
        clip_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
        auto clip_param = std::shared_ptr<ClipLayerParam>(new ClipLayerParam);
        {
            clip_layer_info_->type = LAYER_CLIP;
            clip_layer_info_->name = layer_info_->name;
            clip_layer_info_->inputs = layer_info_->inputs;
            clip_layer_info_->outputs = layer_info_->outputs;
            clip_layer_info_->param = clip_param;
            {
                clip_param->min = 0;
                clip_param->max = 6;
            }
        }
        //put clip output shape to net resource
        net_resource_->blob_shapes_map[clip_layer_info_->outputs[0]] = output_shape;
        
        RETURN_ON_NEQ(coreml_layer_clip_->Init(clip_layer_info_.get(), nullptr),  TNN_OK);
    }
    
    return TNN_OK;
}

Status CoreMLRelu6Layer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLRelu6Layer::BuildLayerInputs() {
    return std::vector<std::string>();
}

std::vector<std::string> CoreMLRelu6Layer::BuildLayerOutputs() {
    return std::vector<std::string>();
}

REGISTER_COREML_LAYER(Relu6, LAYER_RELU6);

}  // namespace TNN_NS
