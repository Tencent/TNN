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
#include "coreml_const_layer.h"

namespace TNN_NS {

std::shared_ptr<char> NullTerminatedCString(std::string & name) {
    auto cstring = std::shared_ptr<char>(new char[name.size() + 1], [](char* p) { delete[] p; });
    char *ptr = cstring.get();
    for (int i = 0; i < name.size(); i++) {
        ptr[i] = name[i];
        
    }
    ptr[name.size()] = '\0';
    return cstring;
}

CoreMLBaseLayer::CoreMLBaseLayer(LayerType type) {
    this->type_ = type;
}

CoreMLBaseLayer::~CoreMLBaseLayer(){};

Status CoreMLBaseLayer::Convert() {
    auto status = BuildConstantWeightsLayer();
    RETURN_ON_NEQ(status, TNN_OK);
    
    status = BuildLayerType();
    RETURN_ON_NEQ(status, TNN_OK);
    
    status = BuildLayerParam();
    RETURN_ON_NEQ(status, TNN_OK);
    
    auto inputs = BuildLayerInputs();
    auto outputs = BuildLayerOutputs();
    RETURN_ON_NEQ(status, TNN_OK);
    
    SetLayerInputs(inputs);
    SetLayerOutputs(outputs);
    return status;
};

std::vector<CoreML__Specification__NeuralNetworkLayer*> CoreMLBaseLayer::GetCoreMLLayerPtrs() {
    std::vector<CoreML__Specification__NeuralNetworkLayer*> layer_ptrs;
    for (auto& iter : coreml_layer_constant_weights_) {
        auto const_ptr = iter->GetCoreMLLayerPtrs();
        layer_ptrs.insert(layer_ptrs.end(), const_ptr.begin(), const_ptr.end());
    }
    
    if (coreml_layer_before_) {
        auto before_layer = coreml_layer_before_.get();
        auto before_layer_ptr = before_layer->GetCoreMLLayerPtrs();
        layer_ptrs.insert(layer_ptrs.end(), before_layer_ptr.begin(), before_layer_ptr.end());
    }
    
    if (coreml_layer_) {
        layer_ptrs.push_back(coreml_layer_.get());
    }
    
    if (coreml_layer_after_) {
        auto after_layer = coreml_layer_after_.get();
        auto after_layer_ptr = after_layer->GetCoreMLLayerPtrs();
        layer_ptrs.insert(layer_ptrs.end(), after_layer_ptr.begin(), after_layer_ptr.end());
    }
    return layer_ptrs;
}

Status CoreMLBaseLayer::BuildLayerType() {
    return TNN_OK;
}

Status CoreMLBaseLayer::BuildLayerParam() {
    return TNN_OK;
}

Status CoreMLBaseLayer::BuildConstantWeightsLayer() {
    return TNN_OK;
}

std::vector<std::string> CoreMLBaseLayer::BuildLayerInputs() {
    if (!layer_info_) {
        return std::vector<std::string>();
    } else {
        return layer_info_->inputs;
    }
}

std::vector<std::string> CoreMLBaseLayer::BuildLayerOutputs() {
    if (!layer_info_) {
        return std::vector<std::string>();
    } else {
        return layer_info_->outputs;
    }
}

Status CoreMLBaseLayer::Init(LayerInfo* layer_info ,LayerResource* layer_resource) {
    coreml_layer_.reset(new CoreML__Specification__NeuralNetworkLayer);
    core_ml__specification__neural_network_layer__init(coreml_layer_.get());
    
    layer_resource_ = layer_resource;
    layer_info_ = layer_info;
    
    if (layer_info) {
        SetLayerName(layer_info->name);
    }
    
    return Convert();
}

void CoreMLBaseLayer::SetNetResource(NetResource *net_resource) {
    net_resource_ = net_resource;
}

void CoreMLBaseLayer::SetLayerName(std::string& name) {
    coreml_layer_name_ = NullTerminatedCString(name);
    if (coreml_layer_) {
        coreml_layer_->name = coreml_layer_name_.get();
    }
 }

std::string CoreMLBaseLayer::GetLayerName() {
    return layer_info_ ? layer_info_->name : "";
}

void CoreMLBaseLayer::SetLayerInputs(std::vector<std::string>& inputs) {
    if (!coreml_layer_) {
        return;
    }
    
    coreml_layer_->n_input = inputs.size();
    if (inputs.size() > 0) {
        coreml_layer_inputs_arr_ = std::shared_ptr<char*>(new char* [inputs.size()], [](char** p) { delete[] p; });
        coreml_layer_->input = coreml_layer_inputs_arr_.get();
    } else {
        coreml_layer_inputs_arr_ = nullptr;
        coreml_layer_->input = nullptr;
    }
    
    coreml_layer_inputs_.clear();
    for (int i = 0; i < inputs.size(); i++) {
        auto cinput = NullTerminatedCString(inputs[i]);
        coreml_layer_inputs_.push_back(cinput);
        coreml_layer_->input[i] = cinput.get();
     }
}

void CoreMLBaseLayer::SetLayerOutputs(std::vector<std::string>& outputs) {
    if (!coreml_layer_) {
        return;
    }
    
    coreml_layer_->n_output = outputs.size();
    if (outputs.size() > 0) {
        coreml_layer_outputs_arr_ = std::shared_ptr<char*>(new char* [outputs.size()], [](char** p) { delete[] p; });
        coreml_layer_->output = coreml_layer_outputs_arr_.get();
    } else {
        coreml_layer_outputs_arr_ = nullptr;
        coreml_layer_->output = nullptr;
    }
    
    coreml_layer_outputs_.clear();
    for (int i = 0; i < outputs.size(); i++) {
        auto coutput = NullTerminatedCString(outputs[i]);
        coreml_layer_outputs_.push_back(coutput);
        coreml_layer_->output[i] = coutput.get();
     }
}

std::map<LayerType, std::shared_ptr<CoreMLLayerCreator>> &GetGlobalCoreMLLayerCreatorMap() {
    static std::once_flag once;
    static std::shared_ptr<std::map<LayerType, std::shared_ptr<CoreMLLayerCreator>>> creators;
    std::call_once(once, []() { creators.reset(new std::map<LayerType, std::shared_ptr<CoreMLLayerCreator>>); });
    return *creators;
}

std::shared_ptr<CoreMLBaseLayer> CreateCoreMLBaseLayer(LayerType type) {
    std::shared_ptr<CoreMLBaseLayer> cur_layer = nullptr;
    auto &layer_creater_map   = GetGlobalCoreMLLayerCreatorMap();
    if (layer_creater_map.count(type) > 0) {
        cur_layer = layer_creater_map[type]->CreateCoreMLBaseLayer();
    }
    return cur_layer;
}

}  // namespace TNN_NS
