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
#include "tnn/utils/data_type_utils.h"

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

Status RawBuffer2CoreMLWeight(RawBuffer *rawbuffer,
                              shared_ptr<CoreML__Specification__WeightParams> &coreml_weight, shared_ptr<RawBuffer> &rawbuffer_fp32) {
    return RawBuffer2CoreMLWeight(rawbuffer->GetDataCount(), rawbuffer->force_to<void *>(), rawbuffer->GetDataType(), rawbuffer->GetBufferDims(),
                                  coreml_weight, rawbuffer_fp32);
}

Status RawBuffer2CoreMLWeight(int data_count, void *data_ptr, DataType data_type, DimsVector data_dims,
                              shared_ptr<CoreML__Specification__WeightParams> &coreml_weight, shared_ptr<RawBuffer> &rawbuffer_fp32) {
    coreml_weight = std::shared_ptr<CoreML__Specification__WeightParams>(new CoreML__Specification__WeightParams);
    core_ml__specification__weight_params__init(coreml_weight.get());
    
    const int byte_size = DataTypeUtils::GetBytesSize(data_type);
    
    //TODO: to chcek data type
    switch (data_type) {
        case DATA_TYPE_FLOAT:
        {
            coreml_weight->n_floatvalue = data_count;
            coreml_weight->floatvalue = (float *)data_ptr;
        }
            break;
        case DATA_TYPE_INT32:
            {
                //CoreML only support FP32, so we need convert int32 to fp32
                rawbuffer_fp32 = shared_ptr<RawBuffer>(new RawBuffer(data_count*sizeof(float), data_dims));
                float *data_fp32_ptr = rawbuffer_fp32->force_to<float *>();
                int *int32_data = (int *)data_ptr;
                for (int i=0; i<data_count; i++) {
                    data_fp32_ptr[i] = int32_data[i];
                }
                coreml_weight->n_floatvalue = data_count;
                coreml_weight->floatvalue = data_fp32_ptr;
            }
            break;
        case DATA_TYPE_HALF:
            {
#if TNN_COREML_FULL_PRECISION
                rawbuffer_fp32 = shared_ptr<RawBuffer>(new RawBuffer(data_count*sizeof(float), data_dims));
                float *data_fp32_ptr = rawbuffer_fp32->force_to<float *>();
                RETURN_ON_NEQ(ConvertFromHalfToFloat((void *)data_ptr, (float *)data_fp32_ptr, data_count),TNN_OK);
                
                coreml_weight->n_floatvalue = data_count;
                coreml_weight->floatvalue = data_fp32_ptr;
#else
                coreml_weight->float16value.len = data_count*byte_size;
                coreml_weight->float16value.data = (uint8_t *)data_ptr;
#endif
            }
            break;
        default:
            {
                LOGE("RawBuffer2CoreMLWeight dont support data type (%d)\n", data_type);
                return Status(TNNERR_PARAM_ERR, "RawBuffer2CoreMLWeight dont support data type");
            }
            break;
    }
    return TNN_OK;
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
    //Note layer_ptrs must be added by compute order, otherwise mlmodel compiling error wil raise.
    //e.g.  protobuf spec. validator error: Layer '39' consumes an input named 'input_expanded' which is not present in this network.
    std::vector<CoreML__Specification__NeuralNetworkLayer*> layer_ptrs;
    for (auto& iter : coreml_layer_constant_weights_) {
        auto const_ptr = iter->GetCoreMLLayerPtrs();
        layer_ptrs.insert(layer_ptrs.end(), const_ptr.begin(), const_ptr.end());
    }
    
    for (auto iter : coreml_layers_before_) {
        auto before_layer_ptr = iter->GetCoreMLLayerPtrs();
        layer_ptrs.insert(layer_ptrs.end(), before_layer_ptr.begin(), before_layer_ptr.end());
    }
    
    if (coreml_layer_) {
        layer_ptrs.push_back(coreml_layer_.get());
    }
    
    for (auto iter : coreml_layers_after_) {
        auto after_layer_ptr = iter->GetCoreMLLayerPtrs();
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
    //dont create constantlayer in CoreMLBaseLayer, do it in each layer's BuildConstantWeightsLayer
    //because some layer use constant in constant_map for layer resource, we dont need create a constant layer, see LSTM
    
    //weight in constantmap
    if (!layer_info_ || !net_resource_) {
        LOGE("CoreMLBaseLayer has invalid layer info or net resource\n");
        return Status(TNNERR_MODEL_ERR, "CoreMLBaseLayer has invalid layer info or net resource");
    }
    return BuildConstantWeightsLayer(layer_info_->inputs);
}

Status CoreMLBaseLayer::BuildConstantWeightsLayer(std::vector<std::string> const_names) {
    for (auto iter : const_names) {
        //only load data blob with flag DATA_FLAG_CHANGE_NEVER, ignore DATA_FLAG_CHANGE_IF_SHAPE_DIFFER
        if (net_resource_->constant_blob_flags.find(iter) != net_resource_->constant_blob_flags.end()) {
            auto blob_flag = net_resource_->constant_blob_flags[iter];
            if (blob_flag != DATA_FLAG_CHANGE_NEVER) {
                continue;
            }
        }

        if (net_resource_->constant_map.find(iter) != net_resource_->constant_map.end()) {
            auto weight_buffer = net_resource_->constant_map[iter];
            auto weight_layer = std::make_shared<CoreMLConstLayer>(LAYER_CONST);
            auto status = weight_layer->Init(iter, *(weight_buffer.get()));
            RETURN_ON_NEQ(status, TNN_OK);

            coreml_layer_constant_weights_.push_back(weight_layer);
        }
    }
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

void CoreMLBaseLayer::SetLayerName(std::string name) {
    coreml_layer_name_ = NullTerminatedCString(name);
    if (coreml_layer_) {
        coreml_layer_->name = coreml_layer_name_.get();
    }
 }

std::string CoreMLBaseLayer::GetLayerName() {
    if (coreml_layer_name_) {
        return coreml_layer_name_.get();
    }
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
