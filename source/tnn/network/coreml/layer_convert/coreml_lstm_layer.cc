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
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

DECLARE_COREML_LAYER_WITH_FUNC_DATA(LSTM, LAYER_LSTMONNX,
                                     virtual std::vector<CoreML__Specification__NeuralNetworkLayer*> GetCoreMLLayerPtrs();
                                    Status BuildUnsqueezeInputLayer();
                                    Status BuildUnsqueezeH0Layer();
                                    Status BuildUnsqueezeC0Layer();
                                    Status BuildSqueezeOutputLayer();
                                    Status BuildSqueezeHtLayer();
                                    Status BuildSqueezeCtLayer();,
                                    std::shared_ptr<CoreML__Specification__LSTMParams> lstm_param_;
                                    std::shared_ptr<CoreML__Specification__LSTMWeightParams> lstm_weight_param_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_W_i_;
                                    std::shared_ptr<CoreML__Specification__ActivationParams*> lstm_activations_ptrs_;
                                    std::shared_ptr<CoreML__Specification__ActivationParams> lstm_activations_;
                                    std::shared_ptr<CoreML__Specification__ActivationSigmoid> lstm_activation_sigmoid_;
                                    std::shared_ptr<CoreML__Specification__ActivationTanh> lstm_activation_tanh0_;
                                    std::shared_ptr<CoreML__Specification__ActivationTanh> lstm_activation_tanh1_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_W_i_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_W_o_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_W_o_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_W_f_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_W_f_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_W_c_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_W_c_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_R_i_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_R_i_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_R_o_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_R_o_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_R_f_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_R_f_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_R_c_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_R_c_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_WRBisa_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_WRBisa_i_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_WRBisa_i_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_WRBisa_o_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_WRBisa_o_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_WRBisa_f_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_WRBisa_f_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_WRBisa_c_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_WRBisa_c_;
                                    std::shared_ptr<LayerInfo> layer_info_unsqueeze_input_;
                                    std::shared_ptr<CoreMLBaseLayer> coreml_layer_unsquezze_input_;
                                    std::shared_ptr<LayerInfo> layer_info_unsqueeze_h0_;
                                    std::shared_ptr<CoreMLBaseLayer> coreml_layer_unsquezze_h0_;
                                    std::shared_ptr<LayerInfo> layer_info_unsqueeze_c0_;
                                    std::shared_ptr<CoreMLBaseLayer> coreml_layer_unsquezze_c0_;
                                    std::shared_ptr<LayerInfo> layer_info_squeeze_output_;
                                    std::shared_ptr<CoreMLBaseLayer> coreml_layer_squezze_output_;
                                    std::shared_ptr<LayerInfo> layer_info_squeeze_ht_;
                                    std::shared_ptr<CoreMLBaseLayer> coreml_layer_squezze_ht_;
                                    std::shared_ptr<LayerInfo> layer_info_squeeze_ct_;
                                    std::shared_ptr<CoreMLBaseLayer> coreml_layer_squezze_ct_;);

std::vector<CoreML__Specification__NeuralNetworkLayer*> CoreMLLSTMLayer::GetCoreMLLayerPtrs() {
    auto all_ptrs = CoreMLBaseLayer::GetCoreMLLayerPtrs();
    if (coreml_layer_unsquezze_input_) {
        auto ptrs = coreml_layer_unsquezze_input_->GetCoreMLLayerPtrs();
        all_ptrs.insert(all_ptrs.end(), ptrs.begin(), ptrs.end());
    }
    if (coreml_layer_unsquezze_h0_) {
        auto ptrs = coreml_layer_unsquezze_h0_->GetCoreMLLayerPtrs();
        all_ptrs.insert(all_ptrs.end(), ptrs.begin(), ptrs.end());
    }
    if (coreml_layer_unsquezze_c0_) {
        auto ptrs = coreml_layer_unsquezze_c0_->GetCoreMLLayerPtrs();
        all_ptrs.insert(all_ptrs.end(), ptrs.begin(), ptrs.end());
    }
    if (coreml_layer_squezze_output_) {
        auto ptrs = coreml_layer_squezze_output_->GetCoreMLLayerPtrs();
        all_ptrs.insert(all_ptrs.end(), ptrs.begin(), ptrs.end());
    }
    if (coreml_layer_squezze_ht_) {
        auto ptrs = coreml_layer_squezze_ht_->GetCoreMLLayerPtrs();
        all_ptrs.insert(all_ptrs.end(), ptrs.begin(), ptrs.end());
    }
    if (coreml_layer_squezze_ct_) {
        auto ptrs = coreml_layer_squezze_ct_->GetCoreMLLayerPtrs();
        all_ptrs.insert(all_ptrs.end(), ptrs.begin(), ptrs.end());
    }
    return all_ptrs;
}

Status CoreMLLSTMLayer::BuildLayerType() {
    auto param = dynamic_cast<LSTMONNXLayerParam *>(layer_info_->param.get());
    CHECK_PARAM_NULL(param);
    
    if (param->direction == 2) {
        return Status(TNNERR_COMMON_ERROR, "CoreMLLSTMLayer dont support bidirection LSTM");
    }
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_UNI_DIRECTIONAL_LSTM;
    return TNN_OK;
}

Status CoreMLLSTMLayer::BuildLayerParam() {
    auto param = dynamic_cast<LSTMONNXLayerParam *>(layer_info_->param.get());
    CHECK_PARAM_NULL(param);
    
    if (layer_info_->inputs.size() < 6 || !net_resource_) {
        return Status(TNNERR_LAYER_ERR, "CoreMLLSTMLayer has invalid inputs size");
    }
    
    RETURN_ON_NEQ(BuildUnsqueezeInputLayer(), TNN_OK);
    RETURN_ON_NEQ(BuildUnsqueezeH0Layer(), TNN_OK);
    RETURN_ON_NEQ(BuildUnsqueezeC0Layer(), TNN_OK);
    RETURN_ON_NEQ(BuildSqueezeOutputLayer(), TNN_OK);
    RETURN_ON_NEQ(BuildSqueezeHtLayer(), TNN_OK);
    RETURN_ON_NEQ(BuildSqueezeCtLayer(), TNN_OK);
    
    auto blob_name_W = layer_info_->inputs[1];
    auto blob_name_R = layer_info_->inputs[2];
    auto blob_name_B = layer_info_->inputs[3];
    auto blob_name_h0 = layer_info_->inputs[4];
    auto blob_name_c0 = layer_info_->inputs[5];
    
    //get input and output shape
    DimsVector shape_input;
    if (net_resource_->blob_shapes_map.find(layer_info_->inputs[0]) != net_resource_->blob_shapes_map.end()) {
        shape_input = net_resource_->blob_shapes_map[layer_info_->inputs[0]];
    }
    
    DimsVector shape_h0;
    if (net_resource_->blob_shapes_map.find(blob_name_h0) != net_resource_->blob_shapes_map.end()) {
        shape_h0 = net_resource_->blob_shapes_map[blob_name_h0];
    }
    
    DimsVector shape_c0;
    if (net_resource_->blob_shapes_map.find(blob_name_c0) != net_resource_->blob_shapes_map.end()) {
        shape_c0 = net_resource_->blob_shapes_map[blob_name_c0];
    }
    
    DimsVector shape_output;
    if (net_resource_->blob_shapes_map.find(layer_info_->outputs[0]) != net_resource_->blob_shapes_map.end()) {
        shape_output = net_resource_->blob_shapes_map[layer_info_->outputs[0]];
    }
    
    if (shape_input.size() <= 0 || shape_h0.size() <= 0) {
        LOGE("CoreMLLSTMLayer has no fixed input or output shape\n");
        return Status(TNNERR_MODEL_ERR, "CoreMLLSTMLayer has no fixed input or output shape");
    }
    
    std::shared_ptr<RawBuffer> buffer_W = nullptr;
    if (net_resource_->constant_map.find(blob_name_W) != net_resource_->constant_map.end()) {
        buffer_W = net_resource_->constant_map[blob_name_W];
    }
    
    std::shared_ptr<RawBuffer> buffer_R = nullptr;
    if (net_resource_->constant_map.find(blob_name_R) != net_resource_->constant_map.end()) {
        buffer_R = net_resource_->constant_map[blob_name_R];
    }
    
    std::shared_ptr<RawBuffer> buffer_B = nullptr;
    if (net_resource_->constant_map.find(blob_name_B) != net_resource_->constant_map.end()) {
        buffer_B = net_resource_->constant_map[blob_name_B];
    }
    
    if (!buffer_W || !buffer_R || !buffer_B) {
        LOGE("CoreMLLSTMLayer has empty weight\n");
        return Status(TNNERR_MODEL_ERR, "CoreMLLSTMLayer has empty weight");
    }
    
    const int input_size = shape_input.back();
    const int hidden_size = param->hidden_size;
    
    //set lstm param
    {
        coreml_layer_param_ = std::shared_ptr<CoreML__Specification__UniDirectionalLSTMLayerParams>(new CoreML__Specification__UniDirectionalLSTMLayerParams);
        coreml_layer_->unidirectionallstm = (CoreML__Specification__UniDirectionalLSTMLayerParams *)coreml_layer_param_.get();
        core_ml__specification__uni_directional_lstmlayer_params__init(coreml_layer_->unidirectionallstm);
        
        coreml_layer_->unidirectionallstm->inputvectorsize = input_size;
        coreml_layer_->unidirectionallstm->outputvectorsize = hidden_size;
        coreml_layer_->unidirectionallstm->reverseinput = param->direction == 1;
        
        //activation
        coreml_layer_->unidirectionallstm->n_activations = 3;
        {
            lstm_activations_ptrs_ = std::shared_ptr<CoreML__Specification__ActivationParams*>(new CoreML__Specification__ActivationParams* [3],
                                                                      [](CoreML__Specification__ActivationParams** p) { delete [] p; });
            auto lstm_activations_ptrs_temp = lstm_activations_ptrs_.get();
            lstm_activations_ =  std::shared_ptr<CoreML__Specification__ActivationParams>(new CoreML__Specification__ActivationParams [3],
                                                                                          [](CoreML__Specification__ActivationParams* p) { delete[] p; });
            auto lstm_activations_temp = lstm_activations_.get();
            { //sigmoid
                lstm_activations_ptrs_temp[0] = lstm_activations_temp;
                auto activation_sigmoid = lstm_activations_ptrs_temp[0];
                core_ml__specification__activation_params__init(activation_sigmoid);
                activation_sigmoid->nonlinearity_type_case = CORE_ML__SPECIFICATION__ACTIVATION_PARAMS__NONLINEARITY_TYPE_SIGMOID;
                lstm_activation_sigmoid_ = std::shared_ptr<CoreML__Specification__ActivationSigmoid>(new CoreML__Specification__ActivationSigmoid);
                activation_sigmoid->sigmoid = lstm_activation_sigmoid_.get();
                core_ml__specification__activation_sigmoid__init(activation_sigmoid->sigmoid);
            }
            { //tanh
                lstm_activations_ptrs_temp[1] = lstm_activations_temp + 1;
                auto activation_tanh = lstm_activations_ptrs_temp[1];
                core_ml__specification__activation_params__init(activation_tanh);
                activation_tanh->nonlinearity_type_case = CORE_ML__SPECIFICATION__ACTIVATION_PARAMS__NONLINEARITY_TYPE_TANH;
                lstm_activation_tanh0_ = std::shared_ptr<CoreML__Specification__ActivationTanh>(new CoreML__Specification__ActivationTanh);
                activation_tanh->tanh = lstm_activation_tanh0_.get();
                core_ml__specification__activation_tanh__init(activation_tanh->tanh);
            }
            { //tanh
                lstm_activations_ptrs_temp[2] = lstm_activations_temp + 2;
                auto activation_tanh = lstm_activations_ptrs_temp[2];
                core_ml__specification__activation_params__init(activation_tanh);
                activation_tanh->nonlinearity_type_case = CORE_ML__SPECIFICATION__ACTIVATION_PARAMS__NONLINEARITY_TYPE_TANH;
                lstm_activation_tanh1_ = std::shared_ptr<CoreML__Specification__ActivationTanh>(new CoreML__Specification__ActivationTanh);
                activation_tanh->tanh = lstm_activation_tanh1_.get();
                core_ml__specification__activation_tanh__init(activation_tanh->tanh);
            }
            coreml_layer_->unidirectionallstm->activations = lstm_activations_ptrs_.get();
        }
        
        //param
        lstm_param_ = std::shared_ptr<CoreML__Specification__LSTMParams>(new CoreML__Specification__LSTMParams);
        core_ml__specification__lstmparams__init(lstm_param_.get());
        coreml_layer_->unidirectionallstm->params = lstm_param_.get();
        lstm_param_->sequenceoutput = shape_input.front() > 1;
        lstm_param_->hasbiasvectors = true;
        
        //weight
        lstm_weight_param_ = std::shared_ptr<CoreML__Specification__LSTMWeightParams>(new CoreML__Specification__LSTMWeightParams);
        core_ml__specification__lstmweight_params__init(lstm_weight_param_.get());
        coreml_layer_->unidirectionallstm->weightparams = lstm_weight_param_.get();
        //W[iofc], weight tensor for the gates, shape [num_directions, 4*hidden_size, input_size]
        {
            const int data_page_size = hidden_size * input_size;
            DimsVector data_dims = {hidden_size, input_size};
            auto data_type = buffer_W->GetDataType();
            const int byte_size = DataTypeUtils::GetBytesSize(data_type);
            char *data_ptr = buffer_W->force_to<char *>();
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(data_page_size, (void *)data_ptr, data_type, data_dims,
                                                 lstm_weight_W_i_, rawbuffer_fp32_W_i_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(data_page_size, (void *)(data_ptr + data_page_size*byte_size), data_type, data_dims,
                                                 lstm_weight_W_o_, rawbuffer_fp32_W_o_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(data_page_size, (void *)(data_ptr + 2*data_page_size*byte_size), data_type, data_dims,
                                                 lstm_weight_W_f_, rawbuffer_fp32_W_f_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(data_page_size, (void *)(data_ptr + 3*data_page_size*byte_size), data_type, data_dims,
                                                 lstm_weight_W_c_, rawbuffer_fp32_W_c_), TNN_OK);
            
            lstm_weight_param_->inputgateweightmatrix = lstm_weight_W_i_.get();
            lstm_weight_param_->outputgateweightmatrix = lstm_weight_W_o_.get();
            lstm_weight_param_->forgetgateweightmatrix = lstm_weight_W_f_.get();
            lstm_weight_param_->blockinputweightmatrix = lstm_weight_W_c_.get();
        }
        
        //R[iofc], recurrence weight tensor, shape [num_directions, 4*hidden_size, hidden_size]
        {
            const int data_page_size = hidden_size * hidden_size;
            DimsVector data_dims = {hidden_size, hidden_size};
            auto data_type = buffer_R->GetDataType();
            const int byte_size = DataTypeUtils::GetBytesSize(data_type);
            char *data_ptr = buffer_R->force_to<char *>();
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(data_page_size, (void *)data_ptr, data_type, data_dims,
                                                 lstm_weight_R_i_, rawbuffer_fp32_R_i_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(data_page_size, (void *)(data_ptr + data_page_size*byte_size), data_type, data_dims,
                                                 lstm_weight_R_o_, rawbuffer_fp32_R_o_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(data_page_size, (void *)(data_ptr + 2*data_page_size*byte_size), data_type, data_dims,
                                                 lstm_weight_R_f_, rawbuffer_fp32_R_f_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(data_page_size, (void *)(data_ptr + 3*data_page_size*byte_size), data_type, data_dims,
                                                 lstm_weight_R_c_, rawbuffer_fp32_R_c_), TNN_OK);
            
            lstm_weight_param_->inputgaterecursionmatrix = lstm_weight_R_i_.get();
            lstm_weight_param_->outputgaterecursionmatrix = lstm_weight_R_o_.get();
            lstm_weight_param_->forgetgaterecursionmatrix = lstm_weight_R_f_.get();
            lstm_weight_param_->blockinputrecursionmatrix = lstm_weight_R_c_.get();
        }
        
        //B[iofc] Concatenation of [Wb[iofc], Rb[iofc]], [num_directions, 8*hidden_size]
        {
            const DimsVector data_dims = {hidden_size};
            auto data_count = buffer_B->GetDataCount();
            auto data_type = buffer_B->GetDataType();
            const int byte_size = DataTypeUtils::GetBytesSize(data_type);
            char *data_ptr = buffer_B->force_to<char *>();
            
            rawbuffer_fp32_WRBisa_ = shared_ptr<RawBuffer>(new RawBuffer(data_count*sizeof(float), {4, hidden_size}));
            float *data_fp32_ptr = rawbuffer_fp32_WRBisa_->force_to<float *>();
            if (data_type == DATA_TYPE_HALF) {
                RETURN_ON_NEQ(ConvertFromHalfToFloat((void *)data_ptr, (float *)data_fp32_ptr, data_count),TNN_OK);
            } else {
                memcpy(data_fp32_ptr, data_ptr, data_count*byte_size);
            }
            
            for (int index = 0; index<4*hidden_size; index++) {
                data_fp32_ptr[index] += data_fp32_ptr[index + 4*hidden_size];
            }
            
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(hidden_size, (void *)data_fp32_ptr, DATA_TYPE_FLOAT, data_dims,
                                                 lstm_weight_WRBisa_i_, rawbuffer_fp32_WRBisa_i_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(hidden_size, (void *)(data_fp32_ptr + hidden_size), DATA_TYPE_FLOAT, data_dims,
                                                 lstm_weight_WRBisa_o_, rawbuffer_fp32_WRBisa_o_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(hidden_size, (void *)(data_fp32_ptr + 2*hidden_size), DATA_TYPE_FLOAT, data_dims,
                                                 lstm_weight_WRBisa_f_, rawbuffer_fp32_WRBisa_f_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(hidden_size, (void *)(data_fp32_ptr + 3*hidden_size), DATA_TYPE_FLOAT, data_dims,
                                                 lstm_weight_WRBisa_c_, rawbuffer_fp32_WRBisa_c_), TNN_OK);
            
            lstm_weight_param_->inputgatebiasvector = lstm_weight_WRBisa_i_.get();
            lstm_weight_param_->outputgatebiasvector = lstm_weight_WRBisa_o_.get();
            lstm_weight_param_->forgetgatebiasvector = lstm_weight_WRBisa_f_.get();
            lstm_weight_param_->blockinputbiasvector = lstm_weight_WRBisa_c_.get();
        }
    }
    
    return TNN_OK;
}

Status CoreMLLSTMLayer::BuildUnsqueezeInputLayer() {
    auto unsqueeze_layer = CreateCoreMLBaseLayer(LAYER_UNSQUEEZE);
    if (!unsqueeze_layer) {
        LOGE("Error: CreateCoreMLBaseLayer failed\n");
        return Status(TNNERR_PARAM_ERR, "Error: CreateCoreMLBaseLayer failed");
    }
    unsqueeze_layer->SetNetResource(net_resource_);
    
    layer_info_unsqueeze_input_ = std::shared_ptr<LayerInfo>(new LayerInfo);
    auto param_unsqueeze_input = std::shared_ptr<UnsqueezeLayerParam>(new UnsqueezeLayerParam);
    {
        layer_info_unsqueeze_input_->type = LAYER_UNSQUEEZE;
        layer_info_unsqueeze_input_->name = layer_info_->name + "-unsqueeze-input";
        layer_info_unsqueeze_input_->name = "input_expanded";
        layer_info_unsqueeze_input_->inputs = {layer_info_->inputs[0]};
        layer_info_unsqueeze_input_->outputs = {layer_info_unsqueeze_input_->name};
        layer_info_unsqueeze_input_->param = param_unsqueeze_input;
        {
            param_unsqueeze_input->type = layer_info_unsqueeze_input_->type;
            param_unsqueeze_input->name = layer_info_unsqueeze_input_->name;
            param_unsqueeze_input->axes = {3, 4};
        }
    }
    RETURN_ON_NEQ(unsqueeze_layer->Init(layer_info_unsqueeze_input_.get(), nullptr),  TNN_OK);
    
    coreml_layer_unsquezze_input_ = unsqueeze_layer;
    return TNN_OK;
}

Status CoreMLLSTMLayer::BuildUnsqueezeH0Layer() {
    auto unsqueeze_layer = CreateCoreMLBaseLayer(LAYER_UNSQUEEZE);
    if (!unsqueeze_layer) {
        LOGE("Error: CreateCoreMLBaseLayer failed\n");
        return Status(TNNERR_PARAM_ERR, "Error: CreateCoreMLBaseLayer failed");
    }
    unsqueeze_layer->SetNetResource(net_resource_);
    
    layer_info_unsqueeze_h0_ = std::shared_ptr<LayerInfo>(new LayerInfo);
    auto param_unsqueeze_input = std::shared_ptr<UnsqueezeLayerParam>(new UnsqueezeLayerParam);
    {
        layer_info_unsqueeze_h0_->type = LAYER_UNSQUEEZE;
        layer_info_unsqueeze_h0_->name = layer_info_->name + "-unsqueeze-h0";
        layer_info_unsqueeze_h0_->inputs = {layer_info_->inputs[4]};
        layer_info_unsqueeze_h0_->outputs = {layer_info_unsqueeze_h0_->name};
        layer_info_unsqueeze_h0_->param = param_unsqueeze_input;
        {
            param_unsqueeze_input->type = layer_info_unsqueeze_h0_->type;
            param_unsqueeze_input->name = layer_info_unsqueeze_h0_->name;
            param_unsqueeze_input->axes = {3, 4};
        }
    }
    RETURN_ON_NEQ(unsqueeze_layer->Init(layer_info_unsqueeze_h0_.get(), nullptr),  TNN_OK);
    
    coreml_layer_unsquezze_h0_ = unsqueeze_layer;
    return TNN_OK;
}

Status CoreMLLSTMLayer::BuildUnsqueezeC0Layer() {
    auto unsqueeze_layer = CreateCoreMLBaseLayer(LAYER_UNSQUEEZE);
    if (!unsqueeze_layer) {
        LOGE("Error: CreateCoreMLBaseLayer failed\n");
        return Status(TNNERR_PARAM_ERR, "Error: CreateCoreMLBaseLayer failed");
    }
    unsqueeze_layer->SetNetResource(net_resource_);
    
    layer_info_unsqueeze_c0_ = std::shared_ptr<LayerInfo>(new LayerInfo);
    auto param_unsqueeze_input = std::shared_ptr<UnsqueezeLayerParam>(new UnsqueezeLayerParam);
    {
        layer_info_unsqueeze_c0_->type = LAYER_UNSQUEEZE;
        layer_info_unsqueeze_c0_->name = layer_info_->name + "-unsqueeze-c0";
        layer_info_unsqueeze_c0_->inputs = {layer_info_->inputs[5]};
        layer_info_unsqueeze_c0_->outputs = {layer_info_unsqueeze_c0_->name};
        layer_info_unsqueeze_c0_->param = param_unsqueeze_input;
        {
            param_unsqueeze_input->type = layer_info_unsqueeze_c0_->type;
            param_unsqueeze_input->name = layer_info_unsqueeze_c0_->name;
            param_unsqueeze_input->axes = {3, 4};
        }
    }
    RETURN_ON_NEQ(unsqueeze_layer->Init(layer_info_unsqueeze_c0_.get(), nullptr),  TNN_OK);
    
    coreml_layer_unsquezze_c0_ = unsqueeze_layer;
    return TNN_OK;
}

Status CoreMLLSTMLayer::BuildSqueezeOutputLayer() {
    auto squeeze_layer = CreateCoreMLBaseLayer(LAYER_SQUEEZE);
    if (!squeeze_layer) {
        LOGE("Error: CreateCoreMLBaseLayer failed\n");
        return Status(TNNERR_PARAM_ERR, "Error: CreateCoreMLBaseLayer failed");
    }
    squeeze_layer->SetNetResource(net_resource_);
    
    layer_info_squeeze_output_ = std::shared_ptr<LayerInfo>(new LayerInfo);
    auto param_squeeze_output = std::shared_ptr<UnsqueezeLayerParam>(new UnsqueezeLayerParam);
    {
        layer_info_squeeze_output_->type = LAYER_UNSQUEEZE;
        layer_info_squeeze_output_->name = layer_info_->name + "-squeeze-output";
        layer_info_squeeze_output_->inputs = {layer_info_squeeze_output_->name};
        layer_info_squeeze_output_->outputs = {layer_info_->outputs[0]};
        layer_info_squeeze_output_->param = param_squeeze_output;
        {
            param_squeeze_output->type = layer_info_squeeze_output_->type;
            param_squeeze_output->name = layer_info_squeeze_output_->name;
            param_squeeze_output->axes = {3, 4};
        }
    }
    RETURN_ON_NEQ(squeeze_layer->Init(layer_info_squeeze_output_.get(), nullptr),  TNN_OK);
    
    coreml_layer_squezze_output_ = squeeze_layer;
    return TNN_OK;
}

Status CoreMLLSTMLayer::BuildSqueezeHtLayer() {
    auto squeeze_layer = CreateCoreMLBaseLayer(LAYER_SQUEEZE);
    if (!squeeze_layer) {
        LOGE("Error: CreateCoreMLBaseLayer failed\n");
        return Status(TNNERR_PARAM_ERR, "Error: CreateCoreMLBaseLayer failed");
    }
    squeeze_layer->SetNetResource(net_resource_);
    
    layer_info_squeeze_ht_ = std::shared_ptr<LayerInfo>(new LayerInfo);
    auto param_squeeze_output = std::shared_ptr<UnsqueezeLayerParam>(new UnsqueezeLayerParam);
    {
        layer_info_squeeze_ht_->type = LAYER_UNSQUEEZE;
        layer_info_squeeze_ht_->name = layer_info_->name + "-squeeze-ht";
        layer_info_squeeze_ht_->inputs = {layer_info_squeeze_ht_->name};
        layer_info_squeeze_ht_->outputs = {layer_info_->outputs[1]};
        layer_info_squeeze_ht_->param = param_squeeze_output;
        {
            param_squeeze_output->type = layer_info_squeeze_output_->type;
            param_squeeze_output->name = layer_info_squeeze_output_->name;
            param_squeeze_output->axes = {3, 4};
        }
    }
    RETURN_ON_NEQ(squeeze_layer->Init(layer_info_squeeze_ht_.get(), nullptr),  TNN_OK);
    
    coreml_layer_squezze_ht_ = squeeze_layer;
    return TNN_OK;
}

Status CoreMLLSTMLayer::BuildSqueezeCtLayer() {
    auto squeeze_layer = CreateCoreMLBaseLayer(LAYER_SQUEEZE);
    if (!squeeze_layer) {
        LOGE("Error: CreateCoreMLBaseLayer failed\n");
        return Status(TNNERR_PARAM_ERR, "Error: CreateCoreMLBaseLayer failed");
    }
    squeeze_layer->SetNetResource(net_resource_);
    
    layer_info_squeeze_ct_ = std::shared_ptr<LayerInfo>(new LayerInfo);
    auto param_squeeze_output = std::shared_ptr<UnsqueezeLayerParam>(new UnsqueezeLayerParam);
    {
        layer_info_squeeze_ct_->type = LAYER_UNSQUEEZE;
        layer_info_squeeze_ct_->name = layer_info_->name + "-squeeze-ct";
        layer_info_squeeze_ct_->inputs = {layer_info_squeeze_ct_->name};
        layer_info_squeeze_ct_->outputs = {layer_info_->outputs[2]};
        layer_info_squeeze_ct_->param = param_squeeze_output;
        {
            param_squeeze_output->type = layer_info_squeeze_ct_->type;
            param_squeeze_output->name = layer_info_squeeze_ct_->name;
            param_squeeze_output->axes = {3, 4};
        }
    }
    RETURN_ON_NEQ(squeeze_layer->Init(layer_info_squeeze_ct_.get(), nullptr),  TNN_OK);
    
    coreml_layer_squezze_ct_ = squeeze_layer;
    return TNN_OK;
}

Status CoreMLLSTMLayer::BuildConstantWeightsLayer() {
    return TNN_OK;
//    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLLSTMLayer::BuildLayerInputs() {
    return {layer_info_unsqueeze_input_->outputs[0],
        layer_info_unsqueeze_h0_->outputs[0],
        layer_info_unsqueeze_c0_->outputs[0]
    };
}

std::vector<std::string> CoreMLLSTMLayer::BuildLayerOutputs() {
    return {layer_info_squeeze_output_->inputs[0],
        layer_info_squeeze_ht_->inputs[0],
        layer_info_squeeze_ct_->inputs[0]
    };
}

REGISTER_COREML_LAYER(LSTM, LAYER_LSTMONNX);

}  // namespace TNN_NS
