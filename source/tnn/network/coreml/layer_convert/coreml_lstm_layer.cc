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
#include "coreml_const_layer.h"

#if 1
namespace TNN_NS {

DECLARE_COREML_LAYER_WITH_FUNC_DATA(LSTM, LAYER_LSTMONNX,
                                    Status BuildSplitLayer(std::string input, std::vector<std::string> outputs,
                                                               std::shared_ptr<CoreMLBaseLayer> &coreml_layer, std::shared_ptr<LayerInfo> &layer_info);
                                    Status BuildReshapeLayer(std::string input, std::string output,
                                                               std::shared_ptr<CoreMLBaseLayer> &coreml_layer, std::shared_ptr<LayerInfo> &layer_info);
                                    Status BuildConcatLayer(std::vector<std::string> inputs, std::string output,
                                                               std::shared_ptr<CoreMLBaseLayer> &coreml_layer, std::shared_ptr<LayerInfo> &layer_info);
                                    Status BuildSqueezeLayer(std::string input, std::string output,
                                                               std::shared_ptr<CoreMLBaseLayer> &coreml_layer, std::shared_ptr<LayerInfo> &layer_info);,
                                    std::shared_ptr<CoreML__Specification__LSTMParams> lstm_param_;
                                    std::shared_ptr<CoreML__Specification__LSTMWeightParams*> lstm_weight_param_ptrs_;
                                    std::shared_ptr<CoreML__Specification__LSTMWeightParams> lstm_weight_param_;
                                    std::shared_ptr<CoreML__Specification__ActivationParams*> lstm_activations_ptrs_;
                                    std::shared_ptr<CoreML__Specification__ActivationParams> lstm_activations_;
                                    std::shared_ptr<CoreML__Specification__ActivationSigmoid> lstm_activation_sigmoid_;
                                    std::shared_ptr<CoreML__Specification__ActivationTanh> lstm_activation_tanh0_;
                                    std::shared_ptr<CoreML__Specification__ActivationTanh> lstm_activation_tanh1_;
                                    std::shared_ptr<CoreML__Specification__ActivationSigmoid> lstm_activation_backword_sigmoid_;
                                    std::shared_ptr<CoreML__Specification__ActivationTanh> lstm_activation_backword_tanh0_;
                                    std::shared_ptr<CoreML__Specification__ActivationTanh> lstm_activation_backword_tanh1_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_W_i_;
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
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_backword_W_i_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_backword_W_i_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_backword_W_o_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_backword_W_o_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_backword_W_f_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_backword_W_f_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_backword_W_c_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_backword_W_c_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_backword_R_i_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_backword_R_i_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_backword_R_o_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_backword_R_o_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_backword_R_f_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_backword_R_f_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_backword_R_c_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_backword_R_c_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_WRBisa_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_WRBisa_i_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_WRBisa_i_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_WRBisa_o_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_WRBisa_o_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_WRBisa_f_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_WRBisa_f_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_WRBisa_c_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_WRBisa_c_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_backword_WRBisa_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_backword_WRBisa_i_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_backword_WRBisa_i_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_backword_WRBisa_o_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_backword_WRBisa_o_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_backword_WRBisa_f_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_backword_WRBisa_f_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_backword_WRBisa_c_;
                                    shared_ptr<RawBuffer> rawbuffer_fp32_backword_WRBisa_c_;
                                    std::shared_ptr<LayerInfo> layer_info_reshape_input_;
                                    std::shared_ptr<LayerInfo> layer_info_split_h0_;
                                    std::shared_ptr<LayerInfo> layer_info_reshape_h0_;
                                    std::shared_ptr<LayerInfo> layer_info_split_c0_;
                                    std::shared_ptr<LayerInfo> layer_info_reshape_c0_;
                                    std::shared_ptr<LayerInfo> layer_info_reshape_backword_h0_;
                                    std::shared_ptr<LayerInfo> layer_info_reshape_backword_c0_;
                                    std::shared_ptr<LayerInfo> layer_info_squeeze_output_;
                                    std::shared_ptr<LayerInfo> layer_info_squeeze_ht_;
                                    std::shared_ptr<LayerInfo> layer_info_concat_ht_;
                                    std::shared_ptr<LayerInfo> layer_info_concat_ct_;
                                    std::shared_ptr<LayerInfo> layer_info_squeeze_ct_;
                                    std::shared_ptr<LayerInfo> layer_info_squeeze_backword_ht_;
                                    std::shared_ptr<LayerInfo> layer_info_squeeze_backword_ct_;);

Status CoreMLLSTMLayer::BuildLayerType() {
    auto param = dynamic_cast<LSTMONNXLayerParam *>(layer_info_->param.get());
    CHECK_PARAM_NULL(param);
    
    //layer type
    if (param->direction == 0) {
        coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_UNI_DIRECTIONAL_LSTM;
    } else if (param->direction == 1) {
        coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_UNI_DIRECTIONAL_LSTM;
    } else if (param->direction == 2) {
        coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_BI_DIRECTIONAL_LSTM;
    } else  {
        return Status(TNNERR_COMMON_ERROR, "CoreMLLSTMLayer dont support reverse LSTM");
    }

    return TNN_OK;
}

/*
 *NOTE1:
 *CoreML now only support LSTM at CPU device at 2022.07.19.
 *Both optiones MLComputeUnitsCPUOnly and MLComputeUnitsAll have the same benchmark time for model crnn_lite_lstm
 * And both are  much more slower than TNN arm.
 *
 *NOTE2:
 *CoreML bidirection LSTM always produce wrong result for the second slice of ht and ct.
 *We test it with two seperate uniLSTM, not correct either
*/
Status CoreMLLSTMLayer::BuildLayerParam() {
    auto param = dynamic_cast<LSTMONNXLayerParam *>(layer_info_->param.get());
    CHECK_PARAM_NULL(param);
    
    if (layer_info_->inputs.size() < 6 || !net_resource_) {
        return Status(TNNERR_LAYER_ERR, "CoreMLLSTMLayer has invalid inputs size");
    }
    
    if (param->direction == 0 || param->direction == 1) {
        std::shared_ptr<CoreMLBaseLayer> coreml_layer_reshape_input, coreml_layer_reshape_h0, coreml_layer_reshape_c0;
        RETURN_ON_NEQ(BuildReshapeLayer(layer_info_->inputs[0], layer_info_->name + "-reshape-input",
                                        coreml_layer_reshape_input, layer_info_reshape_input_), TNN_OK);
        RETURN_ON_NEQ(BuildReshapeLayer(layer_info_->inputs[4], layer_info_->name + "-reshape-h0",
                                        coreml_layer_reshape_h0, layer_info_reshape_h0_), TNN_OK);
        RETURN_ON_NEQ(BuildReshapeLayer(layer_info_->inputs[5], layer_info_->name + "-reshape-c0",
                                        coreml_layer_reshape_c0, layer_info_reshape_c0_), TNN_OK);
        coreml_layers_before_ = {coreml_layer_reshape_input, coreml_layer_reshape_h0, coreml_layer_reshape_c0};
        
        std::shared_ptr<CoreMLBaseLayer> coreml_layer_squezze_output, coreml_layer_squezze_ht, coreml_layer_squezze_ct;
        RETURN_ON_NEQ(BuildSqueezeLayer(layer_info_->name + "-squeeze-output", layer_info_->outputs[0],
                                        coreml_layer_squezze_output, layer_info_squeeze_output_), TNN_OK);
        coreml_layers_after_ = {coreml_layer_squezze_output};
        if (layer_info_->outputs.size() >= 3) {
            RETURN_ON_NEQ(BuildSqueezeLayer(layer_info_->name + "-squeeze-ht", layer_info_->outputs[1],
                                            coreml_layer_squezze_ht, layer_info_squeeze_ht_), TNN_OK);
            RETURN_ON_NEQ(BuildSqueezeLayer(layer_info_->name + "-squeeze-ct", layer_info_->outputs[2],
                                            coreml_layer_squezze_ct, layer_info_squeeze_ct_), TNN_OK);
            coreml_layers_after_ = {coreml_layer_squezze_output, coreml_layer_squezze_ht, coreml_layer_squezze_ct};
        }
    } else if (param->direction == 2) {
        std::shared_ptr<CoreMLBaseLayer> coreml_layer_reshape_input, coreml_layer_split_h0, coreml_layer_split_c0, coreml_layer_reshape_h0,coreml_layer_reshape_backword_h0,coreml_layer_reshape_c0,coreml_layer_reshape_backword_c0;
        
        RETURN_ON_NEQ(BuildReshapeLayer(layer_info_->inputs[0], layer_info_->name + "-reshape-input",
                                          coreml_layer_reshape_input, layer_info_reshape_input_), TNN_OK);
        RETURN_ON_NEQ(BuildSplitLayer(layer_info_->inputs[4], {layer_info_->name + "-split-h0", layer_info_->name + "-split-backword-h0"}, coreml_layer_split_h0, layer_info_split_h0_), TNN_OK);
        RETURN_ON_NEQ(BuildSplitLayer(layer_info_->inputs[5], {layer_info_->name + "-split-c0", layer_info_->name + "-split-backword-c0"}, coreml_layer_split_c0, layer_info_split_c0_), TNN_OK);
        RETURN_ON_NEQ(BuildReshapeLayer(layer_info_split_h0_->outputs[0], layer_info_->name + "-reshape-h0",
                                          coreml_layer_reshape_h0, layer_info_reshape_h0_), TNN_OK);
        RETURN_ON_NEQ(BuildReshapeLayer(layer_info_split_h0_->outputs[1], layer_info_->name + "-reshape-backword-h0",
                                          coreml_layer_reshape_backword_h0, layer_info_reshape_backword_h0_), TNN_OK);
        RETURN_ON_NEQ(BuildReshapeLayer(layer_info_split_c0_->outputs[0], layer_info_->name + "-reshape-c0",
                                          coreml_layer_reshape_c0, layer_info_reshape_c0_), TNN_OK);
        RETURN_ON_NEQ(BuildReshapeLayer(layer_info_split_c0_->outputs[1], layer_info_->name + "-reshape-backword-c0",
                                          coreml_layer_reshape_backword_c0, layer_info_reshape_backword_c0_), TNN_OK);
        coreml_layers_before_ = {coreml_layer_reshape_input, coreml_layer_split_h0, coreml_layer_split_c0,
            coreml_layer_reshape_h0,coreml_layer_reshape_backword_h0,coreml_layer_reshape_c0,coreml_layer_reshape_backword_c0};
        
        std::shared_ptr<CoreMLBaseLayer> coreml_layer_squezze_output, coreml_layer_concat_ht,
            coreml_layer_concat_ct, coreml_layer_squezze_ht,coreml_layer_squezze_ct;
        RETURN_ON_NEQ(BuildSqueezeLayer(layer_info_->name + "-squeeze-output", layer_info_->outputs[0],
                                        coreml_layer_squezze_output, layer_info_squeeze_output_), TNN_OK);
        coreml_layers_after_ = {coreml_layer_squezze_output, };
        if (layer_info_->outputs.size() >= 3) {
            RETURN_ON_NEQ(BuildConcatLayer({layer_info_->name + "-concat-input-ht", layer_info_->name + "-concat-input-backword-ht"}, layer_info_->name + "-concat-ht",
                                              coreml_layer_concat_ht, layer_info_concat_ht_), TNN_OK);
            RETURN_ON_NEQ(BuildConcatLayer({layer_info_->name + "-concat-input-ct", layer_info_->name + "-concat-input-backword-ct"}, layer_info_->name + "-concat-ct",
                                              coreml_layer_concat_ct, layer_info_concat_ct_), TNN_OK);
            RETURN_ON_NEQ(BuildSqueezeLayer(layer_info_concat_ht_->outputs[0], layer_info_->outputs[1],
                                            coreml_layer_squezze_ht, layer_info_squeeze_ht_), TNN_OK);
            RETURN_ON_NEQ(BuildSqueezeLayer(layer_info_concat_ct_->outputs[0], layer_info_->outputs[2],
                                            coreml_layer_squezze_ct, layer_info_squeeze_ct_), TNN_OK);
            coreml_layers_after_ = {coreml_layer_squezze_output,
                coreml_layer_concat_ht, coreml_layer_concat_ct,
                coreml_layer_squezze_ht,coreml_layer_squezze_ct
            };
        }
    }
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
    
    if (shape_input.size() <= 0) {
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
    if (param->direction == 0 || param->direction == 1) {
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
        lstm_param_->forgetbias = true;
        lstm_param_->cellclipthreshold = 500000;
        
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
            
            rawbuffer_fp32_WRBisa_ = shared_ptr<RawBuffer>(new RawBuffer(data_count*sizeof(float), {8, hidden_size}));
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
    } else if (param->direction == 2) {
        coreml_layer_param_ = std::shared_ptr<CoreML__Specification__BiDirectionalLSTMLayerParams>(new CoreML__Specification__BiDirectionalLSTMLayerParams);
        coreml_layer_->bidirectionallstm = (CoreML__Specification__BiDirectionalLSTMLayerParams *)coreml_layer_param_.get();
        core_ml__specification__bi_directional_lstmlayer_params__init(coreml_layer_->bidirectionallstm);
        
        coreml_layer_->bidirectionallstm->inputvectorsize = input_size;
        coreml_layer_->bidirectionallstm->outputvectorsize = hidden_size;
        
        //activation
        {
            lstm_activations_ptrs_ = std::shared_ptr<CoreML__Specification__ActivationParams*>(new CoreML__Specification__ActivationParams* [6],
                                                                      [](CoreML__Specification__ActivationParams** p) { delete [] p; });
            auto lstm_activations_ptrs_temp = lstm_activations_ptrs_.get();
            lstm_activations_ =  std::shared_ptr<CoreML__Specification__ActivationParams>(new CoreML__Specification__ActivationParams [6],
                                                                                          [](CoreML__Specification__ActivationParams* p) { delete[] p; });
            auto lstm_activations_temp = lstm_activations_.get();
            // forword
            coreml_layer_->bidirectionallstm->n_activationsforwardlstm = 3;
            {
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
                coreml_layer_->bidirectionallstm->activationsforwardlstm = lstm_activations_ptrs_.get();
            }
            // backword
            coreml_layer_->bidirectionallstm->n_activationsbackwardlstm = 3;
            {
                { //sigmoid
                    lstm_activations_ptrs_temp[3] = lstm_activations_temp + 3;
                    auto activation_sigmoid = lstm_activations_ptrs_temp[3];
                    core_ml__specification__activation_params__init(activation_sigmoid);
                    activation_sigmoid->nonlinearity_type_case = CORE_ML__SPECIFICATION__ACTIVATION_PARAMS__NONLINEARITY_TYPE_SIGMOID;
                    lstm_activation_backword_sigmoid_ = std::shared_ptr<CoreML__Specification__ActivationSigmoid>(new CoreML__Specification__ActivationSigmoid);
                    activation_sigmoid->sigmoid = lstm_activation_backword_sigmoid_.get();
                    core_ml__specification__activation_sigmoid__init(activation_sigmoid->sigmoid);
                }
                { //tanh
                    lstm_activations_ptrs_temp[4] = lstm_activations_temp + 4;
                    auto activation_tanh = lstm_activations_ptrs_temp[4];
                    core_ml__specification__activation_params__init(activation_tanh);
                    activation_tanh->nonlinearity_type_case = CORE_ML__SPECIFICATION__ACTIVATION_PARAMS__NONLINEARITY_TYPE_TANH;
                    lstm_activation_backword_tanh0_ = std::shared_ptr<CoreML__Specification__ActivationTanh>(new CoreML__Specification__ActivationTanh);
                    activation_tanh->tanh = lstm_activation_backword_tanh0_.get();
                    core_ml__specification__activation_tanh__init(activation_tanh->tanh);
                }
                { //tanh
                    lstm_activations_ptrs_temp[5] = lstm_activations_temp + 5;
                    auto activation_tanh = lstm_activations_ptrs_temp[5];
                    core_ml__specification__activation_params__init(activation_tanh);
                    activation_tanh->nonlinearity_type_case = CORE_ML__SPECIFICATION__ACTIVATION_PARAMS__NONLINEARITY_TYPE_TANH;
                    lstm_activation_backword_tanh1_ = std::shared_ptr<CoreML__Specification__ActivationTanh>(new CoreML__Specification__ActivationTanh);
                    activation_tanh->tanh = lstm_activation_backword_tanh1_.get();
                    core_ml__specification__activation_tanh__init(activation_tanh->tanh);
                }
                coreml_layer_->bidirectionallstm->activationsbackwardlstm = (CoreML__Specification__ActivationParams **)(&(lstm_activations_ptrs_temp[3]));
            }
        }
        
        //param
        lstm_param_ = std::shared_ptr<CoreML__Specification__LSTMParams>(new CoreML__Specification__LSTMParams);
        core_ml__specification__lstmparams__init(lstm_param_.get());
        coreml_layer_->bidirectionallstm->params = lstm_param_.get();
        lstm_param_->sequenceoutput = shape_input.front() > 1;
        lstm_param_->sequenceoutput = true;
        lstm_param_->hasbiasvectors = true;
        lstm_param_->forgetbias = true;
        lstm_param_->cellclipthreshold = 500000;
        
        //weight
        lstm_weight_param_ptrs_ = std::shared_ptr<CoreML__Specification__LSTMWeightParams*>(new CoreML__Specification__LSTMWeightParams*[2],
                                                                  [](CoreML__Specification__LSTMWeightParams** p) { delete [] p; });
        auto lstm_weight_param_ptrs_temp = lstm_weight_param_ptrs_.get();
        lstm_weight_param_ = std::shared_ptr<CoreML__Specification__LSTMWeightParams>(new CoreML__Specification__LSTMWeightParams[2],
                                                                                       [](CoreML__Specification__LSTMWeightParams* p) { delete [] p; });
        auto lstm_weight_param_temp = lstm_weight_param_.get();
        lstm_weight_param_ptrs_temp[0] = lstm_weight_param_temp;
        lstm_weight_param_ptrs_temp[1] = lstm_weight_param_temp + 1;
        core_ml__specification__lstmweight_params__init(lstm_weight_param_ptrs_temp[0]);
        core_ml__specification__lstmweight_params__init(lstm_weight_param_ptrs_temp[1]);
        coreml_layer_->bidirectionallstm->n_weightparams = 2;
        coreml_layer_->bidirectionallstm->weightparams = lstm_weight_param_ptrs_temp;
        
        auto lstm_weight_param_forword = lstm_weight_param_ptrs_temp[0];
        auto lstm_weight_param_backword = lstm_weight_param_ptrs_temp[1];
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
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(data_page_size, (void *)(data_ptr + 4*data_page_size*byte_size), data_type, data_dims,
                                                 lstm_weight_backword_W_i_, rawbuffer_fp32_backword_W_i_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(data_page_size, (void *)(data_ptr + 5*data_page_size*byte_size), data_type, data_dims,
                                                 lstm_weight_backword_W_o_, rawbuffer_fp32_backword_W_o_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(data_page_size, (void *)(data_ptr + 6*data_page_size*byte_size), data_type, data_dims,
                                                 lstm_weight_backword_W_f_, rawbuffer_fp32_backword_W_f_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(data_page_size, (void *)(data_ptr + 7*data_page_size*byte_size), data_type, data_dims,
                                                 lstm_weight_backword_W_c_, rawbuffer_fp32_backword_W_c_), TNN_OK);
            
            lstm_weight_param_forword->inputgateweightmatrix = lstm_weight_W_i_.get();
            lstm_weight_param_forword->outputgateweightmatrix = lstm_weight_W_o_.get();
            lstm_weight_param_forword->forgetgateweightmatrix = lstm_weight_W_f_.get();
            lstm_weight_param_forword->blockinputweightmatrix = lstm_weight_W_c_.get();
            lstm_weight_param_backword->inputgateweightmatrix = lstm_weight_backword_W_i_.get();
            lstm_weight_param_backword->outputgateweightmatrix = lstm_weight_backword_W_o_.get();
            lstm_weight_param_backword->forgetgateweightmatrix = lstm_weight_backword_W_f_.get();
            lstm_weight_param_backword->blockinputweightmatrix = lstm_weight_backword_W_c_.get();
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
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(data_page_size, (void *)(data_ptr + 4*data_page_size*byte_size), data_type, data_dims,
                                                 lstm_weight_backword_R_i_, rawbuffer_fp32_backword_R_i_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(data_page_size, (void *)(data_ptr + 5*data_page_size*byte_size), data_type, data_dims,
                                                 lstm_weight_backword_R_o_, rawbuffer_fp32_backword_R_o_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(data_page_size, (void *)(data_ptr + 6*data_page_size*byte_size), data_type, data_dims,
                                                 lstm_weight_backword_R_f_, rawbuffer_fp32_backword_R_f_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(data_page_size, (void *)(data_ptr + 7*data_page_size*byte_size), data_type, data_dims,
                                                 lstm_weight_backword_R_c_, rawbuffer_fp32_backword_R_c_), TNN_OK);
            
            lstm_weight_param_forword->inputgaterecursionmatrix = lstm_weight_R_i_.get();
            lstm_weight_param_forword->outputgaterecursionmatrix = lstm_weight_R_o_.get();
            lstm_weight_param_forword->forgetgaterecursionmatrix = lstm_weight_R_f_.get();
            lstm_weight_param_forword->blockinputrecursionmatrix = lstm_weight_R_c_.get();
            lstm_weight_param_backword->inputgaterecursionmatrix = lstm_weight_backword_R_i_.get();
            lstm_weight_param_backword->outputgaterecursionmatrix = lstm_weight_backword_R_o_.get();
            lstm_weight_param_backword->forgetgaterecursionmatrix = lstm_weight_backword_R_f_.get();
            lstm_weight_param_backword->blockinputrecursionmatrix = lstm_weight_backword_R_c_.get();
        }
        
        //B[iofc] Concatenation of [Wb[iofc], Rb[iofc]], [num_directions, 8*hidden_size]
        {
            const DimsVector data_dims = {hidden_size};
            auto data_count = buffer_B->GetDataCount();
            auto data_type = buffer_B->GetDataType();
            const int byte_size = DataTypeUtils::GetBytesSize(data_type);
            char *data_ptr = buffer_B->force_to<char *>();
            
            rawbuffer_fp32_WRBisa_ = shared_ptr<RawBuffer>(new RawBuffer(8*hidden_size*sizeof(float), {8, hidden_size}));
            float *data_fp32_ptr = rawbuffer_fp32_WRBisa_->force_to<float *>();
            if (data_type == DATA_TYPE_HALF) {
                RETURN_ON_NEQ(ConvertFromHalfToFloat((void *)data_ptr, (float *)data_fp32_ptr, 8*hidden_size),TNN_OK);
            } else {
                memcpy(data_fp32_ptr, data_ptr, 8*hidden_size*byte_size);
            }
            
            rawbuffer_fp32_backword_WRBisa_ = shared_ptr<RawBuffer>(new RawBuffer(8*hidden_size*sizeof(float), {8, hidden_size}));
            float *data_fp32_backword_ptr = rawbuffer_fp32_backword_WRBisa_->force_to<float *>();
            if (data_type == DATA_TYPE_HALF) {
                RETURN_ON_NEQ(ConvertFromHalfToFloat((void *)(data_ptr + 8*hidden_size*sizeof(float)), (float *)data_fp32_backword_ptr, 8*hidden_size),TNN_OK);
            } else {
                memcpy(data_fp32_backword_ptr, (data_ptr + 8*hidden_size*sizeof(float)), 8*hidden_size*byte_size);
            }
            
            for (int index = 0; index<4*hidden_size; index++) {
                data_fp32_ptr[index] += data_fp32_ptr[index + 4*hidden_size];
                data_fp32_backword_ptr[index] += data_fp32_backword_ptr[index + 4*hidden_size];
            }
            
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(hidden_size, (void *)data_fp32_ptr, DATA_TYPE_FLOAT, data_dims,
                                                 lstm_weight_WRBisa_i_, rawbuffer_fp32_WRBisa_i_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(hidden_size, (void *)(data_fp32_ptr + hidden_size), DATA_TYPE_FLOAT, data_dims,
                                                 lstm_weight_WRBisa_o_, rawbuffer_fp32_WRBisa_o_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(hidden_size, (void *)(data_fp32_ptr + 2*hidden_size), DATA_TYPE_FLOAT, data_dims,
                                                 lstm_weight_WRBisa_f_, rawbuffer_fp32_WRBisa_f_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(hidden_size, (void *)(data_fp32_ptr + 3*hidden_size), DATA_TYPE_FLOAT, data_dims,
                                                 lstm_weight_WRBisa_c_, rawbuffer_fp32_WRBisa_c_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(hidden_size, (void *)data_fp32_backword_ptr, DATA_TYPE_FLOAT, data_dims,
                                                 lstm_weight_backword_WRBisa_i_, rawbuffer_fp32_backword_WRBisa_i_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(hidden_size, (void *)(data_fp32_backword_ptr + hidden_size), DATA_TYPE_FLOAT, data_dims,
                                                 lstm_weight_backword_WRBisa_o_, rawbuffer_fp32_backword_WRBisa_o_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(hidden_size, (void *)(data_fp32_backword_ptr + 2*hidden_size), DATA_TYPE_FLOAT, data_dims,
                                                 lstm_weight_backword_WRBisa_f_, rawbuffer_fp32_backword_WRBisa_f_), TNN_OK);
            RETURN_ON_NEQ(RawBuffer2CoreMLWeight(hidden_size, (void *)(data_fp32_backword_ptr + 3*hidden_size), DATA_TYPE_FLOAT, data_dims,
                                                 lstm_weight_backword_WRBisa_c_, rawbuffer_fp32_backword_WRBisa_c_), TNN_OK);
            
            lstm_weight_param_forword->inputgatebiasvector = lstm_weight_WRBisa_i_.get();
            lstm_weight_param_forword->outputgatebiasvector = lstm_weight_WRBisa_o_.get();
            lstm_weight_param_forword->forgetgatebiasvector = lstm_weight_WRBisa_f_.get();
            lstm_weight_param_forword->blockinputbiasvector = lstm_weight_WRBisa_c_.get();
            lstm_weight_param_backword->inputgatebiasvector = lstm_weight_backword_WRBisa_i_.get();
            lstm_weight_param_backword->outputgatebiasvector = lstm_weight_backword_WRBisa_o_.get();
            lstm_weight_param_backword->forgetgatebiasvector = lstm_weight_backword_WRBisa_f_.get();
            lstm_weight_param_backword->blockinputbiasvector = lstm_weight_backword_WRBisa_c_.get();
        }
    }
    
    return TNN_OK;
}

Status CoreMLLSTMLayer::BuildSplitLayer(std::string input, std::vector<std::string> outputs,
                       std::shared_ptr<CoreMLBaseLayer> &coreml_layer, std::shared_ptr<LayerInfo> &layer_info) {
    auto splitv_layer = CreateCoreMLBaseLayer(LAYER_SPLITV);
    if (!splitv_layer) {
        LOGE("Error: CreateCoreMLBaseLayer failed\n");
        return Status(TNNERR_PARAM_ERR, "Error: CreateCoreMLBaseLayer failed");
    }
    splitv_layer->SetNetResource(net_resource_);
    
    layer_info = std::shared_ptr<LayerInfo>(new LayerInfo);
    auto param_splitv = std::shared_ptr<SplitVLayerParam>(new SplitVLayerParam);
    {
        layer_info->type = LAYER_SPLITV;
        layer_info->name = input + "-splitv";
        layer_info->inputs = {input};
        layer_info->outputs = outputs;
        layer_info->param = param_splitv;
        {
            param_splitv->type = layer_info->type;
            param_splitv->name = layer_info->name;
            param_splitv->axis = 0;
            param_splitv->slices = {1, 1};
        }
    }
    RETURN_ON_NEQ(splitv_layer->Init(layer_info.get(), nullptr),  TNN_OK);
    
    coreml_layer = splitv_layer;
    return TNN_OK;
}

Status CoreMLLSTMLayer::BuildReshapeLayer(std::string input, std::string output,
                                          std::shared_ptr<CoreMLBaseLayer> &coreml_layer, std::shared_ptr<LayerInfo> &layer_info) {
    auto reshape_layer = CreateCoreMLBaseLayer(LAYER_RESHAPE);
    if (!reshape_layer) {
        LOGE("Error: CreateCoreMLBaseLayer failed for LAYER_RESHAPE\n");
        return Status(TNNERR_PARAM_ERR, "Error: CreateCoreMLBaseLayer failed");
    }
    reshape_layer->SetNetResource(net_resource_);
    
    layer_info = std::shared_ptr<LayerInfo>(new LayerInfo);
    auto param = std::shared_ptr<ReshapeLayerParam>(new ReshapeLayerParam);
    {
        layer_info->type = LAYER_RESHAPE;
        layer_info->name = output;
        layer_info->inputs = {input};
        layer_info->outputs = {output};
        layer_info->param = param;
        {
            param->type = layer_info->type;
            param->name = layer_info->name;
            param->num_axes = 5;
            param->shape = {0, -1, 0, 1, 1};
        }
    }
    RETURN_ON_NEQ(reshape_layer->Init(layer_info.get(), nullptr),  TNN_OK);
    
    coreml_layer = reshape_layer;
    return TNN_OK;
}

Status CoreMLLSTMLayer::BuildConcatLayer(std::vector<std::string> inputs, std::string output,
                        std::shared_ptr<CoreMLBaseLayer> &coreml_layer, std::shared_ptr<LayerInfo> &layer_info) {
    auto concat_layer = CreateCoreMLBaseLayer(LAYER_CONCAT);
    if (!concat_layer) {
        LOGE("Error: CreateCoreMLBaseLayer failed\n");
        return Status(TNNERR_PARAM_ERR, "Error: CreateCoreMLBaseLayer failed");
    }
    concat_layer->SetNetResource(net_resource_);
    
    layer_info = std::shared_ptr<LayerInfo>(new LayerInfo);
    auto param = std::shared_ptr<ConcatLayerParam>(new ConcatLayerParam);
    {
        layer_info->type = LAYER_CONCAT;
        layer_info->name = output + "-concat";
        layer_info->inputs = inputs;
        layer_info->outputs = {output};
        layer_info->param = param;
        {
            param->type = layer_info->type;
            param->name = layer_info->name;
            param->axis = 0;
        }
    }
    RETURN_ON_NEQ(concat_layer->Init(layer_info.get(), nullptr),  TNN_OK);
    
    coreml_layer = concat_layer;
    return TNN_OK;
}

Status CoreMLLSTMLayer::BuildSqueezeLayer(std::string input, std::string output,
                         std::shared_ptr<CoreMLBaseLayer> &coreml_layer, std::shared_ptr<LayerInfo> &layer_info) {
    auto squeeze_layer = CreateCoreMLBaseLayer(LAYER_SQUEEZE);
    if (!squeeze_layer) {
        LOGE("Error: CreateCoreMLBaseLayer failed\n");
        return Status(TNNERR_PARAM_ERR, "Error: CreateCoreMLBaseLayer failed");
    }
    squeeze_layer->SetNetResource(net_resource_);
    
    layer_info = std::shared_ptr<LayerInfo>(new LayerInfo);
    auto param_squeeze_output = std::shared_ptr<SqueezeLayerParam>(new SqueezeLayerParam);
    {
        layer_info->type = LAYER_SQUEEZE;
        layer_info->name = input;
        layer_info->inputs = {input};
        layer_info->outputs = {output};
        layer_info->param = param_squeeze_output;
        {
            param_squeeze_output->type = layer_info->type;
            param_squeeze_output->name = layer_info->name;
            param_squeeze_output->axes = {3, 4};
        }
    }
    RETURN_ON_NEQ(squeeze_layer->Init(layer_info.get(), nullptr),  TNN_OK);
    
    coreml_layer = squeeze_layer;
    return TNN_OK;
}

Status CoreMLLSTMLayer::BuildConstantWeightsLayer() {
    //weight in constantmap
    //blobs of W、R、B are used as layer resource, no need to generate constant layer
    //blobs of h0 and c0 are used as layer const inputs, sowe must generate constant layer
    if (layer_info_->inputs.size() >= 6) {
        std::vector<std::string> init_inputs = {layer_info_->inputs[4], layer_info_->inputs[5]};
        return CoreMLBaseLayer::BuildConstantWeightsLayer(init_inputs);
    }
    return TNN_OK;
}

std::vector<std::string> CoreMLLSTMLayer::BuildLayerInputs() {
    auto param = dynamic_cast<LSTMONNXLayerParam *>(layer_info_->param.get());
    std::vector<std::string> inputs;
    
    if (param && (param->direction == 0 || param->direction == 1)) {
        inputs = {
            layer_info_reshape_input_->outputs[0],
            layer_info_reshape_h0_->outputs[0],
            layer_info_reshape_c0_->outputs[0]
        };
    } else if (param && param->direction == 2) {
        inputs = {
            layer_info_reshape_input_->outputs[0],
            layer_info_reshape_h0_->outputs[0],
            layer_info_reshape_c0_->outputs[0],
            layer_info_reshape_backword_h0_->outputs[0],
            layer_info_reshape_backword_c0_->outputs[0]
        };
    }
    return inputs;
}

std::vector<std::string> CoreMLLSTMLayer::BuildLayerOutputs() {
    auto param = dynamic_cast<LSTMONNXLayerParam *>(layer_info_->param.get());
    std::vector<std::string> outputs;
    
    if (param && (param->direction == 0 || param->direction == 1)) {
        if (layer_info_->outputs.size() >= 3) {
            outputs = {
                layer_info_squeeze_output_->inputs[0],
                layer_info_squeeze_ht_->inputs[0],
                layer_info_squeeze_ct_->inputs[0]
            };
        } else {
            outputs = {
                layer_info_squeeze_output_->inputs[0],
                param->name + "-ht-forword-ignore",
                param->name + "-ct-forword-ignore",
            };
        }
    } else if (param && param->direction == 2) {
        if (layer_info_->outputs.size() >= 3) {
            outputs = {
                layer_info_squeeze_output_->inputs[0],
                layer_info_concat_ht_->inputs[0],
                layer_info_concat_ct_->inputs[0],
                layer_info_concat_ht_->inputs[1],
                layer_info_concat_ct_->inputs[1],
            };
        } else {
            outputs = {
                layer_info_squeeze_output_->inputs[0],
                param->name + "-ht-forword-ignore",
                param->name + "-ct-forword-ignore",
                param->name + "-ht-reverse-ignore",
                param->name + "-ct-reverse-ignore",
            };
        }

    }
    return outputs;
}

REGISTER_COREML_LAYER(LSTM, LAYER_LSTMONNX);

}  // namespace TNN_NS

#else
/****************************************************************************/
// Code below is tested for using two uniLSTM to simulate a biLSTM
#include "coreml_base_layer.h"
#include "tnn/utils/data_type_utils.h"
#include "coreml_const_layer.h"

namespace TNN_NS {

  class CoreMLLSTMSingleLayer : public CoreMLBaseLayer {
    public:
        CoreMLLSTMSingleLayer(LayerType layer_type) : CoreMLBaseLayer(layer_type){};
        CoreMLLSTMSingleLayer(std::vector<std::string> inputs, std::vector<std::string> outputs,
                              std::string blob_name_W, std::string blob_name_R, std::string blob_name_B, std::string blob_name_h0, std::string blob_name_c0,
                              int sequence_length, int batch_size,  int input_size, int output_size, int direction, bool is_splited_form_bidirection);
        virtual ~CoreMLLSTMSingleLayer(){};

    protected:
        virtual Status BuildLayerType();
        virtual Status BuildLayerParam();
        virtual Status BuildConstantWeightsLayer();
        virtual std::vector<std::string> BuildLayerInputs();
        virtual std::vector<std::string> BuildLayerOutputs();

  protected:
      std::string blob_name_W_;
      std::string blob_name_R_;
      std::string blob_name_B_;
      std::string blob_name_h0_;
      std::string blob_name_c0_;
      std::vector<std::string> inputs_direct_;
      std::vector<std::string> outputs_direct_;
      int sequence_length_ = 0;
      int batch_size_ = 0;
      int input_size_ = 0;
      int output_size_ = 0;
      int direction_ = -1;
      bool is_splited_form_bidirection_ = false;

      std::shared_ptr<CoreML__Specification__LSTMParams> lstm_param_;
      std::shared_ptr<CoreML__Specification__LSTMWeightParams*> lstm_weight_param_ptrs_;
      std::shared_ptr<CoreML__Specification__LSTMWeightParams> lstm_weight_param_;
      std::shared_ptr<CoreML__Specification__ActivationParams*> lstm_activations_ptrs_;
      std::shared_ptr<CoreML__Specification__ActivationParams> lstm_activations_;
      std::shared_ptr<CoreML__Specification__ActivationSigmoid> lstm_activation_sigmoid_;
      std::shared_ptr<CoreML__Specification__ActivationTanh> lstm_activation_tanh0_;
      std::shared_ptr<CoreML__Specification__ActivationTanh> lstm_activation_tanh1_;
      std::shared_ptr<CoreML__Specification__WeightParams> lstm_weight_W_i_;
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
  };

CoreMLLSTMSingleLayer::CoreMLLSTMSingleLayer(std::vector<std::string> inputs, std::vector<std::string> outputs,
                                             std::string blob_name_W, std::string blob_name_R, std::string blob_name_B, std::string blob_name_h0, std::string blob_name_c0,
                                             int sequence_length, int batch_size,  int input_size, int output_size, int direction,
                                             bool is_splited_form_bidirection) : CoreMLBaseLayer(LAYER_NOT_SUPPORT) {
    inputs_direct_ = inputs;
    outputs_direct_ = outputs;

    blob_name_W_ = blob_name_W;
    blob_name_R_ = blob_name_R;
    blob_name_B_ = blob_name_B;
    blob_name_h0_ = blob_name_h0;
    blob_name_c0_ = blob_name_c0;

    sequence_length_ = sequence_length;
    batch_size_ = batch_size;
    input_size_ = input_size;
    output_size_ = output_size;
    direction_ = direction;
    is_splited_form_bidirection_ = is_splited_form_bidirection;
}

Status CoreMLLSTMSingleLayer::BuildLayerType() {
    //layer type
    if (direction_ == 0) {
        coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_UNI_DIRECTIONAL_LSTM;
    } else if (direction_ == 1) {
        coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_UNI_DIRECTIONAL_LSTM;
    } else  {
        return Status(TNNERR_COMMON_ERROR, "CoreMLLSTMLayer dont support reverse LSTM");
    }

    return TNN_OK;
}

Status CoreMLLSTMSingleLayer::BuildLayerParam() {
    if (!net_resource_) {
        return Status(TNNERR_LAYER_ERR, "CoreMLLSTMLayer has invalid net resource");
    }
    
    auto blob_name_W = blob_name_W_;
    auto blob_name_R = blob_name_R_;
    auto blob_name_B = blob_name_B_;
    auto blob_name_h0 = blob_name_h0_;
    auto blob_name_c0 = blob_name_c0_;
    
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

    const int input_size = input_size_;
    const int hidden_size = output_size_;

    //set lstm param
    if (direction_ == 0 || direction_ == 1) {
        coreml_layer_param_ = std::shared_ptr<CoreML__Specification__UniDirectionalLSTMLayerParams>(new CoreML__Specification__UniDirectionalLSTMLayerParams);
        coreml_layer_->unidirectionallstm = (CoreML__Specification__UniDirectionalLSTMLayerParams *)coreml_layer_param_.get();
        core_ml__specification__uni_directional_lstmlayer_params__init(coreml_layer_->unidirectionallstm);

        coreml_layer_->unidirectionallstm->inputvectorsize = input_size;
        coreml_layer_->unidirectionallstm->outputvectorsize = hidden_size;
        coreml_layer_->unidirectionallstm->reverseinput = direction_ == 1;

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
        lstm_param_->sequenceoutput = sequence_length_ > 1;
        lstm_param_->hasbiasvectors = true;
        lstm_param_->forgetbias = true;
        lstm_param_->cellclipthreshold = 500000;

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
            if (direction_ == 1 &&  is_splited_form_bidirection_) {
                data_ptr += 4*data_page_size*byte_size;
            }

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
            if (direction_ == 1 &&  is_splited_form_bidirection_) {
                data_ptr += 4*data_page_size*byte_size;
            }

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
            if (direction_ == 1 &&  is_splited_form_bidirection_) {
                data_ptr += 8*hidden_size*byte_size;
            }

            rawbuffer_fp32_WRBisa_ = shared_ptr<RawBuffer>(new RawBuffer(8*hidden_size*sizeof(float), {8, hidden_size}));
            float *data_fp32_ptr = rawbuffer_fp32_WRBisa_->force_to<float *>();
            if (data_type == DATA_TYPE_HALF) {
                RETURN_ON_NEQ(ConvertFromHalfToFloat((void *)data_ptr, (float *)data_fp32_ptr, 8*hidden_size),TNN_OK);
            } else {
                memcpy(data_fp32_ptr, data_ptr, 8*hidden_size*byte_size);
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

Status CoreMLLSTMSingleLayer::BuildConstantWeightsLayer() {
    //weight in constantmap
    //blobs of W、R、B are used as layer resource, no need to generate constant layer
    //blobs of h0 and c0 are used as layer const inputs, sowe must generate constant layer
    //do BuildConstantWeightsLayer in CoreMLLSTMLayer
    return TNN_OK;
}

std::vector<std::string> CoreMLLSTMSingleLayer::BuildLayerInputs() {
    return inputs_direct_;
}

std::vector<std::string> CoreMLLSTMSingleLayer::BuildLayerOutputs() {
    return outputs_direct_;
}

DECLARE_COREML_LAYER_WITH_FUNC_DATA(LSTM, LAYER_LSTMONNX,
                                    virtual std::vector<CoreML__Specification__NeuralNetworkLayer*> GetCoreMLLayerPtrs();
                                    Status BuildSplitLayer(std::string input, std::vector<std::string> outputs,
                                                               std::shared_ptr<CoreMLBaseLayer> &coreml_layer, std::shared_ptr<LayerInfo> &layer_info);
                                    Status BuildReshapeLayer(std::string input, std::string output,
                                                               std::shared_ptr<CoreMLBaseLayer> &coreml_layer, std::shared_ptr<LayerInfo> &layer_info);
                                    Status BuildConcatLayer(std::vector<std::string> inputs, std::string output, int axis,
                                                               std::shared_ptr<CoreMLBaseLayer> &coreml_layer, std::shared_ptr<LayerInfo> &layer_info);
                                    Status BuildSqueezeLayer(std::string input, std::string output,
                                                               std::shared_ptr<CoreMLBaseLayer> &coreml_layer, std::shared_ptr<LayerInfo> &layer_info);,
                                    std::shared_ptr<CoreMLLSTMSingleLayer> coreml_layer_lstm_forword_;
                                    std::shared_ptr<CoreMLLSTMSingleLayer> coreml_layer_lstm_reverse_;
                                    std::shared_ptr<LayerInfo> layer_info_reshape_input_;
                                    std::shared_ptr<LayerInfo> layer_info_split_h0_;
                                    std::shared_ptr<LayerInfo> layer_info_reshape_h0_;
                                    std::shared_ptr<LayerInfo> layer_info_split_c0_;
                                    std::shared_ptr<LayerInfo> layer_info_reshape_c0_;
                                    std::shared_ptr<LayerInfo> layer_info_reshape_backword_h0_;
                                    std::shared_ptr<LayerInfo> layer_info_reshape_backword_c0_;
                                    std::shared_ptr<LayerInfo> layer_info_concat_output_;
                                    std::shared_ptr<LayerInfo> layer_info_concat_ht_;
                                    std::shared_ptr<LayerInfo> layer_info_concat_ct_;
                                    std::shared_ptr<LayerInfo> layer_info_squeeze_output_;
                                    std::shared_ptr<LayerInfo> layer_info_squeeze_ht_;
                                    std::shared_ptr<LayerInfo> layer_info_squeeze_ct_;);

std::vector<CoreML__Specification__NeuralNetworkLayer*> CoreMLLSTMLayer::GetCoreMLLayerPtrs() {
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

    if (coreml_layer_lstm_forword_) {
        auto after_layer_ptr = coreml_layer_lstm_forword_->GetCoreMLLayerPtrs();
        layer_ptrs.insert(layer_ptrs.end(), after_layer_ptr.begin(), after_layer_ptr.end());
    }

    if (coreml_layer_lstm_reverse_) {
        auto after_layer_ptr = coreml_layer_lstm_reverse_->GetCoreMLLayerPtrs();
        layer_ptrs.insert(layer_ptrs.end(), after_layer_ptr.begin(), after_layer_ptr.end());
    }

    for (auto iter : coreml_layers_after_) {
        auto after_layer_ptr = iter->GetCoreMLLayerPtrs();
        layer_ptrs.insert(layer_ptrs.end(), after_layer_ptr.begin(), after_layer_ptr.end());
    }
    return layer_ptrs;
}

Status CoreMLLSTMLayer::BuildLayerType() {
    //nullfy coreml_layer_, ortherwise GetCoreMLLayerPtrs will get wrong result
    coreml_layer_ = nullptr;

    return TNN_OK;
}

/*
 *NOTE1:
 *CoreML now only support LSTM at CPU device at 2022.07.19.
 *Both optiones MLComputeUnitsCPUOnly and MLComputeUnitsAll have the same benchmark time for model crnn_lite_lstm
 * And both are  much more slower than TNN arm.
 *
 *NOTE2:
 *CoreML bidirection LSTM always produce wrong result for the second slice. so use two uniLSTM to do the work
*/
Status CoreMLLSTMLayer::BuildLayerParam() {
    auto param = dynamic_cast<LSTMONNXLayerParam *>(layer_info_->param.get());
    CHECK_PARAM_NULL(param);

    if (layer_info_->inputs.size() < 6 || !net_resource_) {
        return Status(TNNERR_LAYER_ERR, "CoreMLLSTMLayer has invalid inputs size");
    }

    if (param->direction == 0 || param->direction == 1) {
        std::shared_ptr<CoreMLBaseLayer> coreml_layer_reshape_input, coreml_layer_reshape_h0, coreml_layer_reshape_c0;
        RETURN_ON_NEQ(BuildReshapeLayer(layer_info_->inputs[0], layer_info_->name + "-reshape-input",
                                        coreml_layer_reshape_input, layer_info_reshape_input_), TNN_OK);
        RETURN_ON_NEQ(BuildReshapeLayer(layer_info_->inputs[4], layer_info_->name + "-reshape-h0",
                                        coreml_layer_reshape_h0, layer_info_reshape_h0_), TNN_OK);
        RETURN_ON_NEQ(BuildReshapeLayer(layer_info_->inputs[5], layer_info_->name + "-reshape-c0",
                                        coreml_layer_reshape_c0, layer_info_reshape_c0_), TNN_OK);
        coreml_layers_before_ = {coreml_layer_reshape_input, coreml_layer_reshape_h0, coreml_layer_reshape_c0};

        std::shared_ptr<CoreMLBaseLayer> coreml_layer_squezze_output, coreml_layer_squezze_ht, coreml_layer_squezze_ct;
        RETURN_ON_NEQ(BuildSqueezeLayer(layer_info_->name + "-squeeze-output", layer_info_->outputs[0],
                                        coreml_layer_squezze_output, layer_info_squeeze_output_), TNN_OK);

        coreml_layers_after_ = {coreml_layer_squezze_output};
        if (layer_info_->outputs.size() >= 3) {
            RETURN_ON_NEQ(BuildSqueezeLayer(layer_info_->name + "-squeeze-ht", layer_info_->outputs[1],
                                            coreml_layer_squezze_ht, layer_info_squeeze_ht_), TNN_OK);
            RETURN_ON_NEQ(BuildSqueezeLayer(layer_info_->name + "-squeeze-ct", layer_info_->outputs[2],
                                            coreml_layer_squezze_ct, layer_info_squeeze_ct_), TNN_OK);
            coreml_layers_after_ = {coreml_layer_squezze_output, coreml_layer_squezze_ht, coreml_layer_squezze_ct};
        }

    } else if (param->direction == 2) {
        std::shared_ptr<CoreMLBaseLayer> coreml_layer_reshape_input, coreml_layer_split_h0, coreml_layer_split_c0, coreml_layer_reshape_h0,coreml_layer_reshape_backword_h0,coreml_layer_reshape_c0,coreml_layer_reshape_backword_c0;

        RETURN_ON_NEQ(BuildReshapeLayer(layer_info_->inputs[0], layer_info_->name + "-reshape-input",
                                          coreml_layer_reshape_input, layer_info_reshape_input_), TNN_OK);
        RETURN_ON_NEQ(BuildSplitLayer(layer_info_->inputs[4], {layer_info_->name + "-split-h0", layer_info_->name + "-split-backword-h0"}, coreml_layer_split_h0, layer_info_split_h0_), TNN_OK);
        RETURN_ON_NEQ(BuildSplitLayer(layer_info_->inputs[5], {layer_info_->name + "-split-c0", layer_info_->name + "-split-backword-c0"}, coreml_layer_split_c0, layer_info_split_c0_), TNN_OK);
        RETURN_ON_NEQ(BuildReshapeLayer(layer_info_split_h0_->outputs[0], layer_info_->name + "-reshape-h0",
                                          coreml_layer_reshape_h0, layer_info_reshape_h0_), TNN_OK);
        RETURN_ON_NEQ(BuildReshapeLayer(layer_info_split_h0_->outputs[1], layer_info_->name + "-reshape-backword-h0",
                                          coreml_layer_reshape_backword_h0, layer_info_reshape_backword_h0_), TNN_OK);
        RETURN_ON_NEQ(BuildReshapeLayer(layer_info_split_c0_->outputs[0], layer_info_->name + "-reshape-c0",
                                          coreml_layer_reshape_c0, layer_info_reshape_c0_), TNN_OK);
        RETURN_ON_NEQ(BuildReshapeLayer(layer_info_split_c0_->outputs[1], layer_info_->name + "-reshape-backword-c0",
                                          coreml_layer_reshape_backword_c0, layer_info_reshape_backword_c0_), TNN_OK);
        coreml_layers_before_ = {coreml_layer_reshape_input, coreml_layer_split_h0, coreml_layer_split_c0,
            coreml_layer_reshape_h0,coreml_layer_reshape_backword_h0,
            coreml_layer_reshape_c0,coreml_layer_reshape_backword_c0};

        std::shared_ptr<CoreMLBaseLayer> coreml_layer_squezze_output, coreml_layer_concat_output, coreml_layer_concat_ht,
            coreml_layer_concat_ct, coreml_layer_squezze_ht,coreml_layer_squezze_ct;
        RETURN_ON_NEQ(BuildConcatLayer({layer_info_->name + "-concat-output-forword", layer_info_->name + "-concat-output-backword"}, layer_info_->name + "-concat-output", 2,
                                          coreml_layer_concat_output, layer_info_concat_output_), TNN_OK);
        RETURN_ON_NEQ(BuildSqueezeLayer(layer_info_->name + "-concat-output", layer_info_->outputs[0],
                                        coreml_layer_squezze_output, layer_info_squeeze_output_), TNN_OK);

        coreml_layers_after_ = {coreml_layer_concat_output, coreml_layer_squezze_output};
        if (layer_info_->outputs.size() >= 3) {
            RETURN_ON_NEQ(BuildConcatLayer({layer_info_->name + "-concat-input-ht", layer_info_->name + "-concat-input-backword-ht"}, layer_info_->name + "-concat-ht", 1,
                                              coreml_layer_concat_ht, layer_info_concat_ht_), TNN_OK);
            RETURN_ON_NEQ(BuildConcatLayer({layer_info_->name + "-concat-input-ct", layer_info_->name + "-concat-input-backword-ct"}, layer_info_->name + "-concat-ct", 1,
                                              coreml_layer_concat_ct, layer_info_concat_ct_), TNN_OK);
            RETURN_ON_NEQ(BuildSqueezeLayer(layer_info_concat_ht_->outputs[0], layer_info_->outputs[1],
                                            coreml_layer_squezze_ht, layer_info_squeeze_ht_), TNN_OK);
            RETURN_ON_NEQ(BuildSqueezeLayer(layer_info_concat_ct_->outputs[0], layer_info_->outputs[2],
                                            coreml_layer_squezze_ct, layer_info_squeeze_ct_), TNN_OK);
            coreml_layers_after_ = {coreml_layer_concat_output, coreml_layer_squezze_output,
                coreml_layer_concat_ht, coreml_layer_concat_ct,
                coreml_layer_squezze_ht, coreml_layer_squezze_ct
            };
        }

    }
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

    if (shape_input.size() <= 0) {
        LOGE("CoreMLLSTMLayer has no fixed input or output shape\n");
        return Status(TNNERR_MODEL_ERR, "CoreMLLSTMLayer has no fixed input or output shape");
    }

    const int sequence_length = shape_input[0];
    const int batch_size = shape_input[1];
    const int input_size = shape_input.back();
    const int hidden_size = param->hidden_size;

    if (param->direction == 0 || param->direction == 1) {
        std::vector<std::string> inputs_direct, outputs_direct;
        inputs_direct = {
            layer_info_reshape_input_->outputs[0],
            layer_info_reshape_h0_->outputs[0],
            layer_info_reshape_c0_->outputs[0]
        };
        if (layer_info_->outputs.size() >= 3) {
            outputs_direct = {
                layer_info_squeeze_output_->inputs[0],
                layer_info_squeeze_ht_->inputs[0],
                layer_info_squeeze_ct_->inputs[0]
            };
        } else {
            outputs_direct = {
                layer_info_squeeze_output_->inputs[0],
                param->name + "-ht-forword-ignore",
                param->name + "-ct-forword-ignore",
            };
        }

        coreml_layer_lstm_forword_ = std::shared_ptr<CoreMLLSTMSingleLayer>(new CoreMLLSTMSingleLayer(inputs_direct, outputs_direct,
                                                                                                    blob_name_W, blob_name_R, blob_name_B, blob_name_h0, blob_name_c0,
                                                                                                    sequence_length, batch_size, input_size, hidden_size, param->direction, false));
        coreml_layer_lstm_forword_->SetNetResource(net_resource_);
        RETURN_ON_NEQ(coreml_layer_lstm_forword_->Init(nullptr, nullptr),  TNN_OK);
        coreml_layer_lstm_forword_->SetLayerName(param->name);
    }  else if (param->direction == 2) {
        std::vector<std::string> inputs_direct_forword, outputs_direct_forword;
        inputs_direct_forword = {
            layer_info_reshape_input_->outputs[0],
            layer_info_reshape_h0_->outputs[0],
            layer_info_reshape_c0_->outputs[0],
        };
        if (layer_info_->outputs.size() >= 3) {
            outputs_direct_forword = {
                layer_info_concat_output_->inputs[0],
                layer_info_concat_ht_->inputs[0],
                layer_info_concat_ct_->inputs[0],
            };
        } else {
            outputs_direct_forword = {
                layer_info_concat_output_->inputs[0],
                param->name + "-ht-forword-ignore",
                param->name + "-ct-forword-ignore",
            };
        }

        coreml_layer_lstm_forword_ = std::shared_ptr<CoreMLLSTMSingleLayer>(new CoreMLLSTMSingleLayer(inputs_direct_forword, outputs_direct_forword,
                                                                                                    blob_name_W, blob_name_R, blob_name_B, blob_name_h0, blob_name_c0,
                                                                                                    sequence_length, batch_size, input_size, hidden_size, 0, true));
        std::vector<std::string> inputs_direct_reverse, outputs_direct_reverse;
        inputs_direct_reverse = {
            layer_info_reshape_input_->outputs[0],
            layer_info_reshape_backword_h0_->outputs[0],
            layer_info_reshape_backword_c0_->outputs[0],
        };
        if (layer_info_->outputs.size() >= 3) {
            outputs_direct_reverse = {
                layer_info_concat_output_->inputs[1],
                layer_info_concat_ht_->inputs[1],
                layer_info_concat_ct_->inputs[1],
            };
        } else {
            outputs_direct_reverse = {
                layer_info_concat_output_->inputs[1],
                param->name + "-ht-reverse-ignore",
                param->name + "-ct-reverse-ignore",
            };
        }

        coreml_layer_lstm_reverse_ = std::shared_ptr<CoreMLLSTMSingleLayer>(new CoreMLLSTMSingleLayer(inputs_direct_reverse, outputs_direct_reverse,
                                                                                                    blob_name_W, blob_name_R, blob_name_B, blob_name_h0, blob_name_c0,
                                                                                                    sequence_length, batch_size, input_size, hidden_size, 1, true));
        coreml_layer_lstm_forword_->SetNetResource(net_resource_);
        coreml_layer_lstm_reverse_->SetNetResource(net_resource_);
        RETURN_ON_NEQ(coreml_layer_lstm_forword_->Init(nullptr, nullptr),  TNN_OK);
        RETURN_ON_NEQ(coreml_layer_lstm_reverse_->Init(nullptr, nullptr),  TNN_OK);
        coreml_layer_lstm_forword_->SetLayerName(param->name+"-forword");
        coreml_layer_lstm_reverse_->SetLayerName(param->name+"-reverse");
    }

    return TNN_OK;
}

Status CoreMLLSTMLayer::BuildSplitLayer(std::string input, std::vector<std::string> outputs,
                       std::shared_ptr<CoreMLBaseLayer> &coreml_layer, std::shared_ptr<LayerInfo> &layer_info) {
    auto splitv_layer = CreateCoreMLBaseLayer(LAYER_SPLITV);
    if (!splitv_layer) {
        LOGE("Error: CreateCoreMLBaseLayer failed\n");
        return Status(TNNERR_PARAM_ERR, "Error: CreateCoreMLBaseLayer failed");
    }
    splitv_layer->SetNetResource(net_resource_);

    layer_info = std::shared_ptr<LayerInfo>(new LayerInfo);
    auto param_splitv = std::shared_ptr<SplitVLayerParam>(new SplitVLayerParam);
    {
        layer_info->type = LAYER_SPLITV;
        layer_info->name = input + "-splitv";
        layer_info->inputs = {input};
        layer_info->outputs = outputs;
        layer_info->param = param_splitv;
        {
            param_splitv->type = layer_info->type;
            param_splitv->name = layer_info->name;
            param_splitv->axis = 0;
            param_splitv->slices = {1, 1};
        }
    }
    RETURN_ON_NEQ(splitv_layer->Init(layer_info.get(), nullptr),  TNN_OK);

    coreml_layer = splitv_layer;
    return TNN_OK;
}

Status CoreMLLSTMLayer::BuildReshapeLayer(std::string input, std::string output,
                                          std::shared_ptr<CoreMLBaseLayer> &coreml_layer, std::shared_ptr<LayerInfo> &layer_info) {
    auto reshape_layer = CreateCoreMLBaseLayer(LAYER_RESHAPE);
    if (!reshape_layer) {
        LOGE("Error: CreateCoreMLBaseLayer failed for LAYER_RESHAPE\n");
        return Status(TNNERR_PARAM_ERR, "Error: CreateCoreMLBaseLayer failed");
    }
    reshape_layer->SetNetResource(net_resource_);

    layer_info = std::shared_ptr<LayerInfo>(new LayerInfo);
    auto param = std::shared_ptr<ReshapeLayerParam>(new ReshapeLayerParam);
    {
        layer_info->type = LAYER_RESHAPE;
        layer_info->name = output;
        layer_info->inputs = {input};
        layer_info->outputs = {output};
        layer_info->param = param;
        {
            param->type = layer_info->type;
            param->name = layer_info->name;
            param->num_axes = 5;
            param->shape = {0, -1, 0, 1, 1};
        }
    }
    RETURN_ON_NEQ(reshape_layer->Init(layer_info.get(), nullptr),  TNN_OK);

    coreml_layer = reshape_layer;
    return TNN_OK;
}

Status CoreMLLSTMLayer::BuildConcatLayer(std::vector<std::string> inputs, std::string output, int axis,
                        std::shared_ptr<CoreMLBaseLayer> &coreml_layer, std::shared_ptr<LayerInfo> &layer_info) {
    auto concat_layer = CreateCoreMLBaseLayer(LAYER_CONCAT);
    if (!concat_layer) {
        LOGE("Error: CreateCoreMLBaseLayer failed\n");
        return Status(TNNERR_PARAM_ERR, "Error: CreateCoreMLBaseLayer failed");
    }
    concat_layer->SetNetResource(net_resource_);

    layer_info = std::shared_ptr<LayerInfo>(new LayerInfo);
    auto param = std::shared_ptr<ConcatLayerParam>(new ConcatLayerParam);
    {
        layer_info->type = LAYER_CONCAT;
        layer_info->name = output + "-concat";
        layer_info->inputs = inputs;
        layer_info->outputs = {output};
        layer_info->param = param;
        {
            param->type = layer_info->type;
            param->name = layer_info->name;
            param->axis = axis;
        }
    }
    RETURN_ON_NEQ(concat_layer->Init(layer_info.get(), nullptr),  TNN_OK);

    coreml_layer = concat_layer;
    return TNN_OK;
}

Status CoreMLLSTMLayer::BuildSqueezeLayer(std::string input, std::string output,
                         std::shared_ptr<CoreMLBaseLayer> &coreml_layer, std::shared_ptr<LayerInfo> &layer_info) {
    auto squeeze_layer = CreateCoreMLBaseLayer(LAYER_SQUEEZE);
    if (!squeeze_layer) {
        LOGE("Error: CreateCoreMLBaseLayer failed\n");
        return Status(TNNERR_PARAM_ERR, "Error: CreateCoreMLBaseLayer failed");
    }
    squeeze_layer->SetNetResource(net_resource_);

    layer_info = std::shared_ptr<LayerInfo>(new LayerInfo);
    auto param_squeeze_output = std::shared_ptr<SqueezeLayerParam>(new SqueezeLayerParam);
    {
        layer_info->type = LAYER_SQUEEZE;
        layer_info->name = input;
        layer_info->inputs = {input};
        layer_info->outputs = {output};
        layer_info->param = param_squeeze_output;
        {
            param_squeeze_output->type = layer_info->type;
            param_squeeze_output->name = layer_info->name;
            param_squeeze_output->axes = {3, 4};
        }
    }
    RETURN_ON_NEQ(squeeze_layer->Init(layer_info.get(), nullptr),  TNN_OK);

    coreml_layer = squeeze_layer;
    return TNN_OK;
}

Status CoreMLLSTMLayer::BuildConstantWeightsLayer() {
    //weight in constantmap
    //blobs of W、R、B are used as layer resource, no need to generate constant layer
    //blobs of h0 and c0 are used as layer const inputs, sowe must generate constant layer
    if (layer_info_->inputs.size() >= 6) {
        //weight in constantmap
        //blobs of W、R、B are used as layer resource, no need to generate constant layer
        //blobs of h0 and c0 are used as layer const inputs, sowe must generate constant layer
        if (layer_info_->inputs.size() >= 6) {
            std::vector<std::string> init_inputs = {layer_info_->inputs[4], layer_info_->inputs[5]};
            return CoreMLBaseLayer::BuildConstantWeightsLayer(init_inputs);
        }
        return TNN_OK;
    }
    return TNN_OK;
}

std::vector<std::string> CoreMLLSTMLayer::BuildLayerInputs() {
    return std::vector<std::string>();

    auto param = dynamic_cast<LSTMONNXLayerParam *>(layer_info_->param.get());
    std::vector<std::string> inputs;

    if (param && (param->direction == 0 || param->direction == 1)) {
        inputs = {
            layer_info_reshape_input_->outputs[0],
            layer_info_reshape_h0_->outputs[0],
            layer_info_reshape_c0_->outputs[0]
        };
    } else if (param && param->direction == 2) {
        inputs = {
            layer_info_reshape_input_->outputs[0],
            layer_info_reshape_h0_->outputs[0],
            layer_info_reshape_c0_->outputs[0],
            layer_info_reshape_backword_h0_->outputs[0],
            layer_info_reshape_backword_c0_->outputs[0]
        };
    }
    return inputs;
}

std::vector<std::string> CoreMLLSTMLayer::BuildLayerOutputs() {
    return std::vector<std::string>();

    auto param = dynamic_cast<LSTMONNXLayerParam *>(layer_info_->param.get());
    std::vector<std::string> outputs;

    if (param && (param->direction == 0 || param->direction == 1)) {
        outputs = {
            layer_info_squeeze_output_->inputs[0],
            layer_info_squeeze_ht_->inputs[0],
            layer_info_squeeze_ct_->inputs[0]
        };
    } else if (param && param->direction == 2) {
        outputs = {
            layer_info_squeeze_output_->inputs[0],
            layer_info_concat_ht_->inputs[0],
            layer_info_concat_ct_->inputs[0],
            layer_info_concat_ht_->inputs[1],
            layer_info_concat_ct_->inputs[1]
        };
    }
    return outputs;
}

REGISTER_COREML_LAYER(LSTM, LAYER_LSTMONNX);

}  // namespace TNN_NS

#endif
