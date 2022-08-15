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

DECLARE_COREML_LAYER_WITH_DATA(Batchnorm, LAYER_BATCH_NORM,
                                std::shared_ptr<void> coreml_layer_type_;
                                std::shared_ptr<CoreML__Specification__WeightParams> coreml_layer_gamma_;
                                std::shared_ptr<CoreML__Specification__WeightParams> coreml_layer_beta_;
                                std::shared_ptr<CoreML__Specification__WeightParams> coreml_layer_mean_;
                                std::shared_ptr<CoreML__Specification__WeightParams> coreml_layer_variance_;
                                std::shared_ptr<void> mean_;
                                std::shared_ptr<void> variance_;
                                std::shared_ptr<float> scale_fp32_ptr_;
                                std::shared_ptr<RawBuffer> rawbuffer_shared_expand_scale_;
                                std::shared_ptr<RawBuffer> rawbuffer_shared_expand_bias_;
                                std::shared_ptr<RawBuffer> rawbuffer_scale_fp32_;
                                std::shared_ptr<RawBuffer> rawbuffer_bias_fp32_;
                                std::shared_ptr<float> bias_fp32_ptr_;);

Status CoreMLBatchnormLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_BATCHNORM;
    return TNN_OK;
}

Status CoreMLBatchnormLayer::BuildLayerParam() {
    int channels = 0;
    if (net_resource_ && layer_info_->inputs.size()>0) {
        if (net_resource_->blob_shapes_map.find(layer_info_->inputs[0]) != net_resource_->blob_shapes_map.end()) {
            auto input_shape = net_resource_->blob_shapes_map[layer_info_->inputs[0]];
            if (input_shape.size() > 1) {
                channels = input_shape[1];
            }
        }
    }
    
    
    //layer param
    auto param = dynamic_cast<BatchNormLayerParam *>(layer_info_->param.get());
    CHECK_PARAM_NULL(param);
    
    auto layer_res = dynamic_cast<BatchNormLayerResource *>(layer_resource_);
    CHECK_PARAM_NULL(layer_res);
    auto scale_count = layer_res->scale_handle.GetDataCount();
    auto scale_data_type = layer_res->scale_handle.GetDataType();
    auto bias_count = layer_res->bias_handle.GetDataCount();
    auto bias_data_type = layer_res->bias_handle.GetDataType();
    const int byte_size = DataTypeUtils::GetBytesSize(scale_data_type);
    
    bool share_channel = scale_count==1;
    channels = std::max(channels, scale_count);
    channels = std::max(channels, bias_count);
    
    RETURN_VALUE_ON_NEQ(channels >= scale_count, true, Status(TNNERR_MODEL_ERR, "Batchnorm has invalid scale param"));
    
    // TNN BatchNorm: scale*x + bias
    // CoreML BatchNorm: gamma*(x - mean) / sqrt(var^2 +epsilon) + beta  ->
    //                   gamma* (x/ sqrt(var^2 +epsilon)) - gamma* (mean/ sqrt(var^2 +epsilon)) + beta
    // --> gamma=scale, beta=bias, mean=0, var=1， epsilon=default
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__BatchnormLayerParams>(new CoreML__Specification__BatchnormLayerParams);
    coreml_layer_->batchnorm = (CoreML__Specification__BatchnormLayerParams *)coreml_layer_param_.get();
    core_ml__specification__batchnorm_layer_params__init(coreml_layer_->batchnorm);
    coreml_layer_->batchnorm->channels = channels;
    coreml_layer_->batchnorm->computemeanvar = param->is_instance_norm;
    coreml_layer_->batchnorm->instancenormalization = param->is_instance_norm;
    
    if (share_channel) {
        rawbuffer_shared_expand_scale_ = std::shared_ptr<RawBuffer>(new RawBuffer(channels*byte_size));
        rawbuffer_shared_expand_scale_->SetBufferDims({channels});
        char *scale_data_expand = rawbuffer_shared_expand_scale_->force_to<char *>();
        char *scale_data = layer_res->scale_handle.force_to<char *>();
        for (int index = 0; index < channels; index++) {
            memcpy(scale_data_expand + index*byte_size, scale_data, byte_size);
        }
        RETURN_ON_NEQ(RawBuffer2CoreMLWeight(rawbuffer_shared_expand_scale_.get(),
                                             coreml_layer_gamma_, rawbuffer_scale_fp32_), TNN_OK);
    } else {
        RETURN_ON_NEQ(RawBuffer2CoreMLWeight(&(layer_res->scale_handle),
                                             coreml_layer_gamma_, rawbuffer_scale_fp32_), TNN_OK);
    }
    coreml_layer_->batchnorm->gamma = coreml_layer_gamma_.get();
    
    if (channels > bias_count) {
        rawbuffer_shared_expand_bias_ = std::shared_ptr<RawBuffer>(new RawBuffer(channels*byte_size));
        rawbuffer_shared_expand_bias_->SetBufferDims({channels});
        char *bias_data_expand = rawbuffer_shared_expand_bias_->force_to<char *>();
        char *bias_data = layer_res->bias_handle.force_to<char *>();
        
        if (share_channel && bias_count > 0) {
            for (int index = 0; index < channels; index++) {
                memcpy(bias_data_expand + index*byte_size, bias_data, byte_size);
            }
        } else {
            memset(bias_data_expand, 0, channels*byte_size);
        }
        
        RETURN_ON_NEQ(RawBuffer2CoreMLWeight(rawbuffer_shared_expand_bias_.get(),
                                             coreml_layer_beta_, rawbuffer_bias_fp32_), TNN_OK);
    } else {
        RETURN_ON_NEQ(RawBuffer2CoreMLWeight(&(layer_res->bias_handle),
                                             coreml_layer_beta_, rawbuffer_bias_fp32_), TNN_OK);
    }
    coreml_layer_->batchnorm->beta =  coreml_layer_beta_.get();
    
    if (!param->is_instance_norm) {
        coreml_layer_mean_ = std::shared_ptr<CoreML__Specification__WeightParams>(new CoreML__Specification__WeightParams);
        coreml_layer_->batchnorm->mean = (CoreML__Specification__WeightParams*) coreml_layer_mean_.get();
        core_ml__specification__weight_params__init(coreml_layer_->batchnorm->mean);
#if TNN_COREML_FULL_PRECISION
        coreml_layer_->batchnorm->mean->n_floatvalue = channels;
        mean_ = std::shared_ptr<float>(new float[channels], [](float* p) { delete[] p; });
        coreml_layer_->batchnorm->mean->floatvalue = (float*) mean_.get();
        for(int i=0; i<channels; i++){
            coreml_layer_->batchnorm->mean->floatvalue[i] = 0;
        }
#else
        coreml_layer_->batchnorm->mean->float16value.len = layer_res->scale_handle.GetBytesSize();
        mean_ = std::shared_ptr<uint16_t>(new uint16_t[channels], [](uint16_t* p) { delete[] p; });
        auto mean = (uint16_t*) mean_.get();
        for(int i=0; i<channels; i++){
            mean[i] = 0;
        }
        coreml_layer_->batchnorm->mean->float16value.data = (uint8_t*)mean;
#endif
        
        coreml_layer_variance_ = std::shared_ptr<CoreML__Specification__WeightParams>(new CoreML__Specification__WeightParams);
        coreml_layer_->batchnorm->variance = (CoreML__Specification__WeightParams*) coreml_layer_variance_.get();
        core_ml__specification__weight_params__init(coreml_layer_->batchnorm->variance);
#if TNN_COREML_FULL_PRECISION
        coreml_layer_->batchnorm->variance->n_floatvalue = channels;
        variance_ = std::shared_ptr<float>(new float[channels], [](float* p) { delete[] p; });
        coreml_layer_->batchnorm->variance->floatvalue = (float*) variance_.get();
        for(int i=0; i<channels; i++){
            coreml_layer_->batchnorm->variance->floatvalue[i] = 1;
        }
#else
        coreml_layer_->batchnorm->variance->float16value.len = layer_res->scale_handle.GetBytesSize();
        variance_ = std::shared_ptr<uint16_t>(new uint16_t[channels], [](uint16_t* p) { delete[] p; });
        auto variance = (uint16_t*) variance_.get();
        for(int i=0; i<channels; i++){
            variance[i] = 0x3C00;  // fp16 1.0
        }
        coreml_layer_->batchnorm->variance->float16value.data = (uint8_t*) variance;
#endif
    }
    return TNN_OK;
}

Status CoreMLBatchnormLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLBatchnormLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLBatchnormLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Batchnorm, LAYER_BATCH_NORM);

}  // namespace TNN_NS
