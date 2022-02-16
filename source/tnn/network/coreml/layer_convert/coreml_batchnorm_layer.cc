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

DECLARE_COREML_LAYER_WITH_DATA(Batchnorm, LAYER_BATCH_NORM,
                                std::shared_ptr<void> coreml_layer_type_;
                                std::shared_ptr<void> coreml_layer_gamma_;
                                std::shared_ptr<void> coreml_layer_beta_;
                                std::shared_ptr<void> coreml_layer_mean_;
                                std::shared_ptr<void> coreml_layer_variance_;
                                std::shared_ptr<void> mean_;
                                std::shared_ptr<void> variance_;
                                std::shared_ptr<float> scale_fp32_ptr_;
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
    auto param = layer_info_->param.get();
    
    auto layer_res = dynamic_cast<BatchNormLayerResource *>(layer_resource_);
    CHECK_PARAM_NULL(layer_res);
    auto scale_count = layer_res->scale_handle.GetDataCount();
    auto scale_data_type = layer_res->scale_handle.GetDataType();
    auto bias_count = layer_res->bias_handle.GetDataCount();
    auto bias_data_type = layer_res->bias_handle.GetDataType();
    
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
    coreml_layer_->batchnorm->computemeanvar = false;
    coreml_layer_->batchnorm->instancenormalization = false;
    coreml_layer_gamma_ = std::shared_ptr<CoreML__Specification__WeightParams>(new CoreML__Specification__WeightParams);
    coreml_layer_->batchnorm->gamma = (CoreML__Specification__WeightParams*) coreml_layer_gamma_.get();
    core_ml__specification__weight_params__init(coreml_layer_->batchnorm->gamma);
    switch (scale_data_type) {
        case DATA_TYPE_FLOAT:
            if (share_channel) {
                //share channel mode
                scale_fp32_ptr_ = std::shared_ptr<float>(new float [channels], [](float* p) { delete[] p; });
                auto scale_fp32_ptr = scale_fp32_ptr_.get();
                for (int ii=0; ii<channels; ii++) {
                    scale_fp32_ptr[ii] = *(layer_res->scale_handle.force_to<float *>());
                }
                coreml_layer_->batchnorm->gamma->n_floatvalue = channels;
                coreml_layer_->batchnorm->gamma->floatvalue = scale_fp32_ptr;
            } else {
                coreml_layer_->batchnorm->gamma->n_floatvalue = channels;
                coreml_layer_->batchnorm->gamma->floatvalue = layer_res->scale_handle.force_to<float *>();
            }
            break;
        case DATA_TYPE_HALF:
            {
#if TNN_COREML_FULL_PRECISION
                if (share_channel) {
                    //share channel mode
                    void *scale_data_ptr = layer_res->scale_handle.force_to<void *>();
                    scale_fp32_ptr_ = std::shared_ptr<float>(new float [channels], [](float* p) { delete[] p; });
                    auto scale_fp32_ptr = scale_fp32_ptr_.get();
                    RETURN_ON_NEQ(ConvertFromHalfToFloat((void *)scale_data_ptr, (float *)scale_fp32_ptr, 1),TNN_OK);
                    for (int ii=1; ii<channels; ii++) {
                        scale_fp32_ptr[ii] = scale_fp32_ptr[0];
                    }
                    coreml_layer_->batchnorm->gamma->n_floatvalue = channels;
                    coreml_layer_->batchnorm->gamma->floatvalue = scale_fp32_ptr;
                } else {
                    void *scale_data_ptr = layer_res->scale_handle.force_to<void *>();
                    scale_fp32_ptr_ = std::shared_ptr<float>(new float [channels], [](float* p) { delete[] p; });
                    auto scale_fp32_ptr = scale_fp32_ptr_.get();
                    RETURN_ON_NEQ(ConvertFromHalfToFloat((void *)scale_data_ptr, (float *)scale_fp32_ptr, channels),TNN_OK);
                    coreml_layer_->batchnorm->gamma->n_floatvalue = channels;
                    coreml_layer_->batchnorm->gamma->floatvalue = scale_fp32_ptr;
                }
#else
                coreml_layer_->batchnorm->gamma->float16value.len = layer_res->scale_handle.GetBytesSize();
                coreml_layer_->batchnorm->gamma->float16value.data = layer_res->scale_handle.force_to<uint8_t *>();
#endif
            }
            break;
        default:
            {
                LOGE("CoreMLBatchnormLayer dont support data type (%d)\n", scale_data_type);
                return Status(TNNERR_PARAM_ERR, "CoreMLBatchnormLayer dont support data type");
            }
            break;
    }
    
    coreml_layer_beta_ = std::shared_ptr<CoreML__Specification__WeightParams>(new CoreML__Specification__WeightParams);
    coreml_layer_->batchnorm->beta = (CoreML__Specification__WeightParams*) coreml_layer_beta_.get();
    core_ml__specification__weight_params__init(coreml_layer_->batchnorm->beta);
    if (channels > bias_count) {
        float default_bias_value = 0;
        if (share_channel && bias_count != 0) {
            default_bias_value = *(layer_res->bias_handle.force_to<float *>());
        }
        bias_fp32_ptr_ = std::shared_ptr<float>(new float [channels], [](float* p) { delete[] p; });
        auto bias_fp32_ptr = bias_fp32_ptr_.get();
        for (int ii=0; ii<channels; ii++) {
            bias_fp32_ptr[ii] = default_bias_value;
        }
    }
    
    switch (bias_data_type) {
        case DATA_TYPE_FLOAT:
            if (channels > bias_count) {
                coreml_layer_->batchnorm->beta->n_floatvalue = channels;
                coreml_layer_->batchnorm->beta->floatvalue = bias_fp32_ptr_.get();
            } else {
                coreml_layer_->batchnorm->beta->n_floatvalue = bias_count;
                coreml_layer_->batchnorm->beta->floatvalue = layer_res->bias_handle.force_to<float *>();
            }
            break;
        case DATA_TYPE_HALF:
            {
#if TNN_COREML_FULL_PRECISION
                if (channels > bias_count) {
                    coreml_layer_->batchnorm->beta->n_floatvalue = channels;
                    coreml_layer_->batchnorm->beta->floatvalue = bias_fp32_ptr_.get();
                } else {
                    coreml_layer_->batchnorm->beta->n_floatvalue = bias_count;
                    void *bias_data_ptr = layer_res->bias_handle.force_to<void *>();
                    bias_fp32_ptr_ = std::shared_ptr<float>(new float [bias_count], [](float* p) { delete[] p; });
                    auto bias_fp32_ptr = bias_fp32_ptr_.get();
                    RETURN_ON_NEQ(ConvertFromHalfToFloat((void *)bias_data_ptr, (float *)bias_fp32_ptr, bias_count),TNN_OK);
                    coreml_layer_->batchnorm->beta->floatvalue = bias_fp32_ptr;
                }
#else
                if (channels > bias_count) {
                    coreml_layer_->batchnorm->beta->float16value.len = channels;
                    coreml_layer_->batchnorm->beta->float16value.data = bias_fp32_ptr_.get();
                } else {
                    coreml_layer_->batchnorm->beta->float16value.len = layer_res->bias_handle.GetBytesSize();
                    coreml_layer_->batchnorm->beta->float16value.data = layer_res->bias_handle.force_to<uint8_t *>();
                }
#endif
            }
            break;
        default:
            {
                LOGE("CoreMLBatchnormLayer dont support data type (%d)\n", bias_data_type);
                return Status(TNNERR_PARAM_ERR, "CoreMLBatchnormLayer dont support data type");
            }
            break;
    }
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
