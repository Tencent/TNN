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

DECLARE_COREML_LAYER_WITH_DATA(LayerNorm, LAYER_LAYER_NORM,
                                std::shared_ptr<void> coreml_layer_gamma_;
                                std::shared_ptr<void> coreml_layer_beta_;
                                std::shared_ptr<int64_t> normalizedshape_;
                                std::shared_ptr<float> scale_fp32_ptr_;
                                std::shared_ptr<float> bias_fp32_ptr_;);

Status CoreMLLayerNormLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_LAYER_NORMALIZATION;
    return TNN_OK;
}

Status CoreMLLayerNormLayer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto layer_param = dynamic_cast<LayerNormLayerParam *>(param);
    if (!layer_param || layer_info_->inputs.size() < 3 || !net_resource_) {
        LOGE("CoreMLLayerNormLayer has invalid layer_param or layer_info\n");
        return Status(TNNERR_MODEL_ERR, "CoreMLLayerNormLayer has invalid layer_param or layer_info");
    }
    
    auto scale_name = layer_info_->inputs[1];
    auto bias_name = layer_info_->inputs[2];
    if (net_resource_->constant_map.find(scale_name) == net_resource_->constant_map.end() ||
        net_resource_->constant_map.find(bias_name) == net_resource_->constant_map.end()) {
        LOGE("CoreMLLayerNormLayer has invalid net_resource\n");
        return Status(TNNERR_MODEL_ERR, "CoreMLLayerNormLayer has invalid net_resource");
    }
    
    auto scale_buffer = net_resource_->constant_map[scale_name];
    auto bias_buffer = net_resource_->constant_map[bias_name];
    
    auto scale_count = scale_buffer->GetDataCount();
    auto scale_dims = scale_buffer->GetBufferDims();
    auto scale_type = scale_buffer->GetDataType();
    auto bias_count = bias_buffer->GetDataCount();
    auto bias_type = bias_buffer->GetDataType();
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__LayerNormalizationLayerParams>(new CoreML__Specification__LayerNormalizationLayerParams);
    coreml_layer_->layernormalization = (CoreML__Specification__LayerNormalizationLayerParams *)coreml_layer_param_.get();
    core_ml__specification__layer_normalization_layer_params__init(coreml_layer_->layernormalization);
    coreml_layer_->layernormalization->eps = layer_param->eps;
    
    coreml_layer_gamma_ = std::shared_ptr<CoreML__Specification__WeightParams>(new CoreML__Specification__WeightParams);
    coreml_layer_->layernormalization->gamma = (CoreML__Specification__WeightParams*) coreml_layer_gamma_.get();
    core_ml__specification__weight_params__init(coreml_layer_->layernormalization->gamma);
    switch (scale_type) {
        case DATA_TYPE_FLOAT:
            coreml_layer_->layernormalization->gamma->n_floatvalue = scale_count;
            coreml_layer_->layernormalization->gamma->floatvalue = scale_buffer->force_to<float *>();
            break;
        case DATA_TYPE_HALF:
            {
#if TNN_COREML_FULL_PRECISION
                coreml_layer_->layernormalization->gamma->n_floatvalue = scale_count;
                void *scale_data_ptr = scale_buffer->force_to<void *>();
                scale_fp32_ptr_ = std::shared_ptr<float>(new float [scale_count], [](float* p) { delete[] p; });
                auto scale_fp32_ptr = scale_fp32_ptr_.get();
                RETURN_ON_NEQ(ConvertFromHalfToFloat((void *)scale_data_ptr, (float *)scale_fp32_ptr, scale_count),TNN_OK);
                coreml_layer_->layernormalization->gamma->floatvalue = scale_fp32_ptr;
#else
                coreml_layer_->layernormalization->gamma->float16value.len = scale_buffer->GetBytesSize();
                coreml_layer_->layernormalization->gamma->float16value.data = scale_buffer->force_to<uint8_t *>();
#endif
            }
            break;
        default:
            {
                LOGE("CoreMLLayerNormLayer dont support data type (%d)\n", scale_type);
                return Status(TNNERR_PARAM_ERR, "CoreMLLayerNormLayer dont support data type");
            }
            break;
    }
    coreml_layer_beta_ = std::shared_ptr<CoreML__Specification__WeightParams>(new CoreML__Specification__WeightParams);
    coreml_layer_->layernormalization->beta = (CoreML__Specification__WeightParams*) coreml_layer_beta_.get();
    core_ml__specification__weight_params__init(coreml_layer_->layernormalization->beta);
    switch (bias_type) {
        case DATA_TYPE_FLOAT:
            coreml_layer_->layernormalization->beta->n_floatvalue = bias_count;
            coreml_layer_->layernormalization->beta->floatvalue = bias_buffer->force_to<float *>();
            break;
        case DATA_TYPE_HALF:
            {
#if TNN_COREML_FULL_PRECISION
                coreml_layer_->layernormalization->beta->n_floatvalue = bias_count;
                void *bias_data_ptr = bias_buffer->force_to<void *>();
                bias_fp32_ptr_ = std::shared_ptr<float>(new float [bias_count], [](float* p) { delete[] p; });
                auto bias_fp32_ptr = bias_fp32_ptr_.get();
                RETURN_ON_NEQ(ConvertFromHalfToFloat((void *)bias_data_ptr, (float *)bias_fp32_ptr, bias_count),TNN_OK);
                coreml_layer_->layernormalization->beta->floatvalue = bias_fp32_ptr;
#else
                coreml_layer_->layernormalization->beta->float16value.len = bias_buffer->GetBytesSize();
                coreml_layer_->layernormalization->beta->float16value.data = bias_buffer->force_to<uint8_t *>();
#endif
            }
            break;
        default:
            {
                LOGE("CoreMLLayerNormLayer dont support data type (%d)\n", bias_type);
                return Status(TNNERR_PARAM_ERR, "CoreMLLayerNormLayer dont support data type");
            }
            break;
    }
    normalizedshape_ = std::shared_ptr<int64_t>(new int64_t [scale_dims.size()], [](int64_t* p) { delete[] p; });
    coreml_layer_->layernormalization->n_normalizedshape = scale_dims.size();
    coreml_layer_->layernormalization->normalizedshape = normalizedshape_.get();
    for (int i = 0; i < scale_dims.size(); i++) {
        coreml_layer_->layernormalization->normalizedshape[i] = scale_dims[i];
    }
    return TNN_OK;
}

Status CoreMLLayerNormLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLLayerNormLayer::BuildLayerInputs() {
    if (layer_info_) {
        return {layer_info_->inputs[0]};
    } else {
        return std::vector<std::string>();
    }
    
}

std::vector<std::string> CoreMLLayerNormLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(LayerNorm, LAYER_LAYER_NORM);

}  // namespace TNN_NS
