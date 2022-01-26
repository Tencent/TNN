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

#include "coreml_const_layer.h"

namespace TNN_NS {
Status CoreMLConstLayer::Init(std::string output_name ,RawBuffer raw_buffer) {
    output_name_ = output_name;
    raw_buffer_ = raw_buffer;
    
    auto status = CoreMLBaseLayer::Init(nullptr, nullptr);
    RETURN_ON_NEQ(status, TNN_OK);
    
    SetLayerName(output_name);
    return status;
}

std::string CoreMLConstLayer::GetLayerName() {
    if (output_name_.length() > 0) {
        return output_name_;
    } else {
        return CoreMLBaseLayer::GetLayerName();
    }
}

Status CoreMLConstLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_LOAD_CONSTANT_ND;
    return TNN_OK;
}

Status CoreMLConstLayer::BuildLayerParam() {
    //layer param
    coreml_layer_param_ = std::make_shared<CoreML__Specification__LoadConstantNDLayerParams>();
    coreml_layer_->loadconstantnd = (CoreML__Specification__LoadConstantNDLayerParams *)coreml_layer_param_.get();
    core_ml__specification__load_constant_ndlayer_params__init(coreml_layer_->loadconstantnd);
    
    auto element_size = raw_buffer_.GetDataCount();
    auto element_dims = raw_buffer_.GetBufferDims();
    
    //shape
    shape_ = std::shared_ptr<uint64_t>(new uint64_t [element_dims.size()], [](uint64_t* p) { delete[] p; });
    coreml_layer_->loadconstantnd->shape = shape_.get();
    if (element_dims.size() != 0) {
        coreml_layer_->loadconstant->n_shape = element_dims.size();
        for (int i = 0; i < element_dims.size(); i++) {
            coreml_layer_->loadconstant->shape[i] = element_dims[i];
        }
    } else if (element_size != 0) {
        coreml_layer_->loadconstant->n_shape = element_size;
        coreml_layer_->loadconstant->shape[0] = 1;
    } else {
        LOGE("CoreMLConstLayer weight shape is error\n");
        return Status(TNNERR_PARAM_ERR, "CoreMLConstLayer weight shape is error");
    }
    
    //weight value
    auto data_type = raw_buffer_.GetDataType() ;
    weight_param_ = std::make_shared<CoreML__Specification__WeightParams>();
    coreml_layer_->loadconstantnd->data = weight_param_.get();
    core_ml__specification__weight_params__init(coreml_layer_->loadconstantnd->data);
    
    //TODO: to chcek data type
    switch (data_type) {
        case DATA_TYPE_FLOAT:
            {
                coreml_layer_->loadconstantnd->data->n_floatvalue = element_size;
                coreml_layer_->loadconstantnd->data->floatvalue = raw_buffer_.force_to<float *>();
            }
            break;
        case DATA_TYPE_INT32:
            {
                //CoreML only support FP32, so we need convert int32 to fp32
                cvt_raw_buffer_ = RawBuffer(4*element_size, element_dims);
                cvt_raw_buffer_.SetDataType(DATA_TYPE_FLOAT);
                auto int32_data = raw_buffer_.force_to<int32_t*>();
                auto float_data = cvt_raw_buffer_.force_to<float*>();
                for (int i=0; i<element_size; i++) {
                    float_data[i] = int32_data[i];
                }
                coreml_layer_->loadconstantnd->data->n_floatvalue = element_size;
                coreml_layer_->loadconstantnd->data->floatvalue = cvt_raw_buffer_.force_to<float *>();
            }
            break;
        case DATA_TYPE_HALF:
            {
#if TNN_COREML_FULL_PRECISION
                cvt_raw_buffer_ = RawBuffer(4*element_size, element_dims);
                cvt_raw_buffer_.SetDataType(DATA_TYPE_FLOAT);
                auto fp16_data = raw_buffer_.force_to<void*>();
                auto float_data = cvt_raw_buffer_.force_to<float*>();
                RETURN_ON_NEQ(ConvertFromHalfToFloat((void *)fp16_data, (float *)float_data, element_size),TNN_OK);
                coreml_layer_->loadconstantnd->data->n_floatvalue = element_size;
                coreml_layer_->loadconstantnd->data->floatvalue = cvt_raw_buffer_.force_to<float *>();
#else
                coreml_layer_->loadconstantnd->data->float16value.len = raw_buffer_.GetBytesSize();
                coreml_layer_->loadconstantnd->data->float16value.data = raw_buffer_.force_to<uint8_t *>();
#endif
            }
            break;
        default:
            {
                LOGE("CoreMLConstLayer dont support data type (%d)\n", data_type);
                return Status(TNNERR_PARAM_ERR, "CoreMLConstLayer dont support data type");
            }
            break;
    }
    
    return TNN_OK;
}

Status CoreMLConstLayer::BuildConstantWeightsLayer() {
    return TNN_OK;
}

std::vector<std::string> CoreMLConstLayer::BuildLayerInputs() {
    return std::vector<std::string>();
}

std::vector<std::string> CoreMLConstLayer::BuildLayerOutputs() {
    return std::vector<std::string>{output_name_};
}

REGISTER_COREML_LAYER(Const, LAYER_CONST);
}  // namespace TNN_NS
