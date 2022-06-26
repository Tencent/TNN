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
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

DECLARE_COREML_LAYER_WITH_DATA(MatMul, LAYER_MATMUL,
                                    std::shared_ptr<CoreML__Specification__WeightParams> weight_param_;
                                    RawBuffer matrix_b_column_;);

Status CoreMLMatMulLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_BATCHED_MATMUL;
    return TNN_OK;
}

Status CoreMLMatMulLayer::BuildLayerParam() {
    //layer param
    auto param = dynamic_cast<MatMulLayerParam *>(layer_info_->param.get());
    auto resource = dynamic_cast<MatMulLayerResource *>(layer_resource_);
    if (!param || (layer_info_->inputs.size() == 1 && !resource)) {
        return Status(TNNERR_MODEL_ERR, "CoreMLMatMulLayer has invalid MatMulLayerParam or MatMulLayerResource");
    }
    
    auto matrix_a_dims = param->matrix_a_dims;
    auto matrix_b_dims = param->matrix_b_dims;
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__BatchedMatMulLayerParams>(new CoreML__Specification__BatchedMatMulLayerParams);
    coreml_layer_->batchedmatmul = (CoreML__Specification__BatchedMatMulLayerParams *)coreml_layer_param_.get();
    core_ml__specification__batched_mat_mul_layer_params__init(coreml_layer_->batchedmatmul);
    coreml_layer_->batchedmatmul->weightmatrixfirstdimension = matrix_a_dims.back();
    coreml_layer_->batchedmatmul->weightmatrixseconddimension = matrix_b_dims.back();
    coreml_layer_->batchedmatmul->hasbias = false;
    
    if (layer_info_->inputs.size() == 1) {
        //CoreML only support matrix b with dim size = 2, and matrix b is stored in column order
        if (matrix_b_dims.size() != 2) {
            return Status(TNNERR_MODEL_ERR, "CoreMLMatMulLayer has invalid matrix_b_dims");
        }
        
        if (param->weight_position == 0) {
            return Status(TNNERR_MODEL_ERR, "CoreMLMatMulLayer dont support constant matrix a");
        } else {
            weight_param_ = std::shared_ptr<CoreML__Specification__WeightParams>(new CoreML__Specification__WeightParams);
            coreml_layer_->batchedmatmul->weights = weight_param_.get();
            core_ml__specification__weight_params__init(coreml_layer_->batchedmatmul->weights);
            
            int weight_count = resource->weight.GetDataCount();
            auto matrix_b_data_type = resource->weight.GetDataType();
            switch (matrix_b_data_type) {
                case DATA_TYPE_FLOAT:
                    {
                        matrix_b_column_ = RawBuffer(weight_count*4, {matrix_b_dims[1], matrix_b_dims[0]});
                        matrix_b_column_.SetDataType(DATA_TYPE_FLOAT);
                        auto matrix_b_ptr = resource->weight.force_to<float *>();
                        auto matrix_b_column_ptr = matrix_b_column_.force_to<float *>();
                        for (int i=0; i<matrix_b_dims[1]; i++) {
                            for (int j=0; j<matrix_b_dims[0]; j++) {
                                *(matrix_b_column_ptr++) = matrix_b_ptr[j*matrix_b_dims[1] + i];
                            }
                        }
                        coreml_layer_->batchedmatmul->weights->n_floatvalue = matrix_b_column_.GetDataCount();
                        coreml_layer_->batchedmatmul->weights->floatvalue = matrix_b_column_.force_to<float *>();
                    }
                    break;
                case DATA_TYPE_HALF:
                    {
#if TNN_COREML_FULL_PRECISION
                        matrix_b_column_ = RawBuffer(weight_count*4, {matrix_b_dims[1], matrix_b_dims[0]});
                        matrix_b_column_.SetDataType(DATA_TYPE_FLOAT);
                        auto matrix_b_fp16_ptr = resource->weight.force_to<void *>();
                        int element_size = resource->weight.GetDataCount();
                        auto matrix_b_fp32_ptr_ = std::shared_ptr<float>(new float [element_size], [](float* p) { delete[] p; });
                        auto matrix_b_fp32_ptr = matrix_b_fp32_ptr_.get();
                        RETURN_ON_NEQ(ConvertFromHalfToFloat((void *)matrix_b_fp16_ptr, (float *)matrix_b_fp32_ptr, element_size),TNN_OK);
                        auto matrix_b_column_ptr = matrix_b_column_.force_to<float *>();
                        for (int i=0; i<matrix_b_dims[1]; i++) {
                            for (int j=0; j<matrix_b_dims[0]; j++) {
                                *(matrix_b_column_ptr++) = matrix_b_fp32_ptr[j*matrix_b_dims[1] + i];
                            }
                        }
                        coreml_layer_->batchedmatmul->weights->n_floatvalue = matrix_b_column_.GetDataCount();
                        coreml_layer_->batchedmatmul->weights->floatvalue = matrix_b_column_.force_to<float *>();
#else
                        matrix_b_column_ = RawBuffer(weight_count*2, {matrix_b_dims[1], matrix_b_dims[0]});
                        matrix_b_column_.SetDataType(DATA_TYPE_HALF);
                        auto matrix_b_ptr = resource->weight.force_to<uint16_t *>();
                        auto matrix_b_column_ptr = matrix_b_column_.force_to<uint16_t *>();
                        for (int i=0; i<matrix_b_dims[1]; i++) {
                            for (int j=0; j<matrix_b_dims[0]; j++) {
                                *(matrix_b_column_ptr++) = matrix_b_ptr[j*matrix_b_dims[1] + i];
                            }
                        }
                        coreml_layer_->batchedmatmul->weights->float16value.len = resource->weight.GetBytesSize();
                        coreml_layer_->batchedmatmul->weights->float16value.data = matrix_b_column_.force_to<uint8_t *>();
#endif
                    }
                    break;
                default:
                    {
                        LOGE("CoreMLMatMulLayer dont support data type (%d)\n", matrix_b_data_type);
                        return Status(TNNERR_PARAM_ERR, "CoreMLMatMulLayer dont support data type");
                    }
                    break;
            }
        }
    }
    
    return TNN_OK;
}

Status CoreMLMatMulLayer::BuildConstantWeightsLayer() {
    if (layer_info_ && layer_info_->inputs.size() > 1) {
        //weight in constantmap
        for (auto iter : layer_info_->inputs) {
            if (net_resource_->constant_map.find(iter) != net_resource_->constant_map.end()) {
                auto weight_buffer = net_resource_->constant_map[iter];
                auto weight_layer = std::make_shared<CoreMLConstLayer>(LAYER_CONST);
                auto status = weight_layer->Init(iter, *(weight_buffer.get()));
                RETURN_ON_NEQ(status, TNN_OK);
                
                coreml_layer_constant_weights_.push_back(weight_layer);
            }
        }
    }
    return TNN_OK;
}

std::vector<std::string> CoreMLMatMulLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLMatMulLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(MatMul, LAYER_MATMUL);

}  // namespace TNN_NS
