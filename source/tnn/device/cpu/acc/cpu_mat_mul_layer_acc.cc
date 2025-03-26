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

#include "cpu_layer_acc.h"
#include "tnn/device/cpu/acc/cpu_unary_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {
//DECLARE_CPU_ACC(MatMul, LAYER_MATMUL);

class CpuMatMulLayerAcc : public CpuLayerAcc {
public:
    virtual ~CpuMatMulLayerAcc(){};
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs);
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

private:
    std::shared_ptr<float> weight_ = nullptr;
};

Status CpuMatMulLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
            const std::vector<Blob *> &outputs){
    auto status = CpuLayerAcc::Init(context, param, resource, inputs, outputs);

    if (inputs.size() == 2)
        return TNN_OK;
    auto layer_res = dynamic_cast<MatMulLayerResource *>(resource);

    const int data_size = layer_res->weight.GetDataCount();
    std::shared_ptr<float> weight(new float[data_size], [](float *p) { delete[] p; });

    if (layer_res->weight.GetDataType() == DATA_TYPE_FLOAT) {
        auto src_ptr = layer_res->weight.force_to<float *>();
        memcpy(weight.get(), src_ptr, data_size * sizeof(float));
    } else if (layer_res->weight.GetDataType() == DATA_TYPE_HALF) {
        auto src_ptr = layer_res->weight.force_to<fp16_t *>();
        ConvertFromHalfToFloat(src_ptr, weight.get(), data_size);
    } else {
        return Status(TNNERR_PARAM_ERR, "MatMul has invalid direction param");
    }

    weight_ = weight;

    return TNN_OK;
}

Status CpuMatMulLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}


//in align with onnx, use Tprecision=double to compute here for decision for fp types.
//or for align with bert model, use COSINE distance ??? not checked
template<typename Ta,typename Tb,typename Tprecision,typename Tc>
void CpuMatMulKernel(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs, float *weight,
                     const DimsVector &matrix_a_dims, const DimsVector &matrix_b_dims, const DimsVector & matrix_c_dims,
                     const MatMulLayerParam *param) {
    Ta *matrix_a;
    Tb *matrix_b;
    if (inputs.size() == 2) {
        matrix_a = static_cast<Ta *>(inputs[0]->GetHandle().base);
        matrix_b = static_cast<Tb *>(inputs[1]->GetHandle().base);
    } else {
        //matrix_a = param->weight_position == 0 ? weight_.get() : static_cast<Ta *>(inputs[0]->GetHandle().base);
        //matrix_b = param->weight_position == 1 ? weight_.get() : static_cast<Tb *>(inputs[0]->GetHandle().base);
        matrix_a = param->weight_position == 0 ? reinterpret_cast<Ta *>(weight) : static_cast<Ta *>(inputs[0]->GetHandle().base);
        matrix_b = param->weight_position == 1 ? reinterpret_cast<Tb *>(weight) : static_cast<Tb *>(inputs[0]->GetHandle().base);
    }
 
    Tc *matrix_c = static_cast<Tc *>(outputs[0]->GetHandle().base);
    int M        = matrix_a_dims[matrix_a_dims.size() - 2];
    int N        = matrix_a_dims[matrix_a_dims.size() - 1];
    int K        = matrix_b_dims[matrix_b_dims.size() - 1];
    int count_a  = DimsVectorUtils::Count(matrix_a_dims);
    int count_b  = DimsVectorUtils::Count(matrix_b_dims);
    int count_c  = DimsVectorUtils::Count(matrix_c_dims);
    int batch_a  = count_a / (M * N);
    int batch_b  = count_b / (N * K);
    int batch_c  = count_c / (M * K);

    for (int bc = 0; bc < batch_c; ++bc) {
        int ba = bc % batch_a;
        int bb = bc % batch_b;
            
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < K; ++k) {
                Tprecision sum = 0;
                for (int n = 0; n < N; ++n) {
                    sum += Tprecision(matrix_a[ba * M * N + m * N + n]) * Tprecision(matrix_b[bb * N * K + n * K + k]);
                }
                matrix_c[bc * M * K + m * K + k] = Tc(sum);
            }
        }
    }
}





Status CpuMatMulLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param               = dynamic_cast<MatMulLayerParam *>(param_);
    auto resource            = dynamic_cast<MatMulLayerResource *>(resource_);
    DimsVector matrix_a_dims = param->matrix_a_dims;
    DimsVector matrix_b_dims = param->matrix_b_dims;
    if (matrix_a_dims.size() == 1) {
        matrix_a_dims.insert(matrix_a_dims.begin(), 1);
    }
    if (matrix_b_dims.size() == 1) {
        matrix_b_dims.push_back(1);
    }
    auto matrix_c_dims       = outputs[0]->GetBlobDesc().dims;
    
    DataType matrix_a_dtype;
    DataType matrix_b_dtype;
    DataType matrix_c_dtype  = outputs[0]->GetBlobDesc().data_type;
    if (inputs.size() == 1) {
        if (param->weight_position == 0) {
            //matrix_a_dtype = resource->weight.GetDataType();
            matrix_a_dtype = DATA_TYPE_FLOAT;
            matrix_b_dtype = inputs[0]->GetBlobDesc().data_type;
        } else if (param->weight_position == 1) {
            matrix_a_dtype = inputs[0]->GetBlobDesc().data_type;
            //matrix_b_dtype = resource->weight.GetDataType();
            matrix_b_dtype = DATA_TYPE_FLOAT;
        } else {
            return Status(TNNERR_INVALID_MODEL, "MatMul input size error. param.weight_position invalid when num of input is 1.");
        }
    } else if (inputs.size() == 2) {
        matrix_a_dtype = inputs[0]->GetBlobDesc().data_type;
        matrix_b_dtype = inputs[1]->GetBlobDesc().data_type;
    } else {
        return Status(TNNERR_INVALID_MODEL, "MatMul OP number of inputs should be 1 or 2.");
    }


    if (matrix_c_dtype == DATA_TYPE_FLOAT) {
        if (matrix_a_dtype==DATA_TYPE_FLOAT && matrix_b_dtype==DATA_TYPE_FLOAT) {
            CpuMatMulKernel<float,float,double,float>(inputs, outputs, weight_.get(), matrix_a_dims, matrix_b_dims, matrix_c_dims, param); 
        } else if (matrix_a_dtype==DATA_TYPE_FLOAT && matrix_b_dtype==DATA_TYPE_HALF) {
            CpuMatMulKernel<float,fp16_t,double,float>(inputs, outputs, weight_.get(), matrix_a_dims, matrix_b_dims, matrix_c_dims, param); 
        } else if (matrix_a_dtype==DATA_TYPE_HALF && matrix_b_dtype==DATA_TYPE_FLOAT) {
            CpuMatMulKernel<fp16_t,float,double,float>(inputs, outputs, weight_.get(), matrix_a_dims, matrix_b_dims, matrix_c_dims, param); 
        } else {
            return Status(TNNERR_INVALID_MODEL, "MatMul OP CPU: data type combination of matrix a and b not supported.");
        }
    } else if (matrix_c_dtype == DATA_TYPE_HALF) {
        if (matrix_a_dtype==DATA_TYPE_HALF && matrix_b_dtype==DATA_TYPE_HALF) {
            CpuMatMulKernel<fp16_t,fp16_t,float,fp16_t>(inputs, outputs, weight_.get(), matrix_a_dims, matrix_b_dims, matrix_c_dims, param); 
        } else if (matrix_a_dtype==DATA_TYPE_FLOAT && matrix_b_dtype==DATA_TYPE_HALF) {
            CpuMatMulKernel<float,fp16_t,float,fp16_t>(inputs, outputs, weight_.get(), matrix_a_dims, matrix_b_dims, matrix_c_dims, param); 
        } else if (matrix_a_dtype==DATA_TYPE_HALF && matrix_b_dtype==DATA_TYPE_FLOAT) {
            CpuMatMulKernel<fp16_t,float,float,fp16_t>(inputs, outputs, weight_.get(), matrix_a_dims, matrix_b_dims, matrix_c_dims, param); 
        } else {
            return Status(TNNERR_INVALID_MODEL, "MatMul OP CPU: data type combination of matrix a and b not supported.");
        }
    } else {
        return Status(TNNERR_INVALID_MODEL, "MatMul OP CPU: OUTPUT matrix C, data type not supported.");
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(MatMul, LAYER_MATMUL)

}  // namespace TNN_NS
