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

#include "tnn/device/metal/acc/metal_mat_mul_layer_acc.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/half_utils_inner.h"

namespace TNN_NS {
Status MetalMatMulLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                                   const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Init(context, param, resource, inputs, outputs);
}

MetalMatMulLayerAcc::~MetalMatMulLayerAcc() {}

Status MetalMatMulLayerAcc::ConfigBuffer2MetalBlobDesc(BlobDesc &desc) {
    desc.data_format = DATA_FORMAT_NCHW;
    return TNN_OK;
}

Status MetalMatMulLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    
    id<MTLDevice> device       = [TNNMetalDeviceImpl sharedDevice];
    auto param = dynamic_cast<MatMulLayerParam *>(param_);
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

    const int K = matrix_b_dims[matrix_b_dims.size() - 1];
    const int N = matrix_a_dims[matrix_a_dims.size() - 1];
    const int M = matrix_a_dims[matrix_a_dims.size() - 2];

    int count_a     = DimsVectorUtils::Count(matrix_a_dims);
    int count_b     = DimsVectorUtils::Count(matrix_b_dims);
    int count_c     = DimsVectorUtils::Count(matrix_c_dims);
    int batch_a   = count_a / (M * N);
    int batch_b   = count_b / (N * K);
    int batch_c   = count_c / (M * K);
    // buffer_param_
    {
        MetalMatMulParams metal_params;
        
        metal_params.batch_a = batch_a;
        metal_params.batch_b = batch_b;
        metal_params.batch_c = batch_c;
        metal_params.M       = M;
        metal_params.N       = N;
        metal_params.K       = K;
        
        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalMatMulParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

Status MetalMatMulLayerAcc::AllocateBufferWeight(const std::vector<Blob *> &inputs,
                                             const std::vector<Blob *> &outputs) {
    if (inputs.size() == 2) {
        //no const weight input
        return TNN_OK;
    }
    
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    if (inputs.size() >= 2) {
        //  both inputs are blobs, no layer_resource
        return TNN_OK;
    }
    auto layer_res = dynamic_cast<MatMulLayerResource *>(resource_);
    if (layer_res == nullptr) {
        return TNN_OK;
    }
    auto& weight   = layer_res->weight;

    const auto data_type = weight.GetDataType();
    void *data = weight.force_to<void *>();
    auto bytes = weight.GetBytesSize();
    const auto count = weight.GetDataCount();
    float *data_type_float = nullptr;
    uint16_t *data_type_half  = nullptr;
#if TNN_METAL_FULL_PRECISION
    if (data_type == DATA_TYPE_HALF) {
        data_type_float = new float[count];
        if (ConvertFromHalfToFloat(data, (float *)data_type_float, count) != 0) {
            LOGE("matmul weight DataType is not supported");
            delete[] data_type_float;
            return Status(TNNERR_LAYER_ERR, "matmul weight DataType is not supported");
        }
        data = data_type_float;
        bytes = count * sizeof(float);
    }        
#else       
    if (data_type == DATA_TYPE_FLOAT) {
        data_type_half = new uint16_t[count];
        if (ConvertFromFloatToHalf((float *)data, data_type_half, count) != 0) {
            LOGE("matmul weight DataType is not supported");
            delete[] data_type_half;
            return Status(TNNERR_LAYER_ERR, "matmul weight DataType is not supported");
        }
        data = data_type_half;
        bytes = count * sizeof(uint16_t);
    }
#endif
    
    if (!buffer_weight_) {
        buffer_weight_ = [device newBufferWithBytes:data
                                     length:bytes
                                    options:MTLResourceCPUCacheModeWriteCombined];
    }
    if (data_type_float != nullptr) delete[] data_type_float;
    if (data_type_half != nullptr) delete[] data_type_half;

    return TNN_OK;
}

Status MetalMatMulLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto status = AllocateBufferWeight(inputs, outputs);
    RETURN_ON_NEQ(status, TNN_OK);

    status = MetalLayerAcc::Reshape(inputs, outputs);
    
    return status;
}

std::vector<DataFormat> MetalMatMulLayerAcc::SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) {
    return {DATA_FORMAT_NCHW};
}


std::string MetalMatMulLayerAcc::KernelName(const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs) {
    return "matmul_common";
}

Status MetalMatMulLayerAcc::SetKernelEncoderParam(
                                            id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<MatMulLayerParam *>(param_);

    int bytes_offset_mat_a = 0;
    int bytes_offset_mat_b = 0;
    id<MTLBuffer> matrix_a = nil;
    id<MTLBuffer> matrix_b = nil;
    if (inputs.size() == 2) {
        matrix_a = (__bridge id<MTLBuffer>)(inputs[0]->GetHandle().base);
        matrix_b = (__bridge id<MTLBuffer>)(inputs[1]->GetHandle().base);
        bytes_offset_mat_a = inputs[0]->GetHandle().bytes_offset;
        bytes_offset_mat_b = inputs[1]->GetHandle().bytes_offset;
    } else {
        matrix_a = param->weight_position == 0 ? buffer_weight_ : 
                                (__bridge id<MTLBuffer>)(inputs[0]->GetHandle().base);
        matrix_b = param->weight_position == 1 ? buffer_weight_ :
                                (__bridge id<MTLBuffer>)(inputs[0]->GetHandle().base);
    }
    
    [encoder setBuffer:matrix_a
                offset:(NSUInteger)bytes_offset_mat_a
               atIndex:0];
    [encoder setBuffer:matrix_b
                offset:(NSUInteger)bytes_offset_mat_b
               atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)outputs[0]->GetHandle().base
                offset:(NSUInteger)outputs[0]->GetHandle().bytes_offset
               atIndex:2];
    [encoder setBuffer:buffer_param_ offset:0 atIndex:3];
    
    return TNN_OK;
}

Status MetalMatMulLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    auto param = dynamic_cast<MatMulLayerParam *>(param_);
    DimsVector matrix_a_dims = param->matrix_a_dims;
    DimsVector matrix_b_dims = param->matrix_b_dims;
    if (matrix_a_dims.size() == 1) {
        matrix_a_dims.insert(matrix_a_dims.begin(), 1);
    }
    if (matrix_b_dims.size() == 1) {
        matrix_b_dims.push_back(1);
    }
    auto matrix_c_dims       = outputs[0]->GetBlobDesc().dims;
    const int K = matrix_b_dims[matrix_b_dims.size() - 1];
    const int N = matrix_a_dims[matrix_a_dims.size() - 1];
    const int M = matrix_a_dims[matrix_a_dims.size() - 2];

    int count_c = DimsVectorUtils::Count(matrix_c_dims);
    int batch_c = count_c / (M * K);

    size = MTLSizeMake(K, M, batch_c);
    return TNN_OK;
}

Status MetalMatMulLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Forward(inputs, outputs);
}

REGISTER_METAL_ACC(MatMul, LAYER_MATMUL);
REGISTER_METAL_LAYOUT(LAYER_MATMUL, DATA_FORMAT_NCHW);

} // namespace TNN_NS
