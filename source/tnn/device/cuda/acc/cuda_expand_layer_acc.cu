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


#include "tnn/device/cuda/acc/cuda_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/device/cuda/cuda_macro.h"

namespace TNN_NS {

DECLARE_CUDA_ACC_WITH_FUNC(Expand, LAYER_EXPAND,
                          virtual Status InferRuntimeOutputShape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs););

Status CudaExpandLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);;
}

Status CudaExpandLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaExpandLayerAcc::InferRuntimeOutputShape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto expand_param = dynamic_cast<ExpandLayerParam*>(param_);
    CHECK_PARAM_NULL(expand_param);
    
    if (inputs.size() == 2) {
        auto shape_data = (int *)inputs[1]->GetHandle().base;
        auto shape_data_count = DimsVectorUtils::Count(inputs[1]->GetBlobDesc().dims);
        DimsVector shape_dims(shape_data_count, 1);
        CUDA_CHECK(cudaMemcpy(&shape_dims[0], shape_data, shape_data_count * sizeof(int), cudaMemcpyDeviceToHost));
        expand_param->shape = shape_dims;
        
        auto data_dims = inputs[0]->GetBlobDesc().dims;
        auto output_dims = DimsVectorUtils::Expand(data_dims, shape_dims, nullptr);
        outputs[0]->GetBlobDesc().dims = output_dims;
    }
    
    return AbstractLayerAcc::InferRuntimeOutputShape(inputs, outputs);
}

typedef struct epand_dims_t{
    epand_dims_t(std::vector<int> dims) {
        memset(d, 0, maxDims * sizeof(int));
        nbDims = dims.size();
        for(int i=nbDims - 1;i>=0;i--) {
            d[i] = dims[i];
        }
    }
    constexpr static int maxDims=6;
    int nbDims=0;
    int d[maxDims];
} dims_t; 

typedef struct expand_steps_t {
    expand_steps_t(dims_t dims) {
        memset(d, 0, maxDims * sizeof(int));
        nbDims = dims.nbDims;
        int cnt = 1;
        for(int i=nbDims - 1;i>=0;i--) {
            if (dims.d[i] == 1) {
                d[i] = 0;
            } else {
                d[i] = cnt;
                cnt *= dims.d[i];
            }
        }
    }
    constexpr static int maxDims=6;
    int nbDims=0;
    int d[maxDims];
} steps_t; 

Status trim_dims(dims_t &a, dims_t &b) {
    if (a.nbDims != b.nbDims) {
        LOGE("nbDims not equal");
        return TNNERR_LAYER_ERR;
    }

    // trim the dimension when both are 1
    int i=0;
    int j=0;
    int trim_cnt = 0;
    for(;i<a.nbDims;i++) {
        if (a.d[i] == 1 && b.d[i] == 1) {
            trim_cnt += 1;
            continue;
        }
        a.d[j] = a.d[i];
        b.d[j] = b.d[i];
        j++;
    }
    a.nbDims -= trim_cnt;
    b.nbDims -= trim_cnt;

    // trim the leading broadcasting dims
    while(a.nbDims > 2 && a.d[0] == 1 && a.d[1]==1) {
        b.d[0] *= b.d[1];
        memcpy(b.d+1, b.d+2, (b.nbDims - 2) * sizeof(int));
        memcpy(a.d+1, a.d+2, (a.nbDims - 2) * sizeof(int));
        b.nbDims-= 1;
        a.nbDims-= 1;
    }

    return TNN_OK;
}

template<int ELE_PER_THREAD, int THRADS_PER_BLOCK, int INNER_NBDIMS>
__global__ void expand_kernel(const float *src, const dims_t src_dims, const steps_t src_steps, 
                              float * dst,  const dims_t dst_dims, const steps_t dst_steps, 
                              const int inner_size) 
{
    src += blockIdx.y * src_steps.d[0];
    dst += blockIdx.y * dst_steps.d[0];

    CUDA_KERNEL_LOOP(index, inner_size) {

        int inner_id = index;
        size_t src_idx = 0;
        size_t dst_idx = 0;

        #pragma unroll
        for(int i=INNER_NBDIMS;i>0;i--) {
            int pos = inner_id % dst_dims.d[i];
            inner_id /= dst_dims.d[i];
            src_idx += pos * src_steps.d[i];
            dst_idx += pos * dst_steps.d[i];
        }

        dst[dst_idx] = src[src_idx];
    }
}

Status CudaExpandLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_blob  = inputs[0];
    auto output_blob = outputs[0];
    auto input_dims = input_blob->GetBlobDesc().dims;
    auto output_dims = output_blob->GetBlobDesc().dims;

    // expand input dims
    auto expanded_input_dims = input_dims;
    while(expanded_input_dims.size() < output_dims.size()) {
        expanded_input_dims.insert(expanded_input_dims.begin(), 1);
    }

    dims_t src_dims(expanded_input_dims);
    dims_t dst_dims(output_dims);

    RETURN_ON_NEQ(trim_dims(src_dims, dst_dims), TNN_OK);

    steps_t src_steps(src_dims);
    steps_t dst_steps(dst_dims);

    // calc kernel shapes
    int outter_dim = dst_dims.d[0];
    int inner_dim = 1;
    for(int i=1;i<dst_dims.nbDims;i++) {
        inner_dim *= dst_dims.d[i];
    }

    const int ELE_PER_THREAD   = 1;
    const int THREAD_PER_BLOCK = 128;

    dim3 blocks;
    blocks.x = (inner_dim + THREAD_PER_BLOCK - 1 ) / THREAD_PER_BLOCK;
    blocks.y = outter_dim;

    float * src = (float*)(((char*)input_blob->GetHandle().base) + input_blob->GetHandle().bytes_offset);
    float * dst = (float*)(((char*)output_blob->GetHandle().base) + output_blob->GetHandle().bytes_offset);

    using kernel_function_ptr_t = decltype(&expand_kernel<1,1,1>);
    kernel_function_ptr_t kernel_ptr = nullptr;

    switch (dst_dims.nbDims - 1) {
        case 1 :
            kernel_ptr = expand_kernel<ELE_PER_THREAD, THREAD_PER_BLOCK, 1>;
            break;
        case 2 :
            kernel_ptr = expand_kernel<ELE_PER_THREAD, THREAD_PER_BLOCK, 2>;
            break;
        case 3 :
            kernel_ptr = expand_kernel<ELE_PER_THREAD, THREAD_PER_BLOCK, 3>;
            break;
        case 4 :
            kernel_ptr = expand_kernel<ELE_PER_THREAD, THREAD_PER_BLOCK, 4>;
            break;
        case 5 :
            kernel_ptr = expand_kernel<ELE_PER_THREAD, THREAD_PER_BLOCK, 5>;
            break;
        default:
            LOGE("unsupported configuration");
            return TNNERR_LAYER_ERR; 
    }

    kernel_ptr<<<blocks, THREAD_PER_BLOCK, 0, context_->GetStream()>>>
                    (src, src_dims, src_steps, dst, dst_dims, dst_steps, inner_dim);

    return TNN_OK;
}

REGISTER_CUDA_ACC(Expand, LAYER_EXPAND);

}  // namespace TNN_NS
