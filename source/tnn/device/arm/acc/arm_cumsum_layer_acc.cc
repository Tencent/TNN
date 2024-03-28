// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "tnn/device/arm/acc/arm_cumsum_layer_acc.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/dims_function_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

// TODO: Same as CPU Cumsum right now, add NEON SIMD speed-up
template <typename T>
void ArmCumsumKernel(const T* input, T* output, const int dim_pre, const int dim_curr, const int dim_post,
                     const bool exclusive, const bool exclusive_extend, const bool reverse) {
    // Set std::function according to 'exclusive' and 'reverse' settings.
    std::function<void(const T*, T*, int, int, const int, const int)> func_cumsum_loop;
    if (exclusive && !exclusive_extend && reverse) {
        func_cumsum_loop = [](const T* input, T* output, int offset, int out_offset, const int dim_curr, const int dim_post) -> void {
            T curr_cumsum = T(0);
            offset += dim_post * (dim_curr - 1);
            for (int i=0; i<dim_curr; i++) {
                output[offset] = curr_cumsum;
                curr_cumsum += input[offset];
                offset -= dim_post;
            }
        };
    } else if (exclusive && !exclusive_extend && !reverse) {
        func_cumsum_loop = [](const T* input, T* output, int offset, int out_offset, const int dim_curr, const int dim_post) -> void {
            T curr_cumsum = T(0);
            for (int i=0; i<dim_curr; i++) {
                output[offset] = curr_cumsum;
                curr_cumsum += input[offset];
                offset += dim_post;
            }
        };
    } else if (!exclusive && exclusive_extend && reverse) {
        func_cumsum_loop = [](const T* input, T* output, int offset, int out_offset, const int dim_curr, const int dim_post) -> void {
            T curr_cumsum = T(0);
            offset += dim_post * (dim_curr - 1);
            out_offset += dim_post * dim_curr;
            for (int i=0; i<dim_curr+1; i++) {
                output[out_offset] = curr_cumsum;
                curr_cumsum += input[offset];
                offset -= dim_post;
                out_offset -= dim_post;
            }
        };
    } else if (!exclusive && exclusive_extend && !reverse) {
        func_cumsum_loop = [](const T* input, T* output, int offset, int out_offset, const int dim_curr, const int dim_post) -> void {
            T curr_cumsum = T(0);
            for (int i=0; i<dim_curr+1; i++) {
                output[out_offset] = curr_cumsum;
                curr_cumsum += input[offset];
                offset += dim_post;
                out_offset += dim_post;
            }
        };
    } else if (!exclusive && !exclusive_extend && reverse) {
        func_cumsum_loop = [](const T* input, T* output, int offset, int out_offset, const int dim_curr, const int dim_post) -> void {
            T curr_cumsum = T(0);
            offset += dim_post * (dim_curr - 1);
            for (int i=0; i<dim_curr; i++) {
                curr_cumsum += input[offset];
                output[offset] = curr_cumsum;
                offset -= dim_post;
            }
        };
    } else { // !exclusive && !reverse
        func_cumsum_loop = [](const T* input, T* output, int offset, int out_offset, const int dim_curr, const int dim_post) -> void {
            T curr_cumsum = T(0);
            for (int i=0; i<dim_curr; i++) {
                curr_cumsum += input[offset];
                output[offset] = curr_cumsum;
                offset += dim_post;
            }
        };
    }

    // Main Compute Loop
    for (int i = 0; i<dim_pre; i++) {
        for (int j = 0; j<dim_post; j++) {
            int curr_offset = i * dim_curr * dim_post + j;
            int curr_out_offset = i * (dim_curr+1) * dim_post + j; // used ONLY in Exclusive Extend Mode.
            func_cumsum_loop(input, output, curr_offset, curr_out_offset, dim_curr, dim_post);
        }
    }
}

Status ArmCumsumLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    // Operator Cumsum input.dim == output.dim
    auto input_blob  = inputs[0];
    auto output_blob = outputs[0];
    auto input_dims  = input_blob->GetBlobDesc().dims;

    auto cumsum_param  = dynamic_cast<CumsumLayerParam*>(param_);
    if (cumsum_param == nullptr) {
        LOGE("Error: CpuCumsumLayer forward load layer param failed\n");
        return Status(TNNERR_MODEL_ERR, "Error: CpuCumsumLayer forward Load layer param failed!");
    }
    if (cumsum_param->axis < 0) {
        cumsum_param->axis += input_dims.size();
    }

    int dim_pre  = 1;
    int dim_curr = input_dims[cumsum_param->axis];
    int dim_post = 1;
    for (int i=0; i<cumsum_param->axis; i++) {
        dim_pre *= input_dims[i];
    }
    for (int i=cumsum_param->axis+1; i<input_dims.size(); i++) {
        dim_post *= input_dims[i];
    }

    DataType in_dtype = input_blob->GetBlobDesc().data_type;
    if (in_dtype==DATA_TYPE_FLOAT) {
        float* input_data  = reinterpret_cast<float *>(GetBlobHandlePtr(input_blob->GetHandle()));
        float* output_data  = reinterpret_cast<float *>(GetBlobHandlePtr(output_blob->GetHandle()));
        ArmCumsumKernel<float>(input_data, output_data, dim_pre, dim_curr, dim_post,
                               cumsum_param->exclusive, cumsum_param->exclusive_extend, cumsum_param->reverse);
    } else if (in_dtype==DATA_TYPE_HALF) {
        fp16_t* input_data  = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(input_blob->GetHandle()));
        fp16_t* output_data = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output_blob->GetHandle()));
        ArmCumsumKernel<fp16_t>(input_data, output_data, dim_pre, dim_curr, dim_post,
                                cumsum_param->exclusive, cumsum_param->exclusive_extend, cumsum_param->reverse);
    } else if (in_dtype==DATA_TYPE_INT32) {
        int* input_data  = reinterpret_cast<int *>(GetBlobHandlePtr(input_blob->GetHandle()));
        int* output_data  = reinterpret_cast<int *>(GetBlobHandlePtr(output_blob->GetHandle()));
        ArmCumsumKernel<int>(input_data, output_data, dim_pre, dim_curr, dim_post,
                             cumsum_param->exclusive, cumsum_param->exclusive_extend, cumsum_param->reverse);
    } else {
        LOGE("Error: ArmCumsumLayerAcc don't support data type: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: ArmCumsumLayerAcc don't support data type");
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(Cumsum, LAYER_CUMSUM)
REGISTER_ARM_LAYOUT(LAYER_CUMSUM, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
