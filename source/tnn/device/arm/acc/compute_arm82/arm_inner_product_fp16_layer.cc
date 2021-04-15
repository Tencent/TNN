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

#include "tnn/device/arm/acc/arm_inner_product_layer_acc.h"
#include "tnn/device/arm/acc/compute/compute.h"
#include "tnn/device/arm/acc/compute/gemm_function.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

#if TNN_ARM82

Status ArmInnerProductLayerAcc::allocateBufferWeightHalf(const std::vector<Blob *> &inputs,
                                                         const std::vector<Blob *> &outputs) {
    InnerProductLayerParam *fc_param = dynamic_cast<InnerProductLayerParam *>(param_);
    CHECK_PARAM_NULL(fc_param);
    InnerProductLayerResource *fc_res = dynamic_cast<InnerProductLayerResource *>(resource_);
    CHECK_PARAM_NULL(fc_res);

    if (!buffer_weight_.GetBytesSize()) {
        DimsVector dims_input  = inputs[0]->GetBlobDesc().dims;
        DimsVector dims_output = outputs[0]->GetBlobDesc().dims;

        RawBuffer w_handle = fc_res->weight_handle;
        CHECK_PARAM_NULL(w_handle.force_to<void *>());

        if (w_handle.GetDataType() == DATA_TYPE_HALF)
            w_handle = ConvertHalfHandle(w_handle);

        auto data_byte_size = sizeof(fp16_t);
        int ic              = dims_input[1] * DimsVectorUtils::Count(dims_input, 2);
        const int oc        = fc_param->num_output;
        // weight [oc, ic] -> transpose -> [ic, oc]
        RawBuffer tmp_transpose = RawBuffer(ic * oc * data_byte_size);
        fp16_t *transpose_ptr   = tmp_transpose.force_to<fp16_t *>();
        for (int i = 0; i < ic; ++i) {
            for (int o = 0; o < oc; ++o) {
                transpose_ptr[i * oc + o] = (fp16_t)w_handle.force_to<float *>()[o * ic + i];
            }
        }
        // weight [ic, oc] -> [oc/16, ic, 16]
        buffer_weight_ = RawBuffer(ic * ROUND_UP(oc, 16) * data_byte_size + NEON_KERNEL_EXTRA_LOAD);
        PackB_16(ic, oc, transpose_ptr, oc, buffer_weight_.force_to<fp16_t *>());
    }

    return TNN_OK;
}

Status ArmInnerProductLayerAcc::allocateBufferBiasHalf(const std::vector<Blob *> &inputs,
                                                       const std::vector<Blob *> &outputs) {
    InnerProductLayerParam *fc_param = dynamic_cast<InnerProductLayerParam *>(param_);
    CHECK_PARAM_NULL(fc_param);
    InnerProductLayerResource *fc_res = dynamic_cast<InnerProductLayerResource *>(resource_);
    CHECK_PARAM_NULL(fc_res);
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    if (!buffer_bias_.GetBytesSize()) {
        if (fc_param->has_bias) {
            auto bias_handle = fc_res->bias_handle;

            if (bias_handle.GetDataType() == DATA_TYPE_HALF)
                bias_handle = ConvertHalfHandle(bias_handle);

            int total_byte_size = ROUND_UP(dims_output[1], 8) * sizeof(fp16_t);
            buffer_bias_        = RawBuffer(total_byte_size);
            ConvertFromFloatToHalf(bias_handle.force_to<float *>(), buffer_bias_.force_to<fp16_t *>(), dims_output[1]);
        }
    }
    return TNN_OK;
}

Status ArmInnerProductLayerAcc::ExecFp16(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    InnerProductLayerParam *fc_param = reinterpret_cast<InnerProductLayerParam *>(param_);
    CHECK_PARAM_NULL(fc_param);

    DimsVector dims_input = inputs[0]->GetBlobDesc().dims;
    int batch             = dims_input[0];
    int channel           = dims_input[1];
    int hw                = DimsVectorUtils::Count(dims_input, 2);
    int ic                = dims_input[1] * DimsVectorUtils::Count(dims_input, 2);
    const int oc          = fc_param->num_output;
    auto data_byte_size   = DataTypeUtils::GetBytesSize(DATA_TYPE_HALF);
    const int input_size  = batch * ic * data_byte_size;
    const int bias_size   = oc * data_byte_size;
    const int output_size = batch * oc * data_byte_size;

    fp16_t *input_ptr  = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    fp16_t *output_ptr = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(outputs[0]->GetHandle()));

    // input: nc8hw8 -> nchw if needed
    RawBuffer input_reordered;
    if (!HalfBlobCanIgnorePack(channel, hw)) {
        input_reordered       = RawBuffer(input_size);
        fp16_t *reordered_ptr = input_reordered.force_to<fp16_t *>();
        UnpackHalfBlob(reordered_ptr, input_ptr, batch, channel, hw);
        input_ptr = reordered_ptr;
    }

    fp16_t *tmp_output_ptr = output_ptr;
    RawBuffer output_reordered;
    if (!HalfBlobCanIgnorePack(oc, 1)) {
        output_reordered = RawBuffer(output_size);
        tmp_output_ptr   = output_reordered.force_to<fp16_t *>();
    }

    if (fc_param->has_bias) {
        OMP_PARALLEL_FOR_
        for (int b = 0; b < batch; ++b) {
            // output shape: [batch, oc]
            auto dst_ptr_b = tmp_output_ptr + b * oc;
            memcpy(dst_ptr_b, buffer_bias_.force_to<fp16_t *>(), bias_size);
        }
    } else {
        memset(tmp_output_ptr, 0, output_size);
    }

    // buffer for PackA in gemm
    auto input_pack_ptr = reinterpret_cast<fp16_t *>(context_->GetSharedWorkSpace(input_size + NEON_KERNEL_EXTRA_LOAD));

    GemmHalfPackA(batch, oc, ic, input_ptr, input_pack_ptr, ic, buffer_weight_.force_to<fp16_t *>(), oc, tmp_output_ptr,
                  oc);

    // output: nchw -> nc8hw8 if needed
    if (!HalfBlobCanIgnorePack(oc, 1)) {
        PackHalfBlob(output_ptr, tmp_output_ptr, batch, oc, 1);
    }

    return TNN_OK;
}

Status ArmInnerProductLayerAcc::ExecNchwFp16(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    InnerProductLayerParam *fc_param = reinterpret_cast<InnerProductLayerParam *>(param_);
    CHECK_PARAM_NULL(fc_param);

    DimsVector dims_input = inputs[0]->GetBlobDesc().dims;
    int batch             = dims_input[0];
    int channel           = dims_input[1];
    int hw                = DimsVectorUtils::Count(dims_input, 2);
    int ic                = dims_input[1] * DimsVectorUtils::Count(dims_input, 2);
    const int oc          = fc_param->num_output;
    auto data_byte_size   = DataTypeUtils::GetBytesSize(DATA_TYPE_HALF);
    const int input_size  = batch * ic * data_byte_size;
    const int bias_size   = oc * data_byte_size;
    const int output_size = batch * oc * data_byte_size;

    fp16_t *input_ptr  = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    fp16_t *output_ptr = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(outputs[0]->GetHandle()));
    fp16_t *tmp_output_ptr = output_ptr;

    if (fc_param->has_bias) {
        OMP_PARALLEL_FOR_
        for (int b = 0; b < batch; ++b) {
            // output shape: [batch, oc]
            auto dst_ptr_b = tmp_output_ptr + b * oc;
            memcpy(dst_ptr_b, buffer_bias_.force_to<fp16_t *>(), bias_size);
        }
    } else {
        memset(tmp_output_ptr, 0, output_size);
    }

    // buffer for PackA in gemm
    auto input_pack_ptr = reinterpret_cast<fp16_t *>(context_->GetSharedWorkSpace(input_size + NEON_KERNEL_EXTRA_LOAD));

    GemmHalfPackA(batch, oc, ic, input_ptr, input_pack_ptr, ic, buffer_weight_.force_to<fp16_t *>(), oc, tmp_output_ptr,
                  oc);

    return TNN_OK;
}

#endif  // TNN_ARM82

}  // namespace TNN_NS
