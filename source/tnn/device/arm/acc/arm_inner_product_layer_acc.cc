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

#include "tnn/core/blob_int8.h"
#include "tnn/device/arm/acc/compute/compute.h"
#include "tnn/device/arm/acc/compute/gemm_function.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"
#include "tnn/utils/cpu_utils.h"
#ifdef TNN_ARM82_USE_NEON
#include "tnn/device/arm/acc/compute_arm82/compute_sdot_int8.h"
#endif

namespace TNN_NS {

ArmInnerProductLayerAcc::~ArmInnerProductLayerAcc() {}

// pack int8 kernel: round up ic4, round up oc4
static void packweight_i8(const int8_t *src, int8_t *dst, const int oc, const int ic, const int hw) {
    auto ic_r4       = ROUND_UP(ic, 4);
    auto dst_step    = ic * hw;
    auto dst_step_r4 = ic_r4 * hw;
    for (int o = 0; o < oc; o++) {
        for (int i = 0; i < hw; i++) {
            for (int c = 0; c < ic; c++) {
                dst[o * dst_step_r4 + i * ic_r4 + c] = src[o * dst_step + c * hw + i];
            }
        }
    }
}

// pack float and bfp16 weights
template <typename T>
static void PackWeightO4(const T *src, T *dst, const int oc, const int ic) {
    const int oc_r4 = ROUND_UP(oc, 4);
    const int ic_r4 = ROUND_UP(ic, 4);
    for (int o = 0; o < oc_r4; o++) {
        int o_inner = o % 4;
        int o_outer = o / 4 * 4;
        for (int i = 0; i < ic_r4; i++) {
            if (i >= ic || o >= oc) {
                dst[i * 4 + o_outer * ic_r4 + o_inner] = 0;
            } else {
                dst[i * 4 + o_outer * ic_r4 + o_inner] = src[o * ic + i];
            }
        }
    }
}

// get 4 result at a time
template <typename T>
static void SGEMV(T *dst, const T *src, T *weight, const int oc_r4, const int ic_r4) {
    OMP_PARALLEL_FOR_
    for (int o = 0; o < oc_r4; o += 4) {
        auto weight_z = weight + o * ic_r4;
        Float4 acc(0.f);
        for (int i = 0; i < ic_r4; i += 4) {
            Float4 w0 = Float4::load(weight_z + i * 4 + 0);
            Float4 w1 = Float4::load(weight_z + i * 4 + 4);
            Float4 w2 = Float4::load(weight_z + i * 4 + 8);
            Float4 w3 = Float4::load(weight_z + i * 4 + 12);
            Float4 v0 = Float4::load(src + i);
            Float2 v0_0, v0_1;
            Float4::get_low(v0, v0_0);
            Float4::get_high(v0, v0_1);
            Float4::mla_lane0(acc, w0, v0_0);
            Float4::mla_lane1(acc, w1, v0_0);
            Float4::mla_lane0(acc, w2, v0_1);
            Float4::mla_lane1(acc, w3, v0_1);
        }
        Float4::save(dst + o, acc);
    }
}

Status ArmInnerProductLayerAcc::allocateBufferWeight(const std::vector<Blob *> &inputs,
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

        auto weight_data_type = w_handle.GetDataType();
        int ic                = dims_input[1] * DimsVectorUtils::Count(dims_input, 2);
        const int oc          = fc_param->num_output;
        auto data_byte_size   = DataTypeUtils::GetBytesSize(weight_data_type);
        if (weight_data_type == DATA_TYPE_FLOAT) {
            if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
                // transform weight dims from 4 to 2
                if (DimsVectorUtils::Count(dims_input, 2) > 1) {
                    RawBuffer reorder_buffer = RawBuffer(DimsVectorUtils::Count(dims_input, 2) *
                                                         ROUND_UP(dims_input[1], 4) * oc * data_byte_size);
                    for (int i = 0; i < oc; i++) {
                        auto dst_ptr = reorder_buffer.force_to<float *>() +
                                       i * DimsVectorUtils::Count(dims_input, 2) * ROUND_UP(dims_input[1], 4);
                        auto src_ptr = w_handle.force_to<float *>() + i * ic;
                        PackC4(dst_ptr, src_ptr, DimsVectorUtils::Count(dims_input, 2), dims_input[1]);
                    }

                    ic       = DimsVectorUtils::Count(dims_input, 2) * ROUND_UP(dims_input[1], 4);
                    w_handle = reorder_buffer;
                }

                auto weight_count = ROUND_UP(oc, 4) * ROUND_UP(ic, 4);
                buffer_weight_    = RawBuffer(weight_count * data_byte_size);
                PackWeightO4(w_handle.force_to<float *>(), buffer_weight_.force_to<float *>(), oc, ic);

                // both data and weight will use type bfp16
                RawBuffer bfp16_buffer(weight_count * sizeof(bfp16_t));
                ConvertFromFloatToBFP16(buffer_weight_.force_to<float *>(), bfp16_buffer.force_to<void *>(),
                                        weight_count);
                buffer_weight_ = bfp16_buffer;
            } else {
                // weight [oc, ic] -> transpose -> [ic, oc]
                RawBuffer tmp_transpose = RawBuffer(ic * oc * data_byte_size);
                float *transpose_ptr    = tmp_transpose.force_to<float *>();
                for (int i = 0; i < ic; ++i) {
                    for (int o = 0; o < oc; ++o) {
                        transpose_ptr[i * oc + o] = w_handle.force_to<float *>()[o * ic + i];
                    }
                }
                // weight [ic, oc] -> [oc/8, ic, 8]
                buffer_weight_ = RawBuffer(ic * ROUND_UP(oc, 8) * data_byte_size + NEON_KERNEL_EXTRA_LOAD);
                PackB_8(ic, oc, transpose_ptr, oc, buffer_weight_.force_to<float *>());
            }
        } else {
            auto hw = DimsVectorUtils::Count(dims_input, 2);
            auto weight_count = ROUND_UP(oc, 4) * ROUND_UP(dims_input[1], 4) * hw;
            buffer_weight_    = RawBuffer(weight_count * data_byte_size + NEON_KERNEL_EXTRA_LOAD);
#ifdef TNN_ARM82_USE_NEON
            if (support_int8_sdot_) {
                PackSDOTINT8WeightGemv(w_handle.force_to<int8_t *>(), buffer_weight_.force_to<int8_t *>(), oc, dims_input[1], hw);
            } else {
                packweight_i8(w_handle.force_to<int8_t *>(), buffer_weight_.force_to<int8_t *>(), oc, dims_input[1], hw);
            }
#else
            packweight_i8(w_handle.force_to<int8_t *>(), buffer_weight_.force_to<int8_t *>(), oc, dims_input[1], hw);
#endif
        }
    }

    return TNN_OK;
}

Status ArmInnerProductLayerAcc::allocateBufferBias(const std::vector<Blob *> &inputs,
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

            int total_byte_size = ROUND_UP(dims_output[1], 4) * DataTypeUtils::GetBytesSize(bias_handle.GetDataType());

            const int bias_handle_size = bias_handle.GetBytesSize();

            buffer_bias_ = RawBuffer(total_byte_size);
            memcpy(buffer_bias_.force_to<void *>(), bias_handle.force_to<void *>(), bias_handle_size);
        } else if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8) {
            // int 8 kernel always add bias, if not, set zeros
            buffer_bias_ = RawBuffer(ROUND_UP(dims_output[1], 4) * sizeof(int32_t));
        }
    }

    // alloc scale buffer for int8 kernel
    if (!buffer_scale_.GetBytesSize()) {
        if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8) {
            auto o_scale = reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource()->scale_handle;
            auto w_scale = fc_res->scale_handle;

            if (w_scale.GetDataType() == DATA_TYPE_HALF)
                w_scale = ConvertHalfHandle(w_scale);

            int total_byte_size = ROUND_UP(dims_output[1], 4) * sizeof(float);
            buffer_scale_       = RawBuffer(total_byte_size);
            auto w_scale_ptr    = w_scale.force_to<float *>();
            CHECK_PARAM_NULL(w_scale_ptr);
            auto o_scale_ptr = o_scale.force_to<float *>();
            CHECK_PARAM_NULL(o_scale_ptr);
            for (int i = 0; i < dims_output[1]; i++) {
                int scale_idx_w = w_scale.GetDataCount() == 1 ? 0 : i;
                int scale_idx_o = o_scale.GetDataCount() == 1 ? 0 : i;

                if (o_scale_ptr[scale_idx_o] >= FLT_MIN)
                    buffer_scale_.force_to<float *>()[i] = w_scale_ptr[scale_idx_w] / o_scale_ptr[scale_idx_o];
                else
                    buffer_scale_.force_to<float *>()[i] = 0.0;
            }
        }
    }
    return TNN_OK;
}

Status ArmInnerProductLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                     const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    auto input_data_type = inputs[0]->GetBlobDesc().data_type;
    if (input_data_type == DATA_TYPE_FLOAT || input_data_type == DATA_TYPE_BFP16 || input_data_type == DATA_TYPE_INT8) {
        if (input_data_type == DATA_TYPE_INT8) {
            gemv_func_ = GemvInt8;
#ifdef TNN_ARM82_USE_NEON
            support_int8_sdot_ = CpuUtils::CpuSupportInt8Dot();
            if (support_int8_sdot_) {
                gemv_func_ = GemvInt8Sdot;
            }
#endif
        }
        RETURN_ON_NEQ(allocateBufferWeight(inputs, outputs), TNN_OK);
        RETURN_ON_NEQ(allocateBufferBias(inputs, outputs), TNN_OK);
    }
#if TNN_ARM82
    else if (input_data_type == DATA_TYPE_HALF) {
        RETURN_ON_NEQ(allocateBufferWeightHalf(inputs, outputs), TNN_OK);
        RETURN_ON_NEQ(allocateBufferBiasHalf(inputs, outputs), TNN_OK);
    }
#endif  // TNN_ARM82
    else {
        LOGE("ARM InnerProduct not support data type: %d\n", input_data_type);
        return Status(TNNERR_LAYER_ERR, "ARM InnerProduct not support data type");
    }
    return TNN_OK;
}

/*
general template function for float and bfp16
in bfp16 mode, both data and weight data type are bfp16, is there any precision problem
*/
template <typename T>
Status ArmInnerProductLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    InnerProductLayerParam *fc_param = dynamic_cast<InnerProductLayerParam *>(param_);
    CHECK_PARAM_NULL(fc_param);

    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;
    auto ic          = DimsVectorUtils::Count(dims_input, 2) * ROUND_UP(dims_input[1], 4);
    auto oc_r4       = ROUND_UP(dims_output[1], 4);

    auto input_origin  = reinterpret_cast<T *>(GetBlobHandlePtr(input->GetHandle()));
    auto output_origin = reinterpret_cast<T *>(GetBlobHandlePtr(output->GetHandle()));
    for (int n = 0; n < dims_output[0]; n++) {
        auto input_ptr  = input_origin + n * ic;
        auto output_ptr = output_origin + n * oc_r4;

        SGEMV(output_ptr, input_ptr, buffer_weight_.force_to<T *>(), oc_r4, ic);

        if (fc_param->has_bias) {
            PostAddBias<T>(output_ptr, buffer_bias_.force_to<float *>(), 1, oc_r4 / 4);
        }
    }

    return TNN_OK;
}

template <>
Status ArmInnerProductLayerAcc::Exec<float>(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    InnerProductLayerParam *fc_param = reinterpret_cast<InnerProductLayerParam *>(param_);
    CHECK_PARAM_NULL(fc_param);

    DimsVector dims_input = inputs[0]->GetBlobDesc().dims;
    int batch             = dims_input[0];
    int channel           = dims_input[1];
    int hw                = DimsVectorUtils::Count(dims_input, 2);
    int ic                = dims_input[1] * DimsVectorUtils::Count(dims_input, 2);
    const int oc          = fc_param->num_output;
    auto data_byte_size   = DataTypeUtils::GetBytesSize(DATA_TYPE_FLOAT);
    const int input_size  = batch * ic * data_byte_size;
    const int bias_size   = oc * data_byte_size;
    const int output_size = batch * oc * data_byte_size;

    float *input_ptr  = reinterpret_cast<float *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    float *output_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(outputs[0]->GetHandle()));

    // input: nc4hw4 -> nchw if needed
    RawBuffer input_reordered;
    if (!FloatBlobCanIgnorePack(channel, hw)) {
        input_reordered      = RawBuffer(input_size);
        float *reordered_ptr = input_reordered.force_to<float *>();
        UnpackFloatBlob(reordered_ptr, input_ptr, batch, channel, hw);
        input_ptr = reordered_ptr;
    }

    float *tmp_output_ptr = output_ptr;
    RawBuffer output_reordered;
    if (!FloatBlobCanIgnorePack(oc, 1)) {
        output_reordered = RawBuffer(output_size);
        tmp_output_ptr   = output_reordered.force_to<float *>();
    }

    if (fc_param->has_bias) {
        OMP_PARALLEL_FOR_
        for (int b = 0; b < batch; ++b) {
            // output shape: [batch, oc]
            auto dst_ptr_b = tmp_output_ptr + b * oc;
            memcpy(dst_ptr_b, buffer_bias_.force_to<float *>(), bias_size);
        }
    } else {
        memset(tmp_output_ptr, 0, output_size);
    }

    // buffer for PackA in gemm
    auto input_pack_ptr = reinterpret_cast<float *>(context_->GetSharedWorkSpace(input_size + NEON_KERNEL_EXTRA_LOAD));

    GemmFloatPackA(batch, oc, ic, input_ptr, input_pack_ptr, ic, buffer_weight_.force_to<float *>(), oc, tmp_output_ptr,
                   oc);

    // output: nchw -> nc4hw4 if needed
    if (!FloatBlobCanIgnorePack(oc, 1)) {
        PackFloatBlob(output_ptr, tmp_output_ptr, batch, oc, 1);
    }

    return TNN_OK;
}

/*
template specification for int8
in int8 mode, weight has been packed to oc8
*/
template <>
Status ArmInnerProductLayerAcc::Exec<int8_t>(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    InnerProductLayerParam *fc_param = dynamic_cast<InnerProductLayerParam *>(param_);
    auto dims_input                  = inputs[0]->GetBlobDesc().dims;
    auto dims_output                 = outputs[0]->GetBlobDesc().dims;
    auto input_origin  = reinterpret_cast<int8_t *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    auto output_origin = reinterpret_cast<int8_t *>(GetBlobHandlePtr(outputs[0]->GetHandle()));

    auto ic    = dims_input[1];
    auto ic_r4 = ROUND_UP(ic, 4);
    auto hw    = DimsVectorUtils::Count(dims_input, 2);
    auto ik_r4 = ic_r4 * hw;
    auto oc_r4 = ROUND_UP(dims_output[1], 4);

    for (int n = 0; n < dims_output[0]; n++) {
        auto input_ptr  = input_origin + n * ik_r4;
        auto output_ptr = output_origin + n * oc_r4;

        gemv_func_(output_ptr, input_ptr, buffer_weight_.force_to<int8_t *>(), buffer_bias_.force_to<int32_t *>(),
                 buffer_scale_.force_to<float *>(), ik_r4, oc_r4);
    }

    return TNN_OK;
}

/*
general template function for bfp16 nchw
use n4chw4 impl, nchw impl tbd
*/
template <typename T>
Status ArmInnerProductLayerAcc::ExecNchw(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    InnerProductLayerParam *fc_param = dynamic_cast<InnerProductLayerParam *>(param_);
    CHECK_PARAM_NULL(fc_param);

    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;
    auto ic          = DimsVectorUtils::Count(dims_input, 1);
    auto ic_r4       = DimsVectorUtils::Count(dims_input, 2) * ROUND_UP(dims_input[1], 4);
    auto oc          = dims_output[1];
    auto oc_r4       = ROUND_UP(dims_output[1], 4);

    auto data_byte_size = DataTypeUtils::GetBytesSize(inputs[0]->GetBlobDesc().data_type);
    auto *work_space = reinterpret_cast<T *>(context_->GetSharedWorkSpace((ic_r4 + oc_r4) * data_byte_size));

    auto input_origin  = reinterpret_cast<T *>(GetBlobHandlePtr(input->GetHandle()));
    auto output_origin = reinterpret_cast<T *>(GetBlobHandlePtr(output->GetHandle()));
    auto input_pack    = work_space;
    auto output_pack   = work_space + ic_r4;
    for (int n = 0; n < dims_output[0]; n++) {
        auto input_ptr  = input_origin + n * ic;
        auto output_ptr = output_origin + n * oc;

        PackC4(input_pack, input_ptr, DimsVectorUtils::Count(dims_input, 2), dims_input[1]);

        SGEMV(output_pack, input_pack, buffer_weight_.force_to<T *>(), oc_r4, ic_r4);

        if (fc_param->has_bias) {
            PostAddBias<T>(output_pack, buffer_bias_.force_to<float *>(), 1, oc_r4 / 4);
        }

        UnpackC4(output_ptr, output_pack, 1, oc);
    }

    return TNN_OK;
}

template <>
Status ArmInnerProductLayerAcc::ExecNchw<float>(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    InnerProductLayerParam *fc_param = reinterpret_cast<InnerProductLayerParam *>(param_);
    CHECK_PARAM_NULL(fc_param);

    DimsVector dims_input = inputs[0]->GetBlobDesc().dims;
    int batch             = dims_input[0];
    int channel           = dims_input[1];
    int hw                = DimsVectorUtils::Count(dims_input, 2);
    int ic                = dims_input[1] * DimsVectorUtils::Count(dims_input, 2);
    const int oc          = fc_param->num_output;
    auto data_byte_size   = DataTypeUtils::GetBytesSize(DATA_TYPE_FLOAT);
    const int input_size  = batch * ic * data_byte_size;
    const int bias_size   = oc * data_byte_size;
    const int output_size = batch * oc * data_byte_size;

    float *input_ptr  = reinterpret_cast<float *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    float *output_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(outputs[0]->GetHandle()));
    float *tmp_output_ptr = output_ptr;

    if (fc_param->has_bias) {
        OMP_PARALLEL_FOR_
        for (int b = 0; b < batch; ++b) {
            // output shape: [batch, oc]
            auto dst_ptr_b = tmp_output_ptr + b * oc;
            memcpy(dst_ptr_b, buffer_bias_.force_to<float *>(), bias_size);
        }
    } else {
        memset(tmp_output_ptr, 0, output_size);
    }

    // buffer for PackA in gemm
    auto input_pack_ptr = reinterpret_cast<float *>(context_->GetSharedWorkSpace(input_size + NEON_KERNEL_EXTRA_LOAD));

    GemmFloatPackA(batch, oc, ic, input_ptr, input_pack_ptr, ic, buffer_weight_.force_to<float *>(), oc, tmp_output_ptr,
                   oc);

    return TNN_OK;
}

Status ArmInnerProductLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        return Exec<int8_t>(inputs, outputs);
    }

    if (inputs[0]->GetBlobDesc().data_format == DATA_FORMAT_NCHW) {
        if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
            return ExecNchw<float>(inputs, outputs);
        } else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
            return ExecNchw<bfp16_t>(inputs, outputs);
        }
#if TNN_ARM82
        else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_HALF) {
            return ExecNchwFp16(inputs, outputs);
        }
#endif
        else {
            return Status(TNNERR_LAYER_ERR, "Unsupported data type in innerproduct");
        }
    } else if (inputs[0]->GetBlobDesc().data_format == DATA_FORMAT_NC4HW4 ||
               inputs[0]->GetBlobDesc().data_format == DATA_FORMAT_NC8HW8) {
        if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
            return Exec<float>(inputs, outputs);
        } else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
            return Exec<bfp16_t>(inputs, outputs);
        }
#if TNN_ARM82
        else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_HALF) {
            return ExecFp16(inputs, outputs);
        }
#endif
        else {
            return Status(TNNERR_LAYER_ERR, "Unsupported data type in innerproduct");
        }
    } else {
        return Status(TNNERR_LAYER_ERR, "Unsupported data format in innerproduct");
    }
    return TNNERR_LAYER_ERR;
}

REGISTER_ARM_ACC(InnerProduct, LAYER_INNER_PRODUCT)
REGISTER_ARM_PRECISION_FP16(LAYER_INNER_PRODUCT)
REGISTER_ARM_LAYOUT(LAYER_INNER_PRODUCT, DATA_FORMAT_NC4HW4)
REGISTER_ARM_LAYOUT(LAYER_INNER_PRODUCT, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
