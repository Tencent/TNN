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

#include "tnn/device/arm/acc/convolution/arm_conv_layer_1x1.h"

#ifdef TNN_USE_NEON
#include <arm_neon.h>
#endif

#include "tnn/device/arm/acc/Float4.h"
#include "tnn/device/arm/acc/compute/gemm_function.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

/*
pack func, can be treated as img2col in conv1x1
*/
template <typename T>
static void PackLine(T *dst, T *src, int ih, int iw, int oh, int ow, int c_r4, int pad_h, int pad_w, int stride_h,
                     int stride_w) {
    if (pad_h != 0 || pad_w != 0)
        memset(dst, 0, c_r4 * oh * ow * sizeof(T));

    dst += pad_h * ow * 4 + pad_w * 4;

    for (int c = 0; c < c_r4; c += 4) {
        auto dst_c = dst + c * oh * ow;
        auto src_c = src + c * ih * iw;
        if (stride_w == 1 && stride_h == 1) {
            for (int h = 0; h < ih; h++) {
                memcpy(dst_c + h * ow * 4, src_c + h * iw * 4, iw * 4 * sizeof(T));
            }
        } else if (pad_w == 0 && pad_h == 0) {
            for (int h = 0; h < oh; h++) {
                auto dst_h = dst_c + h * ow * 4;
                auto src_h = src_c + h * stride_h * iw * 4;
                for (int w = 0; w < ow; w++) {
                    Float4::save(dst_h + w * 4, Float4::load(src_h + w * stride_w * 4));
                }
            }
        } else {
            Float4 zeros(0.f);
            for (int h = 0; h < oh; h++) {
                int sh = h * stride_h - pad_h;
                if (sh >= 0 && sh < ih) {
                    auto dst_h = dst_c + (h - pad_h) * ow * 4;
                    auto src_h = src_c + (h * stride_h - pad_h) * iw * 4;
                    for (int w = 0; w < ow; w++) {
                        int sw = w * stride_w - pad_w;
                        if (sw >= 0 && sw < iw) {
                            Float4::save(dst_h + (w - pad_w) * 4, Float4::load(src_h + (w * stride_w - pad_w) * 4));
                        }
                    }
                }
            }
        }
    }
}

bool ArmConvLayer1x1::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                 const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    return param->kernels[0] == 1 && param->kernels[1] == 1 && param->group == 1 &&
           dims_output[1] % ARM_SGEMM_TILE_N == 0;
}

ArmConvLayer1x1::~ArmConvLayer1x1() {}

Status ArmConvLayer1x1::allocateBufferWeight(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (!buffer_weight_.GetBytesSize()) {
        RETURN_ON_NEQ(ArmConvLayerCommon::allocateBufferWeight(inputs, outputs), TNN_OK);
        if (ARM_SGEMM_TILE_N == 8) {
            ConvertWeightsC4ToC8(buffer_weight_.force_to<float *>(), inputs[0]->GetBlobDesc().dims[1],
                                 outputs[0]->GetBlobDesc().dims[1]);
        }
    }
    return TNN_OK;
}

Status ArmConvLayer1x1::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        return Exec<float>(inputs, outputs);
    } else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
        return Exec<bfp16_t>(inputs, outputs);
    }
    return TNNERR_LAYER_ERR;
}

template <typename T>
Status ArmConvLayer1x1::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    DataType data_type = output->GetBlobDesc().data_type;
    int data_byte_size = DataTypeUtils::GetBytesSize(data_type);

    const int batch = dims_output[0];
    auto ic4        = UP_DIV(dims_input[1], 4);
    auto oc4        = UP_DIV(dims_output[1], 4);

    int src_z_step = k_param_->iw * k_param_->ih * 4;
    int dst_z_step = k_param_->ow * k_param_->oh * 4;
    int plane_num  = k_param_->ow * k_param_->oh;

    T *src_origin = reinterpret_cast<T *>(GetBlobHandlePtr(input->GetHandle()));
    T *dst_origin = reinterpret_cast<T *>(GetBlobHandlePtr(output->GetHandle()));

    /*
    get a_block & b_block based on l2 cache size(512K most of the time)
    */
    int max_num_threads = OMP_MAX_THREADS_NUM_;
    int threadbuf_num   = plane_num > oc4 * 4 ? max_num_threads : 1;
    int a_block, b_block;
    set_block_size(a_block, b_block, 512 * 1024 / data_byte_size, plane_num, oc4 * 4, ic4 * 4, data_byte_size);
    int work_space_size = a_block * ic4 * 4 * sizeof(T) * threadbuf_num;
    auto work_space     = reinterpret_cast<T *>(context_->GetSharedWorkSpace(work_space_size + NEON_KERNEL_EXTRA_LOAD));

    /*
    pack inputs when pads or strides are not equal to one
    */
    if ((k_param_->ih != k_param_->oh) || (k_param_->iw != k_param_->ow)) {
        work_space_size += batch * ic4 * 4 * dims_output[2] * dims_output[3] * data_byte_size;
        auto tmp_dst = reinterpret_cast<T *>(context_->GetSharedWorkSpace(work_space_size + NEON_KERNEL_EXTRA_LOAD));
        work_space   = tmp_dst + batch * ic4 * 4 * dims_output[2] * dims_output[3];

        PackLine(tmp_dst, src_origin, k_param_->ih, k_param_->iw, k_param_->oh, k_param_->ow, k_param_->ic_r4 * batch,
                 conv_param->pads[2], conv_param->pads[0], conv_param->strides[1], conv_param->strides[0]);
        src_origin = tmp_dst;
    }

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto input_ptr  = src_origin + batch_idx * k_param_->ow * k_param_->oh * ROUND_UP(dims_input[1], 4);
        auto output_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * ROUND_UP(dims_output[1], 4);
        auto bias_ptr   = reinterpret_cast<float *>(k_param_->bias);

        /*
        call different sgemm func based on input and weight size
        */
        if (plane_num > oc4 * 4) {
            sgemm_repack_lhs(output_ptr, input_ptr, buffer_weight_.force_to<float *>(), ic4, oc4, plane_num, dst_z_step,
                             a_block, b_block, work_space, bias_ptr, conv_param->activation_type,
                             context_->GetPrecision() != PRECISION_HIGH);
        } else {
            sgemm_repack_rhs(output_ptr, input_ptr, buffer_weight_.force_to<float *>(), ic4, oc4, plane_num, dst_z_step,
                             a_block, b_block, work_space, bias_ptr, conv_param->activation_type,
                             context_->GetPrecision() != PRECISION_HIGH);
        }
    }

    return TNN_OK;
}

}  // namespace TNN_NS
