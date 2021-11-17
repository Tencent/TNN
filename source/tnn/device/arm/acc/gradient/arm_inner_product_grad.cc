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

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/train/gradient/layer_grad.h"
#include "tnn/utils/dims_function_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

DECLARE_ARM_LAYER_GRAD(InnerProduct, LAYER_INNER_PRODUCT);

// weigt: oc * ic
// input: batch * ic
// output: batch * oc
// bias: oc
// weight_grad: oc * ic     <- matmul(output_grad^T, input)
// input_grad:  batch * ic  <- matmul(output_grad, weight)
// bias_grad:   1 * oc      <- sigma_batch(output_grad)
template <int acc_weight>
static void ExecWeightGrad(int batch, int oc, int ic, float *weight_grad, float *output_grad, float *input) {
    if (!acc_weight) {
        memset(weight_grad, 0, oc * ic * sizeof(float));
    }

    for (int b = 0; b < batch; ++b) {
        auto o_ptr = output_grad + oc * b;
        auto i_ptr = input + ic * b;
        OMP_PARALLEL_FOR_
        for (int i = 0; i < oc; ++i) {
            auto w_ptr = weight_grad + ic * i;
            Float4 ug(o_ptr[i]);
            for (int j = 0; j < ic - 3; j += 4) {
                Float4 x0 = Float4::load(i_ptr + j);
                Float4::save(w_ptr + j, ug * x0 + Float4::load(w_ptr + j));
            }
            int remain = ic % 4;
            w_ptr += ic << 2 >> 2;
            for (int j = 0; j < remain; ++j) {
                w_ptr[j] += i_ptr[j + (ic << 2 >> 2)] * o_ptr[i];
            }
        }
    }
}

template <int acc_input>
static void ExecInputGrad(int batch, int oc, int ic, float *input_grad, float *output_grad, float *weight,
                          ArmContext *context) {
    size_t pack_a_size    = batch * oc * sizeof(float) + NEON_KERNEL_EXTRA_LOAD;
    size_t pack_b_size    = oc * ROUND_UP(ic, 8) * sizeof(float) + NEON_KERNEL_EXTRA_LOAD;
    size_t workspace_size = pack_a_size + pack_b_size;
    char *workspace       = reinterpret_cast<char *>(context->GetSharedWorkSpace(workspace_size));
    float *pack_a_ptr     = reinterpret_cast<float *>(workspace);
    float *pack_b_ptr     = reinterpret_cast<float *>(workspace + pack_a_size);

    if (!acc_input) {
        memset(input_grad, 0, batch * ic * sizeof(float));
    }

    GemmFloatPackAB(batch, ic, oc, output_grad, pack_a_ptr, oc, weight, pack_b_ptr, ic, input_grad, ic);
}

template <int acc_bias>
static void ExecBiasGrad(int batch, int oc, float *bias_grad, float *output_grad) {
    if (batch == 1 && !acc_bias) {
        memcpy(bias_grad, output_grad, batch * oc * sizeof(float));
        return;
    }
    if (!acc_bias) {
        memset(bias_grad, 0, batch * oc * sizeof(float));
    }
    Float4 ug;
    for (int b = 0; b < batch; ++b) {
        float *dst_ptr = bias_grad;
        float *src_ptr = output_grad + b * oc;
        OMP_PARALLEL_FOR_
        for (int n = 0; n < oc - 3; n += 4) {
            ug = Float4::load(src_ptr + n);
            Float4::save(dst_ptr + n, ug + Float4::load(dst_ptr + n));
        }
        int remain = oc % 4;
        dst_ptr += oc << 2 >> 2;
        src_ptr += oc << 2 >> 2;
        for (int n = 0; n < remain; ++n) {
            dst_ptr[n] += src_ptr[n];
        }
    }
}

Status ArmInnerProductLayerGrad::OnGrad(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                                        LayerResource *resource, LayerParam *param, Context *context,
                                        LayerGradInfo *grad_info) {
    CHECK_PARAM_NULL(grad_info);
    if (grad_info->accumulate_blob_grad.size() < 1) {
        LOGD("ArmInnerProductLayerGrad::OnGrad, accumulate_blob_grad error\n");
        return Status(TNNERR_LAYER_ERR, "accumulate_blob_grad size error");
    }
    bool accumulate_blob_grad0 = grad_info->accumulate_blob_grad[0];
    if (grad_info->accumulate_resource_grad.size() < 2) {
        LOGD("ArmInnerProductLayerGrad::OnGrad, accumulate_resource_grad error\n");
        return Status(TNNERR_LAYER_ERR, "accumulate_resource_grad size error");
    }
    bool accumulate_resource_grad0 = grad_info->accumulate_resource_grad[0];
    bool accumulate_resource_grad1 = grad_info->accumulate_resource_grad[1];

    auto fc_param = dynamic_cast<InnerProductLayerParam *>(param);
    CHECK_PARAM_NULL(param);

    auto arm_context = dynamic_cast<ArmContext *>(context);
    CHECK_PARAM_NULL(arm_context);

    if (inputs.size() != 3 || outputs.size() != 3) {
        return Status(TNNERR_LAYER_ERR, "input size or output size not match in ArmInnerProductLayerGrad");
    }

    auto fw_input      = inputs[0];
    auto fw_output     = inputs[1];
    auto upstream_grad = inputs[2];
    auto input_grad    = outputs[0];
    auto weight_grad   = outputs[1];
    auto bias_grad     = outputs[2];

    auto input_dims  = fw_input->GetBlobDesc().dims;
    auto output_dims = fw_output->GetBlobDesc().dims;
    auto grad_dims   = input_grad->GetBlobDesc().dims;

    if (!DimsVectorUtils::Equal(input_dims, grad_dims)) {
        return Status(TNNERR_LAYER_ERR, "ArmInnerProductLayerGrad input dims and grad dims not match");
    }

    int batch   = DimsFunctionUtils::GetDim(input_dims, 0);
    int channel = DimsFunctionUtils::GetDim(input_dims, 1);
    int hw      = DimsVectorUtils::Count(input_dims, 2);
    int ic      = channel * hw;
    int oc      = DimsFunctionUtils::GetDim(output_dims, 1);

    auto fc_res = dynamic_cast<InnerProductLayerResource *>(resource);
    CHECK_PARAM_NULL(fc_res);
    auto weight = fc_res->weight_handle;
    ASSERT(weight.GetDataCount() == oc * ic);

    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        auto upstream_grad_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(upstream_grad->GetHandle()));

        // nc4hw4 -> nchw if needed
        RawBuffer upstream_grad_reordered;
        if (!FloatBlobCanIgnorePack(oc, 1)) {
            upstream_grad_reordered = RawBuffer(batch * oc);
            float *reordered_ptr    = upstream_grad_reordered.force_to<float *>();
            UnpackFloatBlob(reordered_ptr, upstream_grad_ptr, batch, oc, 1);
            upstream_grad_ptr = reordered_ptr;
        }

        auto input_grad_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(input_grad->GetHandle()));
        RawBuffer input_grad_reordered;
        if (!FloatBlobCanIgnorePack(channel, hw)) {
            input_grad_reordered = RawBuffer(batch * channel * hw);
            float *reordered_ptr = input_grad_reordered.force_to<float *>();
            input_grad_ptr       = reordered_ptr;
        }

        auto weight_ptr = weight.force_to<float *>();

        if (accumulate_blob_grad0) {
            ExecInputGrad<1>(batch, oc, ic, input_grad_ptr, upstream_grad_ptr, weight_ptr, arm_context);
        } else {
            ExecInputGrad<0>(batch, oc, ic, input_grad_ptr, upstream_grad_ptr, weight_ptr, arm_context);
        }

        if (!FloatBlobCanIgnorePack(channel, hw)) {
            PackFloatBlob(reinterpret_cast<float *>(GetBlobHandlePtr(input_grad->GetHandle())), input_grad_ptr, batch,
                          channel, hw);
        }

        auto input_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(fw_input->GetHandle()));
        RawBuffer input_reordered;
        if (!FloatBlobCanIgnorePack(channel, hw)) {
            input_reordered      = RawBuffer(batch * channel * hw);
            float *reordered_ptr = input_reordered.force_to<float *>();
            input_ptr            = reordered_ptr;
        }

        auto weight_grad_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(weight_grad->GetHandle()));
        if (accumulate_resource_grad0) {
            ExecWeightGrad<1>(batch, oc, ic, weight_grad_ptr, upstream_grad_ptr, input_ptr);
        } else {
            ExecWeightGrad<0>(batch, oc, ic, weight_grad_ptr, upstream_grad_ptr, input_ptr);
        }

        if (fc_param->has_bias) {
            auto bias_grad_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(bias_grad->GetHandle()));
            if (accumulate_resource_grad1) {
                ExecBiasGrad<1>(batch, oc, bias_grad_ptr, upstream_grad_ptr);
            } else {
                ExecBiasGrad<0>(batch, oc, bias_grad_ptr, upstream_grad_ptr);
            }
        }
    } else {
        LOGE("ArmInnerProductLayerGrad::OnGrad, dtype not supported\n");
        return Status(TNNERR_LAYER_ERR, "dtype not supported");
    }

    return TNN_OK;
}

REGISTER_ARM_LAYER_GRAD(InnerProduct, LAYER_INNER_PRODUCT)
REGISTER_ARM_GRAD_LAYOUT(LAYER_INNER_PRODUCT, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
