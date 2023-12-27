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

#include "tnn/device/arm/acc/gradient/arm_gradient_layer_acc.h"

namespace TNN_NS {

DECLARE_ARM_GRAD_OP(InnerProduct, LAYER_INNER_PRODUCT);

// weigt: oc * ic
// input: batch * ic
// output: batch * oc
// bias: oc
// weight_grad: oc * ic     <- matmul(output_grad^T, input)
// input_grad:  batch * ic  <- matmul(output_grad, weight)
// bias_grad:   1 * oc      <- sigma_batch(output_grad)
static void ExecWeightGrad(int batch, int oc, int ic, float *weight_grad, float *output_grad, float *input) {
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
            w_ptr += ic >> 2 << 2;
            for (int j = 0; j < remain; ++j) {
                w_ptr[j] += i_ptr[j + (ic >> 2 << 2)] * o_ptr[i];
            }
        }
    }
}

static void ExecInputGrad(int batch, int oc, int ic, float *input_grad, float *output_grad, float *weight,
                          ArmContext *context) {
    size_t pack_a_size    = batch * oc * sizeof(float) + NEON_KERNEL_EXTRA_LOAD;
    size_t pack_b_size    = oc * ROUND_UP(ic, 8) * sizeof(float) + NEON_KERNEL_EXTRA_LOAD;
    size_t workspace_size = pack_a_size + pack_b_size;
    char *workspace       = reinterpret_cast<char *>(context->GetSharedWorkSpace(workspace_size));
    float *pack_a_ptr     = reinterpret_cast<float *>(workspace);
    float *pack_b_ptr     = reinterpret_cast<float *>(workspace + pack_a_size);
    GemmFloatPackAB(batch, ic, oc, output_grad, pack_a_ptr, oc, weight, pack_b_ptr, ic, input_grad, ic);
}

static void ExecBiasGrad(int batch, int oc, float *bias_grad, float *output_grad) {
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
        dst_ptr += oc >> 2 << 2;
        src_ptr += oc >> 2 << 2;
        for (int n = 0; n < remain; ++n) {
            dst_ptr[n] += src_ptr[n];
        }
    }
}

Status ArmInnerProductGradOp::OnGrad(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                                        LayerResource *resource, GradientParam *grad_param, Context *context,
                                        const GradOpInfo &grad_info) {
    ON_GRAD_PREPARATION_IOR(1, 1, 2);

    auto arm_context = dynamic_cast<ArmContext *>(context);
    CHECK_PARAM_NULL(arm_context);

    auto inner_product_param = dynamic_cast<InnerProductLayerParam *>(grad_param->forward_param);
    CHECK_PARAM_NULL(inner_product_param);
    bool has_bias = inner_product_param->has_bias;

    auto inner_product_res = dynamic_cast<InnerProductLayerResource *>(resource);
    CHECK_PARAM_NULL(inner_product_res);
    auto weight = inner_product_res->weight_handle;
    auto bias   = inner_product_res->bias_handle;

    int batch = DimsFunctionUtils::GetDim(input_dims[0], 0);
    int ic    = DimsVectorUtils::Count(input_dims[0], 1);
    int oc    = DimsFunctionUtils::GetDim(output_dims[0], 1);
    if (weight.GetDataCount() != oc * ic) {
        LOGD("ArmInnerProductGradOp::OnGrad ERROR, weight data count error\n");
        return Status(TNNERR_TRAIN_ERROR, "weight data count error");
    }

    if (fw_inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        auto input_ptr       = reinterpret_cast<float *>(GetBlobHandlePtr(fw_inputs[0]->GetHandle()));
        auto weight_ptr      = weight.force_to<float *>();
        auto output_grad_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(output_grads[0]->GetHandle()));
        auto input_grad_ptr  = reinterpret_cast<float *>(GetBlobHandlePtr(input_grads[0]->GetHandle()));

        // nc4hw4 -> nchw if needed
        bool input_need_reformat =
            fw_inputs[0]->GetBlobDesc().data_format != DATA_FORMAT_NCHW && !FloatBlobCanIgnorePack(input_dims[0]);
        bool output_need_reformat =
            fw_inputs[0]->GetBlobDesc().data_format != DATA_FORMAT_NCHW && !FloatBlobCanIgnorePack(output_dims[0]);
        RawBuffer output_grad_reordered;
        if (output_need_reformat) {  // 将y_grad转为nchw
            output_grad_reordered = RawBuffer(batch * oc * sizeof(float));
            float *reordered_ptr  = output_grad_reordered.force_to<float *>();
            UnpackFloatBlob(reordered_ptr, output_grad_ptr, output_dims[0]);
            output_grad_ptr = reordered_ptr;
        }
        RawBuffer input_grad_reordered;
        if (input_need_reformat) {  // 将使用一个新的nchw的存储来存x_grad
            input_grad_reordered = RawBuffer(batch * ic * sizeof(float));
            float *reordered_ptr = input_grad_reordered.force_to<float *>();
            input_grad_ptr       = reordered_ptr;
        }

        ExecInputGrad(batch, oc, ic, input_grad_ptr, output_grad_ptr, weight_ptr, arm_context);

        if (input_need_reformat) {  // 将nchw的x_grad转为NC4HW4
            PackFloatBlob(reinterpret_cast<float *>(GetBlobHandlePtr(input_grads[0]->GetHandle())), input_grad_ptr,
                          input_dims[0]);
        }

        if (resource_need_train) {
            resource_grads[0]->GetBlobDesc().data_format = DATA_FORMAT_NCHW;
            resource_grads[1]->GetBlobDesc().data_format = DATA_FORMAT_NCHW;
            auto weight_grad_ptr = resource_grads[0]->GetHandle().force_to<float *>();
            auto bias_grad_ptr   = resource_grads[1]->GetHandle().force_to<float *>();

            RawBuffer input_reordered;
            if (input_need_reformat) {
                input_reordered      = RawBuffer(batch * ic * sizeof(float));
                float *reordered_ptr = input_reordered.force_to<float *>();
                UnpackFloatBlob(reordered_ptr, input_ptr, input_dims[0]);
                input_ptr = reordered_ptr;
            }
            ExecWeightGrad(batch, oc, ic, weight_grad_ptr, output_grad_ptr, input_ptr);

            if (has_bias && bias.GetDataCount() > 0) {
                ExecBiasGrad(batch, oc, bias_grad_ptr, output_grad_ptr);
            }
        }
    } else {
        LOGE("ArmInnerProductGradOp::OnGrad, dtype not supported\n");
        return Status(TNNERR_TRAIN_ERROR, "dtype not supported");
    }

    return TNN_OK;
}

REGISTER_ARM_GRAD_OP(InnerProduct, LAYER_INNER_PRODUCT)
REGISTER_ARM_GRAD_LAYOUT(LAYER_INNER_PRODUCT, DATA_FORMAT_NCHW)
REGISTER_ARM_GRAD_LAYOUT(LAYER_INNER_PRODUCT, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS

