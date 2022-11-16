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

#include "tnn/device/x86/x86_common.h"
#include "tnn/device/x86/x86_context.h"
#include "tnn/device/x86/x86_util.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/device/x86/acc/compute/x86_compute.h"
#include "tnn/device/x86/acc/compute/x86_compute_int8.h"
#include "tnn/device/x86/acc/x86_inner_product_layer_acc.h"
#include "tnn/interpreter/layer_resource_generator.h"

namespace TNN_NS {
using namespace x86;

Status X86InnerProductLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                     const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto status = X86LayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(status, TNN_OK);

    auto input_dims   = inputs[0]->GetBlobDesc().dims;
    auto output_dims  = outputs[0]->GetBlobDesc().dims;
    if (input_dims.size() < 2) {
        LOGE("Error: input dim size must >= 2, but it is %lu\n", input_dims.size());
        return Status(TNNERR_MODEL_ERR, "input dim size is not supported");
    }
    if (output_dims.size() < 2) {
        LOGE("Error: output dim size must >= 2, but it is %lu\n", output_dims.size());
        return Status(TNNERR_MODEL_ERR, "output dim size is not supported");
    }

    size_t flops = 2 * output_dims[0] * 
        DimsVectorUtils::Count(input_dims, 1) * 
        DimsVectorUtils::Count(output_dims, 1);
    size_t in_bytes = DimsVectorUtils::Count(input_dims) * sizeof(float);
    size_t out_bytes = DimsVectorUtils::Count(output_dims) * sizeof(float);
    size_t weight_bytes = DimsVectorUtils::Count(input_dims, 1) * 
                        DimsVectorUtils::Count(output_dims, 1) * sizeof(float);
    float ai = (float)flops / (float)(in_bytes + out_bytes + weight_bytes);

    impl_ = InnerProductSgemv;
    if (ai >= 2.f) {
        impl_ = InnerProductSgemm;
    }

    auto res = dynamic_cast<InnerProductLayerResource *>(resource);
    CHECK_PARAM_NULL(res);

    Status ret;
    if (res->weight_handle.GetDataType() == DATA_TYPE_HALF) {
        LayerResource *fp32_res = nullptr;
        RETURN_ON_NEQ(ConvertHalfResource(LAYER_INNER_PRODUCT, res, &fp32_res), TNN_OK);
        fc_acc_f32_resource_ = std::shared_ptr<LayerResource>(fp32_res);
        ret = X86LayerAcc::Init(context, param, fc_acc_f32_resource_.get(), inputs, outputs);
    } else {
        ret = X86LayerAcc::Init(context, param, resource, inputs, outputs);
    }

    RETURN_ON_NEQ(ret, TNN_OK);
    RETURN_ON_NEQ(allocateBufferWeight(inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferBias(inputs, outputs), TNN_OK);

    // converted weights are assumed to be packed, and can be freed now
    if (fc_acc_f32_resource_) {
        fc_acc_f32_resource_.reset();
    }

    return TNN_OK;
}

X86InnerProductLayerAcc::~X86InnerProductLayerAcc() {}

Status X86InnerProductLayerAcc::allocateBufferWeight(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    InnerProductLayerParam *param = dynamic_cast<InnerProductLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    InnerProductLayerResource *res = dynamic_cast<InnerProductLayerResource *>(resource_);
    CHECK_PARAM_NULL(res);

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    auto input_dims   = inputs[0]->GetBlobDesc().dims;
    auto output_dims  = outputs[0]->GetBlobDesc().dims;

    if (!buffer_weight_.GetBytesSize()) {
        if (res->weight_handle.GetDataType() == DATA_TYPE_FLOAT) {
            if (impl_ == InnerProductSgemv) {
                int oc_rup = 8;
                if (arch_ == sse42) {
                    oc_rup = 4;
                }
                const float *src = res->weight_handle.force_to<float *>();
                size_t input_stride = DimsVectorUtils::Count(input_dims, 1);
                size_t weight_count = ROUND_UP(output_dims[1], oc_rup) * input_stride;
                int data_byte_size = DataTypeUtils::GetBytesSize(res->weight_handle.GetDataType());

                RawBuffer temp_buffer(weight_count * data_byte_size, oc_rup * 4);
                float *dst = temp_buffer.force_to<float *>();

                if (arch_ == avx2) {
                    PackC8(dst, src, input_stride, input_stride, input_stride, output_dims[1]);
                } else if (arch_ == sse42) {
                    PackC4(dst, src, input_stride, input_stride, input_stride, output_dims[1]);
                }

                temp_buffer.SetDataType(DATA_TYPE_FLOAT);
                buffer_weight_ = temp_buffer;
            } else {
                int k_c = conv_gemm_conf_.K_c_;
                int m_block = conv_gemm_conf_.m_block_;
                int K = DimsVectorUtils::Count(input_dims, 1);
                int M = DimsVectorUtils::Count(output_dims, 1);
                size_t weight_pack_size = ROUND_UP(K, k_c) * ROUND_UP(M, m_block);
                const float *src = res->weight_handle.force_to<float *>();

                // align pointer of packed weights, since gemm use aligned load for input A
                RawBuffer temp_buffer(weight_pack_size * sizeof(float), 32);
                float *dst = temp_buffer.force_to<float *>();

                conv_pack_col_a_t(M, K, src, K, dst, conv_gemm_conf_);

                temp_buffer.SetDataType(DATA_TYPE_FLOAT);
                buffer_weight_ = temp_buffer;
            }
        } else if (res->weight_handle.GetDataType() == DATA_TYPE_INT8) {
            // trans nchw to nhwc4
            size_t oc      = output_dims[1];
            size_t oc_r4   = ROUND_UP(oc, 4);
            size_t ic      = input_dims[1];
            size_t ic_r4   = ROUND_UP(ic, 4);
            size_t hw_size = DimsVectorUtils::Count(input_dims, 2);

            int data_byte_size = DataTypeUtils::GetBytesSize(res->weight_handle.GetDataType());
            RawBuffer temp_buffer(oc_r4 * ic_r4 * hw_size * data_byte_size);
            const int8_t *weight_ptr = res->weight_handle.force_to<const int8_t*>();

            int i = 0;
            for (; i < oc; i++) {
                auto w_src_oc = weight_ptr + i * ic * hw_size;
                auto w_dst_oc = temp_buffer.force_to<int8_t *>() + i * ic_r4 * hw_size;
                for (int hw = 0; hw < hw_size; hw++) {
                    auto w_src_hw = w_src_oc + hw;
                    auto w_dst_hw = w_dst_oc + hw * ic_r4;
                    int j = 0;
                    for (; j < ic; j++) {
                        w_dst_hw[j] = w_src_hw[j * hw_size];
                    }
                    for (; j < ic_r4; j++) {
                        w_dst_hw[j] = 0;
                    }
                }
            }
            for (; i < oc_r4; i++) {
                auto w_dst_oc = temp_buffer.force_to<int8_t *>() + i * ic_r4 * hw_size;
                memset(w_dst_oc, 0, ic_r4 * hw_size * data_byte_size);
            }

            temp_buffer.SetDataType(DATA_TYPE_INT8);
            buffer_weight_ = temp_buffer;
        } else {
            LOGE("Error: DataType %d not support\n", res->weight_handle.GetDataType());
            return Status(TNNERR_MODEL_ERR, "innerproduct res DataType is not supported");
        }
    }
    return TNN_OK;
}

Status X86InnerProductLayerAcc::allocateBufferBias(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    InnerProductLayerParam *param = dynamic_cast<InnerProductLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    InnerProductLayerResource *res = dynamic_cast<InnerProductLayerResource *>(resource_);
    CHECK_PARAM_NULL(res);

    auto dims_output = outputs[0]->GetBlobDesc().dims;
    if (!buffer_bias_.GetBytesSize()) {
        // int8 bias needs oc_r4 memory space 
        int total_byte_size = ROUND_UP(dims_output[1], 4) * DataTypeUtils::GetBytesSize(res->bias_handle.GetDataType());
        RawBuffer temp_buffer(total_byte_size);
        if (param->has_bias) {
            const int bias_handle_size    = res->bias_handle.GetBytesSize();
            const float *bias_handle_data = res->bias_handle.force_to<float *>();
            memcpy(temp_buffer.force_to<float *>(), res->bias_handle.force_to<float *>(), bias_handle_size);
        }
        buffer_bias_ = temp_buffer;
    }

    // alloc scale buffer for int8 kernel
    if (!buffer_scale_.GetBytesSize()) {
        if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8) {
            auto o_scale        = reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource()->scale_handle;
            auto w_scale        = res->scale_handle;

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

Status X86InnerProductLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param    = dynamic_cast<InnerProductLayerParam *>(param_);
    if (!param) {
        return Status(TNNERR_MODEL_ERR, "Error: InnerProductLayerParam is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    auto input_dims   = inputs[0]->GetBlobDesc().dims;
    auto output_dims  = outputs[0]->GetBlobDesc().dims;

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        auto X86SgemvFunc = X86Sgemv<Float4, 4>;
        void (*X86VecAddFunc)(float*, const float*, long) = X86_VectorAdd<Float4, 4>;
        if (arch_ == avx2) {
            X86SgemvFunc = X86Sgemv<Float8, 8>;
            X86VecAddFunc = X86_VectorAdd<Float8, 8>;
        }

        float *input_data  = handle_ptr<float*>(input_blob->GetHandle());
        float *output_data = handle_ptr<float*>(output_blob->GetHandle());
        float *weight_data = buffer_weight_.force_to<float *>();
        float *bias_data   = buffer_bias_.force_to<float *>();

        if (impl_ == InnerProductSgemv) {
            X86SgemvFunc(output_data, input_data, weight_data, bias_data, input_dims, output_dims);
        } else {
            int k_c = conv_gemm_conf_.K_c_;
            int n_block = conv_gemm_conf_.n_block_;
            int K = DimsVectorUtils::Count(input_dims, 1);
            int N = input_dims[0];
            int M = DimsVectorUtils::Count(output_dims, 1);

            size_t workspace_size = k_c * ROUND_UP(N, n_block) * sizeof(float);
            float *workspace = reinterpret_cast<float *>(context_->GetSharedWorkSpace(workspace_size));

            RawBuffer fake_bias(N * sizeof(float));
            float *fake_bias_ptr = fake_bias.force_to<float *>();

            conv_sgemm_tn_col_major_prepack_a(M, N, K, weight_data, K,
                                input_data, K, output_data, M,
                                fake_bias_ptr, ActivationType_None,
                                workspace, conv_gemm_conf_);
            for (int i = 0; i < N; i++) {
                auto dst = output_data + i * M;
                X86VecAddFunc(dst, bias_data, M);
            }
        }
    } else if (output_blob->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        int8_t *input_data   = handle_ptr<int8_t *>(input_blob->GetHandle());
        int8_t *output_data  = handle_ptr<int8_t *>(output_blob->GetHandle());
        int8_t *weight_data  = buffer_weight_.force_to<int8_t *>();
        int32_t *bias_data   = buffer_bias_.force_to<int32_t *>();
        float *scale_data    = buffer_scale_.force_to<float *>();
        int ic_r4 = ROUND_UP(input_dims[1], 4);
        int oc_r4 = ROUND_UP(output_dims[1], 4);
        int hw    = DimsVectorUtils::Count(input_dims, 2);

        for (int n = 0; n < output_dims[0]; n++) {
            auto input_ptr  = input_data + n * ic_r4 * hw;
            auto output_ptr = output_data + n * oc_r4;
            X86GemvInt8(output_ptr, input_ptr, weight_data, bias_data, scale_data, ic_r4 * hw, oc_r4);
        }
    } else {
        return Status(TNNERR_MODEL_ERR, "blob type is unsupported");
    }
    return TNN_OK;
}

REGISTER_X86_ACC(InnerProduct, LAYER_INNER_PRODUCT);

}  // namespace TNN_NS
