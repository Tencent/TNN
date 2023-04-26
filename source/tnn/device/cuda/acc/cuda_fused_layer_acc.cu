// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "tnn/device/cuda/acc/cuda_fused_layer_acc.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/data_type_utils.h"


namespace TNN_NS {

Status CudaFusedLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = CudaLayerAcc::Init(context, param, resource, inputs, outputs);
    if (ret != TNN_OK) {
        return ret;
    }

    fused_param_ = dynamic_cast<FusedLayerParam *>(param_);

    if (!fused_param_) {
        LOGE("Error: fused layer param is nil\n");
        return Status(TNNERR_PARAM_ERR, "Error: fused layer param is nil");
    }

    cublas_fp32_ = std::make_shared<cublasMMWrapper>(context_->GetCublasHandle(), context_->GetCublasLtHandle());
    cublas_fp32_->setFP32GemmConfig();

    cublas_fp16_ = std::make_shared<cublasMMWrapper>(context_->GetCublasHandle(), context_->GetCublasLtHandle());
    cublas_fp16_->setFP16GemmConfig();

    if (fused_param_->type == FusionType_AddBiasResidualLayerNorm) {
        auto resource = dynamic_cast<EltwiseLayerResource*>(resource_);
        if (!resource) {
            LOGE("Error: fused layer norm resource is nil\n");
            return Status(TNNERR_LAYER_ERR, "Error: fused layer norm resource is nil");
        }

        RETURN_ON_FAIL(PrepareResource(resource->element_handle));
    } else if (fused_param_->type == FusionType_FFN) {
        auto resource = dynamic_cast<FusedLayerResource*>(resource_);
        if (!resource) {
            LOGE("Error: fused ffn resource is nil\n");
            return Status(TNNERR_LAYER_ERR, "Error: fused ffn resource is nil");
        }

        if (fused_param_->ffn_activation == ActivationType_GELU) {
            ffn_layer_fp32_ = std::make_shared<GeluFfnLayer<float>>(cublas_fp32_.get());
            ffn_layer_fp16_ = std::make_shared<GeluFfnLayer<half>>(cublas_fp16_.get());
        } else {
            LOGE("Error: fused ffn activation type not supported: %d\n", int(fused_param_->ffn_activation));
            return Status(TNNERR_LAYER_ERR, "Error: fused ffn activation type not supported");
        }

        RETURN_ON_FAIL(PrepareResource(resource->ffn_matmul_in.weight));
        RETURN_ON_FAIL(PrepareResource(resource->ffn_bias.element_handle));
        RETURN_ON_FAIL(PrepareResource(resource->ffn_matmul_out.weight));
#if 0  // Cuda Fused Attention Has 100mb + Volume
    } else if (fused_param_->type == FusionType_Attention) {
        auto resource = dynamic_cast<FusedLayerResource*>(resource_);
        if (!resource) {
            LOGE("Error: fused attention resource is nil\n");
            return Status(TNNERR_LAYER_ERR, "Error: fused attention resource is nil");
        }

        int sm_version = -1;
        RETURN_ON_FAIL(device_->GetCurrentSMVersion(sm_version));
        if (fused_param_->attention_size_per_head != 64 ||
            (sm_version!=-1 && sm_version!=70 && sm_version!=72 && sm_version!=75 && sm_version!=80 && sm_version!=86)) {
            fp16_run_fused_attention_ = false;
        }

        if (fp16_run_fused_attention_) {
            attention_fp16_ = std::make_shared<FusedAttentionLayer<half>>(
                                      fused_param_->attention_head_num,
                                      fused_param_->attention_size_per_head,
                                      fused_param_->attention_head_num * fused_param_->attention_size_per_head,
                                      fused_param_->attention_q_scaling,
                                      sm_version,
                                      cublas_fp16_.get());
        } else {
            attention_fp16_ = std::make_shared<UnfusedAttentionLayer<half>>(
                                      fused_param_->attention_head_num,
                                      fused_param_->attention_size_per_head,
                                      fused_param_->attention_head_num * fused_param_->attention_size_per_head,
                                      fused_param_->attention_q_scaling,
                                      sm_version,
                                      cublas_fp16_.get());
        }
        RETURN_ON_FAIL(PrepareFp16Resource(resource->attention_q_mm.weight));
        RETURN_ON_FAIL(PrepareFp16Resource(resource->attention_k_mm.weight));
        RETURN_ON_FAIL(PrepareFp16Resource(resource->attention_v_mm.weight));
        RETURN_ON_FAIL(PrepareFp16Resource(resource->attention_o_mm.weight));
        RETURN_ON_FAIL(PrepareFp16Resource(resource->attention_q_bias.element_handle));
        RETURN_ON_FAIL(PrepareFp16Resource(resource->attention_k_bias.element_handle));
        RETURN_ON_FAIL(PrepareFp16Resource(resource->attention_v_bias.element_handle));

        /*
        // TODO: Support Fp32
        attention_fp32_ = std::make_shared<UnfusedAttentionLayer<float>>(
                                fused_param_->attention_head_num,
                                fused_param_->attention_size_per_head,
                                fused_param_->attention_head_num * fused_param_->attention_size_per_head,
                                fused_param_->attention_q_scaling,
                                sm_version,
                                cublas_fp32_.get());
        RETURN_ON_FAIL(PrepareFp32Resource(resource->attention_q_mm.weight));
        RETURN_ON_FAIL(PrepareFp32Resource(resource->attention_k_mm.weight));
        RETURN_ON_FAIL(PrepareFp32Resource(resource->attention_v_mm.weight));
        RETURN_ON_FAIL(PrepareFp32Resource(resource->attention_o_mm.weight));
        RETURN_ON_FAIL(PrepareFp32Resource(resource->attention_q_bias.element_handle));
        RETURN_ON_FAIL(PrepareFp32Resource(resource->attention_k_bias.element_handle));
        RETURN_ON_FAIL(PrepareFp32Resource(resource->attention_v_bias.element_handle));
        */
    } else if (fused_param_->type == FusionType_Flash_Attention) {
        auto resource = dynamic_cast<FusedLayerResource*>(resource_);
        if (!resource) {
            LOGE("Error: fused attention resource is nil\n");
            return Status(TNNERR_LAYER_ERR, "Error: fused attention resource is nil");
        }

        int sm_version = -1;
        RETURN_ON_FAIL(device_->GetCurrentSMVersion(sm_version));
        if (fused_param_->attention_size_per_head != 64 ||
            (sm_version!=-1 && sm_version!=70 && sm_version!=72 && sm_version!=75 && sm_version!=80 && sm_version!=86)) {
            fp16_run_flash_attention_ = false;
        }

       if (fp16_run_fused_attention_) {
            flash_attention_fp16_ = std::make_shared<FlashAttentionLayer<half>>(
                                      sm_version);
            flash_attention_fp16_->allocateSeqlens(128); //TODO: set a better maxBatchSize
            flash_attention_fp16_->createMHARunner();
       }
#endif  // end of #if 0
    } else {
        LOGE("Error: not supported fusion type: %d\n", (int)(fused_param_->type));
        return Status(TNNERR_PARAM_ERR, "Error: not supported fusion type");
    }

    return TNN_OK;
}

Status CudaFusedLayerAcc::PrepareResource(RawBuffer &buf) {
    RETURN_ON_FAIL(PrepareFp32Resource(buf));
    RETURN_ON_FAIL(PrepareFp16Resource(buf));
    return TNN_OK;
}

Status CudaFusedLayerAcc::PrepareFp32Resource(RawBuffer &buf) {
    int data_count = buf.GetDataCount();
    auto buf_fp32 = ConvertHalfHandle(buf);
    int data_size_fp32 = data_count * DataTypeUtils::GetBytesSize(DATA_TYPE_FLOAT);
    CreateTempBuf(data_size_fp32);
    CUDA_CHECK(cudaMemcpy(tempbufs_[tempbufs_.size() - 1].ptr, buf_fp32.force_to<void*>(), data_size_fp32, cudaMemcpyHostToDevice));
    return TNN_OK;
}

Status CudaFusedLayerAcc::PrepareFp16Resource(RawBuffer &buf) {
    int data_count = buf.GetDataCount();
    auto buf_fp16 = ConvertFloatToHalf(buf);
    int data_size_fp16 = data_count * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF);
    CreateTempBuf(data_size_fp16);
    CUDA_CHECK(cudaMemcpy(tempbufs_[tempbufs_.size() - 1].ptr, buf_fp16.force_to<void*>(), data_size_fp16, cudaMemcpyHostToDevice));
    return TNN_OK;
}

CudaFusedLayerAcc::~CudaFusedLayerAcc() {}

Status CudaFusedLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaFusedLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    int rt_token_num = -1;
    auto& info_map = context_->GetExtraInfoMap();
    if (info_map.find("int_transformer_runtime_token_num") != info_map.end()) {
        rt_token_num = any_cast<int>(info_map["int_transformer_runtime_token_num"]); 
    }
    // skip when valid token number is zero
    if (rt_token_num == 0) {
        return TNN_OK;
    }

    if (fused_param_->type == FusionType_AddBiasResidualLayerNorm) {
        if (inputs.size() != 4 || outputs.size() != 1) {
            LOGE("Error: fused layer norm io size error\n");
            return Status(TNNERR_LAYER_ERR, "Error: fused layer norm io size error");
        }

        Blob *att_out_blob = inputs[0];
        Blob *res_in_blob  = inputs[1];
        Blob *scale_blob   = inputs[2];
        Blob *bias_blob    = inputs[3];
        Blob *output_blob  = outputs[0];

        float layernorm_eps = fused_param_->layer_norm_param.eps;
        auto dims = output_blob->GetBlobDesc().dims;
        int m = rt_token_num > 0 ? rt_token_num : DimsVectorUtils::Count(dims, 0, 2);
        int n = DimsVectorUtils::Count(dims, 2);

        if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
            float *att_out = static_cast<float*>(att_out_blob->GetHandle().base);
            float *res_in  = static_cast<float*>(res_in_blob->GetHandle().base);
            float *scale   = static_cast<float*>(scale_blob->GetHandle().base);
            float *bias    = static_cast<float*>(bias_blob->GetHandle().base);
            float *output  = static_cast<float*>(output_blob->GetHandle().base);
            invokeAddBiasResidualLayerNorm(output, att_out, res_in, static_cast<float*>(tempbufs_[0].ptr), scale, bias, layernorm_eps, m, n, context_->GetStream());
        } else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_HALF) {
            half *att_out = static_cast<half*>(att_out_blob->GetHandle().base);
            half *res_in  = static_cast<half*>(res_in_blob->GetHandle().base);
            half *scale   = static_cast<half*>(scale_blob->GetHandle().base);
            half *bias    = static_cast<half*>(bias_blob->GetHandle().base);
            half *output  = static_cast<half*>(output_blob->GetHandle().base);
            invokeAddBiasResidualLayerNorm(output, att_out, res_in, static_cast<half*>(tempbufs_[1].ptr), scale, bias, layernorm_eps, m, n, context_->GetStream());
        } else {
            LOGE("Error: fused layernorm not supported data type: %d\n", inputs[0]->GetBlobDesc().data_type);
            return Status(TNNERR_LAYER_ERR, "Error: fused layernorm not supported data type");
        }

    } else if (fused_param_->type == FusionType_FFN) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("Error: fused ffn io size error\n");
            return Status(TNNERR_LAYER_ERR, "Error: fused ffn io size error");
        }

        Blob *in_blob  = inputs[0];
        Blob *out_blob = outputs[0];

        auto dims = out_blob->GetBlobDesc().dims;

        if (dims.size() < 2) {
            LOGE("Error: fused ffn io dims not support\n");
            return Status(TNNERR_LAYER_ERR, "Error: fused ffn io dims support");
        }

        int token_num = rt_token_num > 0 ? rt_token_num : DimsVectorUtils::Count(dims, 0, dims.size() - 1);
        int hidden_size = DimsVectorUtils::Count(dims, dims.size() - 1);
        int inter_size = fused_param_->ffn_inter_size;

        context_->SetWorkspaceSize(inter_size * token_num * 4);

        if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
            float *input = static_cast<float*>(in_blob->GetHandle().base);
            float *output = static_cast<float*>(out_blob->GetHandle().base);
            float *ffn_matmul_in = static_cast<float*>(tempbufs_[0].ptr);
            float *ffn_bias = static_cast<float*>(tempbufs_[2].ptr);
            float *ffn_matmul_out = static_cast<float*>(tempbufs_[4].ptr);
            float *inter_buf = static_cast<float*>(context_->GetWorkspace());
            ffn_layer_fp32_->forward(output, input, ffn_matmul_in, ffn_bias, ffn_matmul_out, inter_buf, token_num, hidden_size, inter_size, context_->GetStream());
        } else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_HALF) {
            half *input = static_cast<half*>(in_blob->GetHandle().base);
            half *output = static_cast<half*>(out_blob->GetHandle().base);
            half *ffn_matmul_in = static_cast<half*>(tempbufs_[1].ptr);
            half *ffn_bias = static_cast<half*>(tempbufs_[3].ptr);
            half *ffn_matmul_out = static_cast<half*>(tempbufs_[5].ptr);
            half *inter_buf = static_cast<half*>(context_->GetWorkspace());
            ffn_layer_fp16_->forward(output, input, ffn_matmul_in, ffn_bias, ffn_matmul_out, inter_buf, token_num, hidden_size, inter_size, context_->GetStream());
        } else {
            LOGE("Error: fused ffn not supported data type: %d\n", inputs[0]->GetBlobDesc().data_type);
            return Status(TNNERR_LAYER_ERR, "Error: fused ffn not supported data type");
        }
#if 0
    } else if (fused_param_->type == FusionType_Attention) {
        if (inputs.size() < 2 || outputs.size() != 1) {
            LOGE("Error: fused attention io size error\n");
            return Status(TNNERR_LAYER_ERR, "Error: fused attention io size error");
        }

        if (inputs[0]->GetBlobDesc().data_type != DATA_TYPE_HALF) {
            LOGE("Error: fused attention not supported data type: %d\n", inputs[0]->GetBlobDesc().data_type);
            return Status(TNNERR_LAYER_ERR, "Error: fused attention not supported data type");
        }

        Blob *input_blob  = inputs[0];
        Blob *output_blob = outputs[0];

        auto dims = output_blob->GetBlobDesc().dims;

        int batch_size = DimsFunctionUtils::GetDim(dims, 0);
        int seq_len    = DimsFunctionUtils::GetDim(dims, 1);

        int token_num   = rt_token_num > 0 ? rt_token_num : DimsVectorUtils::Count(dims, 0, 2);
        int hidden_size = DimsVectorUtils::Count(dims, 2);

        int data_count = batch_size * seq_len * hidden_size;
        if (fp16_run_fused_attention_) {
            context_->SetWorkspaceSize(data_count * 7 * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF) + (2*batch_size + 1) * sizeof(int));
        } else {
            int dtype_byte_size = DataTypeUtils::GetBytesSize(inputs[0]->GetBlobDesc().data_type);
            context_->SetWorkspaceSize(data_count * 8 * dtype_byte_size + batch_size * fused_param_->attention_head_num * seq_len * seq_len * sizeof(int));
        }

        int* trt_offset = nullptr;
        if (fused_param_->dense_mode) {
            // Suppose batch = 3, max_seq_len = 12, valid_seq_len is [5,7,9] respectively,
            // Then,
            // in Fused Attention Kernel:
            //     trt_offset = [0,5,12,21], length = [batch+1]
            // in Unfused Attention Kernel:
            //     trt_offset = [0,0,0,0,0, 7,7,7,7,7,7,7, 12,12,12,12,12,12,12,12,12], length = [h_token_num], which is 5+7+9=21 here.
            if (fp16_run_fused_attention_ && inputs[0]->GetBlobDesc().data_type == DATA_TYPE_HALF) {
                trt_offset = static_cast<int*>(inputs[2]->GetHandle().base);
            } else {
                trt_offset = static_cast<int*>(inputs[3]->GetHandle().base) + 1; // The first element is [device_token_num], skip the first element.
            }
        }

        if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
            LOGE("Error: FP32 Multi-Head Attention Fused Layer Not Supported.\n");
            return Status(TNNERR_PARAM_ERR, "Error: FP32 Multi-Head Attention Fused Layer Not Supported.");
            
            /*
            float *input = static_cast<float*>(input_blob->GetHandle().base);
            float *output = static_cast<float*>(output_blob->GetHandle().base);
            float *inter_buf = static_cast<float*>(context_->GetWorkspace());
            float *q_weight = static_cast<float*>(tempbufs_[7].ptr);
            float *k_weight = static_cast<float*>(tempbufs_[8].ptr);
            float *v_weight = static_cast<float*>(tempbufs_[9].ptr);
            float *o_weight = static_cast<float*>(tempbufs_[10].ptr);
            float *q_bias = static_cast<float*>(tempbufs_[11].ptr);
            float *k_bias = static_cast<float*>(tempbufs_[12].ptr);
            float *v_bias = static_cast<float*>(tempbufs_[13].ptr);

            Blob *mask_blob = inputs[1];
            float* mask = static_cast<float*>(mask_blob->GetHandle().base);
            int ld_mask = DimsVectorUtils::Count(mask_blob->GetBlobDesc().dims, 1);

            attention_fp32_->forward(output, input, mask, trt_offset, inter_buf,
                                     q_weight, k_weight, v_weight, o_weight,
                                     q_bias, k_bias, v_bias, token_num, seq_len, batch_size, ld_mask, context_->GetStream());
            */
        } else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_HALF) {
            half *input = static_cast<half*>(input_blob->GetHandle().base);
            half *output = static_cast<half*>(output_blob->GetHandle().base);
            half *inter_buf = static_cast<half*>(context_->GetWorkspace());
            half *q_weight = static_cast<half*>(tempbufs_[0].ptr);
            half *k_weight = static_cast<half*>(tempbufs_[1].ptr);
            half *v_weight = static_cast<half*>(tempbufs_[2].ptr);
            half *o_weight = static_cast<half*>(tempbufs_[3].ptr);
            half *q_bias = static_cast<half*>(tempbufs_[4].ptr);
            half *k_bias = static_cast<half*>(tempbufs_[5].ptr);
            half *v_bias = static_cast<half*>(tempbufs_[6].ptr);

            half *mask  = nullptr;
            int ld_mask = 0;
            if ((fused_param_->has_attention_mask && !fused_param_->dense_mode) || !fp16_run_fused_attention_) {
                Blob *mask_blob = inputs[1];
                mask = static_cast<half*>(mask_blob->GetHandle().base);
                ld_mask = DimsVectorUtils::Count(mask_blob->GetBlobDesc().dims, 1);
            }

            attention_fp16_->forward(output, input, mask, trt_offset, inter_buf,
                                     q_weight, k_weight, v_weight, o_weight,
                                     q_bias, k_bias, v_bias, token_num, seq_len, batch_size, ld_mask, context_->GetStream());
        } else {
            LOGE("Error: fused attention not supported data type: %d\n", inputs[0]->GetBlobDesc().data_type);
            return Status(TNNERR_LAYER_ERR, "Error: fused attention not supported data type");
        }
    } else if (fused_param_->type == FusionType_Flash_Attention) {
        if (inputs.size() != 1 || outputs.size() != 1) {
            LOGE("Error: flash attention io size error\n");
            return Status(TNNERR_LAYER_ERR, "Error: flash attention io size error");
        }

        if (inputs[0]->GetBlobDesc().data_type != DATA_TYPE_HALF) {
            LOGE("Error: flash attention not supported data type: %d\n", inputs[0]->GetBlobDesc().data_type);
            return Status(TNNERR_LAYER_ERR, "Error: flash attention not supported data type");
        }

        Blob *input_blob  = inputs[0];
        Blob *output_blob = outputs[0];
        half *input = static_cast<half*>(input_blob->GetHandle().base);
        half *output = static_cast<half*>(output_blob->GetHandle().base);

        auto dims = output_blob->GetBlobDesc().dims;

        int batch_size = DimsFunctionUtils::GetDim(dims, 0);
        int seq_len    = DimsFunctionUtils::GetDim(dims, 1);
        int head_num   = DimsFunctionUtils::GetDim(dims, 2);
        int size_per_head   = DimsFunctionUtils::GetDim(dims, 3);
        //int token_num   = rt_token_num > 0 ? rt_token_num : DimsVectorUtils::Count(dims, 0, 2);
        int hidden_size = DimsVectorUtils::Count(dims, 2);

        int data_count = batch_size * seq_len * hidden_size;
        //workspace size is the same as trt_fused_attention
        //TODO: double check the size
        if (fp16_run_flash_attention_) {
            context_->SetWorkspaceSize(data_count * 7 * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF) + (2*batch_size + 1) * sizeof(int));
        } else {
            int dtype_byte_size = DataTypeUtils::GetBytesSize(inputs[0]->GetBlobDesc().data_type);
            context_->SetWorkspaceSize(data_count * 8 * dtype_byte_size + batch_size * head_num * seq_len * seq_len * sizeof(int));
        }
        flash_attention_fp16_->forward(input, output, batch_size, head_num, size_per_head, seq_len,
                                     context_->GetStream());

#endif  //#if 0
    } else {
        LOGE("Error: not supported fusion type: %d\n", (int)(fused_param_->type));
        return Status(TNNERR_PARAM_ERR, "Error: not supported fusion type");
    }

    return TNN_OK;
}

REGISTER_CUDA_ACC(Fused, LAYER_FUSED);

}  // namespace TNN_NS
