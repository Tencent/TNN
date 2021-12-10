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

#include "tnn/core/blob_int8.h"
#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/interpreter/layer_resource_generator.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/bfp16_utils.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

// DECLARE_CPU_ACC(InnerProduct, LAYER_INNER_PRODUCT);

class CpuInnerProductLayerAcc : public CpuLayerAcc {
public:
    virtual ~CpuInnerProductLayerAcc(){};
    Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                const std::vector<Blob *> &outputs);
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

private:
    RawBuffer buffer_scale_;
    std::shared_ptr<LayerResource> fp32_resource_ = nullptr;
};

Status CpuInnerProductLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                     const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    CPU_CONVERT_HALF_RESOURCE(LAYER_INNER_PRODUCT);
    if (runtime_model_ != RUNTIME_MODE_NORMAL) {
        return TNN_OK;
    }

    auto layer_param = dynamic_cast<InnerProductLayerParam *>(param);
    CHECK_PARAM_NULL(layer_param);
    auto layer_res = dynamic_cast<InnerProductLayerResource *>(resource_);
    CHECK_PARAM_NULL(layer_res);
    if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        if (!buffer_scale_.GetBytesSize()) {
            auto dims_output    = outputs[0]->GetBlobDesc().dims;
            int total_byte_size = dims_output[1] * sizeof(float);

            const float *w_scale = layer_res->scale_handle.force_to<float *>();
            CHECK_PARAM_NULL(w_scale);

            const float *o_scale =
                reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource()->scale_handle.force_to<float *>();
            int scale_len_w = layer_res->scale_handle.GetDataCount();
            int scale_len_o = reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource()->scale_handle.GetDataCount();
            RawBuffer temp_buffer(total_byte_size);
            float *temp_ptr = temp_buffer.force_to<float *>();

            int32_t *bias_ptr = layer_res->bias_handle.force_to<int32_t *>();
            for (int i = 0; i < dims_output[1]; i++) {
                int w_scale_idx = scale_len_w == 1 ? 0 : i;
                int o_scale_idx = scale_len_o == 1 ? 0 : i;
                if (o_scale[o_scale_idx] >= FLT_MIN)
                    temp_ptr[i] = w_scale[w_scale_idx] / o_scale[o_scale_idx];
                else
                    temp_ptr[i] = 0.0;
            }
            buffer_scale_ = temp_buffer;
        }
    }
    return TNN_OK;
}

Status CpuInnerProductLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuInnerProductLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param    = dynamic_cast<InnerProductLayerParam *>(param_);
    auto resource = dynamic_cast<InnerProductLayerResource *>(resource_);
    if (!param) {
        return Status(TNNERR_MODEL_ERR, "Error: InnerProductLayerParam is nil");
    }
    if (!resource) {
        return Status(TNNERR_MODEL_ERR, "Error: InnerProductLayerResource is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];

    int has_bias   = param->has_bias;
    int num_output = param->num_output;

    void *input_data  = input_blob->GetHandle().base;
    void *output_data = output_blob->GetHandle().base;
    void *weight_data = resource->weight_handle.force_to<void *>();
    void *bias_data   = nullptr;
    if (has_bias) {
        bias_data = resource->bias_handle.force_to<void *>();
    }

    auto dims_input  = input_blob->GetBlobDesc().dims;
    auto dims_output = output_blob->GetBlobDesc().dims;
    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        NaiveFC((float *)input_data, (float *)output_data, (float *)weight_data, (float *)bias_data, dims_input,
                dims_output);
    } else if (output_blob->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        int8_t *zero_point_w_ptr               = resource->zero_point_handle.force_to<int8_t *>();
        int zero_point_len_w                   = resource->zero_point_handle.GetDataCount();
        IntScaleResource *input_blob_resource  = reinterpret_cast<BlobInt8 *>(input_blob)->GetIntResource();
        IntScaleResource *output_blob_resource = reinterpret_cast<BlobInt8 *>(output_blob)->GetIntResource();
        int8_t *zero_point_i_ptr               = input_blob_resource->zero_point_handle.force_to<int8_t *>();
        int8_t *zero_point_o_ptr               = output_blob_resource->zero_point_handle.force_to<int8_t *>();
        int zero_point_len_i                   = input_blob_resource->zero_point_handle.GetDataCount();
        int zero_point_len_o                   = output_blob_resource->zero_point_handle.GetDataCount();
        NaiveFCBias(input_data, output_data, weight_data, buffer_scale_.force_to<float *>(), dims_output[1], bias_data,
                    zero_point_w_ptr, zero_point_i_ptr, zero_point_o_ptr, zero_point_len_w, zero_point_len_i,
                    zero_point_len_o, dims_input, dims_output);
    } else if (output_blob->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
        RawBuffer weight_bf16 = RawBuffer(resource->weight_handle.GetDataCount() * sizeof(bfp16_t));
        ConvertFromFloatToBFP16((float *)weight_data, weight_bf16.force_to<void *>(),
                                resource->weight_handle.GetDataCount());
        NaiveFC((bfp16_t *)input_data, (bfp16_t *)output_data, weight_bf16.force_to<bfp16_t *>(), (float *)bias_data,
                dims_input, dims_output);
    } else {
        return Status(TNNERR_MODEL_ERR, "blob type is unsupported");
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(InnerProduct, LAYER_INNER_PRODUCT);

}  // namespace TNN_NS
