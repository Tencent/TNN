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
#include "tnn/device/x86/acc/x86_inner_product_layer_acc.h"

namespace TNN_NS {

Status X86InnerProductLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                     const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto status = X86LayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(status, TNN_OK);
    RETURN_ON_NEQ(allocateBufferWeight(inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferBias(inputs, outputs), TNN_OK);

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
        int oc_rup = 8;
        if (arch_ == sse42) {
            oc_rup = 4;
        }
        const float *src = res->weight_handle.force_to<float *>();
        size_t input_stride = input_dims[1] * input_dims[2] * input_dims[3];
        size_t weight_count = ROUND_UP(output_dims[1], oc_rup) * input_stride;
        int data_byte_size = DataTypeUtils::GetBytesSize(res->weight_handle.GetDataType());

        if (res->weight_handle.GetDataType() == DATA_TYPE_FLOAT) {
            RawBuffer temp_buffer(weight_count * data_byte_size);
            float *dst = temp_buffer.force_to<float *>();

            if (arch_ == avx2) {
                PackC8(dst, src, input_stride, input_stride, input_stride, output_dims[1]);
            } else if (arch_ == sse42) {
                PackC4(dst, src, input_stride, input_stride, input_stride, output_dims[1]);
            }

            temp_buffer.SetDataType(DATA_TYPE_FLOAT);
            buffer_weight_ = temp_buffer;
        } else {
            LOGE("Error: DataType %d not support\n", res->weight_handle.GetDataType());
            return Status(TNNERR_MODEL_ERR, "conv_res DataType is not supported");
        }
    }
    return TNN_OK;
}

Status X86InnerProductLayerAcc::allocateBufferBias(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    InnerProductLayerParam *param = dynamic_cast<InnerProductLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    InnerProductLayerResource *res = dynamic_cast<InnerProductLayerResource *>(resource_);
    CHECK_PARAM_NULL(res);

    if (!buffer_bias_.GetBytesSize()) {
        auto dims_output = outputs[0]->GetBlobDesc().dims;
        int total_byte_size = dims_output[1] * DataTypeUtils::GetBytesSize(res->bias_handle.GetDataType());
        RawBuffer temp_buffer(total_byte_size);
        if (param->has_bias) {
            const int bias_handle_size    = res->bias_handle.GetBytesSize();
            const float *bias_handle_data = res->bias_handle.force_to<float *>();
            memcpy(temp_buffer.force_to<float *>(), res->bias_handle.force_to<float *>(), bias_handle_size);
        }
        buffer_bias_ = temp_buffer;
    }
    return TNN_OK;
}

Status X86InnerProductLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status X86InnerProductLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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
    auto input_dims   = inputs[0]->GetBlobDesc().dims;
    auto output_dims  = outputs[0]->GetBlobDesc().dims;

    int has_bias   = param->has_bias;
    int num_output = param->num_output;

    float *input_data  = static_cast<float*>(input_blob->GetHandle().base);
    float *output_data = static_cast<float*>(output_blob->GetHandle().base);
    float *weight_data = buffer_weight_.force_to<float *>();
    float *bias_data   = buffer_bias_.force_to<float *>();

    auto dims_input  = input_blob->GetBlobDesc().dims;
    auto dims_output = output_blob->GetBlobDesc().dims;

    auto X86SgemvFunc = X86Sgemv<Float4, 4>;
    if (arch_ == avx2) {
        X86SgemvFunc = X86Sgemv<Float8, 8>;
    }
    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        X86SgemvFunc(output_data, input_data, weight_data, bias_data, input_dims, output_dims);
    } else {
        return Status(TNNERR_MODEL_ERR, "blob type is unsupported");
    }
    return TNN_OK;
}

REGISTER_X86_ACC(InnerProduct, LAYER_INNER_PRODUCT);

}  // namespace TNN_NS
