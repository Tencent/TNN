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

#include <cmath>

#include "tnn/device/x86/acc/x86_inner_product_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/network/openvino/layer_builder/compute/gemmbench_dnnl.h"

namespace TNN_NS {

Status X86InnerProductLayerAcc::Init(Context* context, LayerParam* param, LayerResource* resource,
                                     const std::vector<Blob*> &inputs, const std::vector<Blob*> &outputs) {
    X86LayerAcc::Init(context, param, resource, inputs, outputs);

    int m = inputs[0]->GetBlobDesc().dims[0];
    int n = outputs[0]->GetBlobDesc().dims[1];
    int k = inputs[0]->GetBlobDesc().dims[1];
    acc_index = static_cast<AccIndex>(GemmScan(m, n, k));

    return Reshape(inputs, outputs);
}

Status X86InnerProductLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto resource = dynamic_cast<InnerProductLayerResource*>(resource_);
    if (!resource) {
        return Status(TNNERR_MODEL_ERR, "Error: InnerProductLayerResource is nil");
    }

    auto paramlist = dynamic_cast<InnerProductLayerParam*>(param_);
    auto input_blob         = inputs[0];
    auto output_blob        = outputs[0];
    float *input_data       = static_cast<float *>(input_blob->GetHandle().base);
    float *output_data      = static_cast<float *>(output_blob->GetHandle().base);
    int channel             = input_blob->GetBlobDesc().dims[1];
    int count               = DimsVectorUtils::Count(input_blob->GetBlobDesc().dims);
    RawBuffer weight_handle = resource->weight_handle;
    auto *weight_data       = resource->weight_handle.force_to<float *>();
    RawBuffer scale_handle  = resource->scale_handle;
    auto *scale_data        = resource->scale_handle.force_to<float *>();
    bool share_channel      = scale_handle.GetBytesSize() == DataTypeUtils::GetBytesSize(scale_handle.GetDataType());
    auto *bias_data         = resource->bias_handle.force_to<float *>();

    static engine eng(engine::kind::cpu, 0);
    static stream stm(eng);

    int input_d0 = input_blob->GetBlobDesc().dims[0];
    int input_d1 = input_blob->GetBlobDesc().dims[1];
    int input_d2 = input_blob->GetBlobDesc().dims[2];
    int input_d3 = input_blob->GetBlobDesc().dims[3];

    int output_d0 = output_blob->GetBlobDesc().dims[0];
    int output_d1 = output_blob->GetBlobDesc().dims[1];
    int output_d2 = output_blob->GetBlobDesc().dims[2];
    int output_d3 = output_blob->GetBlobDesc().dims[3];

    int m = input_d0;
    int n = output_d1;
    int k = input_d1;

    int lda = k;
    int ldb = n;
    int ldc = n;

    float alpha = 1.0f;
    float beta = 0.f;

    switch (acc_index) {
        case AccIndex_Sgemm:
            dnnl_sgemm('N', 'N', m, n, k, alpha, input_data, lda, weight_data, ldb, beta, output_data, ldc);
            break;
        case AccIndex_InnerProduct:
            InnerProduct(eng, stm, input_data, weight_data, bias_data, output_data, m, n, k);
            break;
        case AccIndex_MatMul:
            MatMul(eng, stm, input_data, weight_data, bias_data, output_data, m, n, k);
            break;
        default:
            dnnl_sgemm('N', 'N', m, n, k, alpha, input_data, lda, weight_data, ldb, beta, output_data, ldc);
    }

    return TNN_OK;
}

REGISTER_X86_ACC(InnerProduct, LAYER_INNER_PRODUCT);

}
