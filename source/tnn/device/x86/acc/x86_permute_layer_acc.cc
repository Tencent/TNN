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

#include "tnn/device/x86/acc/x86_permute_layer_acc.h"

#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

X86PermuteLayerAcc::~X86PermuteLayerAcc(){};

Status X86PermuteLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto status = X86LayerAcc::Init(context, param, resource, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    return TNN_OK;
}

Status X86PermuteLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status X86PermuteLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<PermuteLayerParam *>(param_);
    if (!param) {
        return Status(TNNERR_MODEL_ERR, "Error: PermuteLayerParam is empyt");
    }
    Blob *input_blob       = inputs[0];
    Blob *output_blob      = outputs[0];
    DataType data_type     = output_blob->GetBlobDesc().data_type;
    DimsVector input_dims  = input_blob->GetBlobDesc().dims;
    DimsVector output_dims = output_blob->GetBlobDesc().dims;
    const int output_count = DimsVectorUtils::Count(output_dims);

    std::vector<int> input_step;
    std::vector<int> output_step;
    int num_dims = int(input_dims.size());
    ASSERT(input_dims.size() == output_dims.size());
    for (int i = 0; i < input_dims.size(); ++i) {
        input_step.push_back(X86PermuteLayerAcc::count(input_dims, i + 1));
        output_step.push_back(X86PermuteLayerAcc::count(output_dims, i + 1));
    }

    if (data_type != DATA_TYPE_INT8) {
        float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
        float *output_data = static_cast<float *>(output_blob->GetHandle().base);
        NaivePermute<float>(output_count, output_dims, input_data, param->orders, input_step, output_step, num_dims, output_data);
    } else {
        // DATA_TYPE_INT8
        int8_t *input_data  = static_cast<int8_t *>(input_blob->GetHandle().base);
        int8_t *output_data = static_cast<int8_t *>(output_blob->GetHandle().base);
        NaivePermute<int8_t>(output_count, output_dims, input_data, param->orders, input_step, output_step, num_dims, output_data);
    }
    return TNN_OK;
}

X86TypeLayerAccRegister<TypeLayerAccCreator<X86PermuteLayerAcc>> g_x86_permute_layer_acc_register(LAYER_PERMUTE);

}  // namespace TNN_NS
