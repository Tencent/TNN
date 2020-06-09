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

#include "tnn/device/arm/acc/arm_nchw_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

DECLARE_ARM_NCHW_ACC(Permute, LAYER_PERMUTE);

Status ArmPermuteLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto permute_param = dynamic_cast<PermuteLayerParam *>(param_);
    CHECK_PARAM_NULL(permute_param);

    AllocConvertBuffer(inputs, outputs);

    UnPackInputs(inputs);
    auto input_dims     = nchw_blob_in[0]->GetBlobDesc().dims;
    auto output_dims    = nchw_blob_out[0]->GetBlobDesc().dims;

    std::vector<int> input_step;
    std::vector<int> output_step;
    int num_dims = int(input_dims.size());
    int output_count = DimsVectorUtils::Count(output_dims);
    for (int i = 0; i < input_dims.size(); ++i) {
        input_step.push_back(DimsVectorUtils::Count(input_dims, i + 1));
        output_step.push_back(DimsVectorUtils::Count(output_dims, i + 1));
    }

    if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = reinterpret_cast<float *>(GetBlobHandlePtr(nchw_blob_in[0]->GetHandle()));
        float *output_data = reinterpret_cast<float *>(GetBlobHandlePtr(nchw_blob_out[0]->GetHandle()));
        NaivePermute<float>(output_count, input_data, permute_param->orders, input_step, output_step, num_dims,
                            output_data);
    }
    PackOutputs(outputs);
    return TNN_OK;
}

REGISTER_ARM_ACC(Permute, LAYER_PERMUTE);

}  // namespace TNN_NS
