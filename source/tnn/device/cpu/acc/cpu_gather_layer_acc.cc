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

#include "cpu_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"
namespace TNN_NS {

DECLARE_CPU_ACC(Gather, LAYER_GATHER);

Status CpuGatherLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuGatherLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_blob   = inputs[0];
    auto &input_dims  = input_blob->GetBlobDesc().dims;
    auto output_blob  = outputs[0];
    auto &output_dims = output_blob->GetBlobDesc().dims;
    auto cur_param    = dynamic_cast<GatherLayerParam *>(param_);
    auto cur_resource = dynamic_cast<GatherLayerResource *>(resource_);
    ASSERT(cur_param->indices_in_resource == true);
    ASSERT(cur_param->data_in_resource == false);
    int axis                 = cur_param->axis;
    const auto &indices_dims = cur_resource->indices.GetBufferDims();
    int indices_count        = DimsVectorUtils::Count(indices_dims);
    auto indices_data        = new int[indices_count];
    auto indices_raw_data    = cur_resource->indices.force_to<int32_t *>();
    for (int i = 0; i < indices_count; ++i) {
        indices_data[i] = indices_raw_data[i];
    }
    if (input_blob->GetBlobDesc().data_type == DATA_TYPE_INT32) {
        auto input_data_ptr  = (int32_t *)input_blob->GetHandle().base;
        auto output_data_ptr = (int32_t *)output_blob->GetHandle().base;
        int steps            = DimsVectorUtils::Count(indices_dims, axis, indices_dims.size());
        for (int i = 0; i < indices_count; ++i) {
            output_data_ptr += i * steps;
            input_data_ptr += indices_data[i] * steps;
            memcpy(output_data_ptr, input_data_ptr, steps * sizeof(int32_t));
        }
    }
    delete[] indices_data;
    return TNN_OK;
}

REGISTER_CPU_ACC(Gather, LAYER_GATHER);
}  // namespace TNN_NS
