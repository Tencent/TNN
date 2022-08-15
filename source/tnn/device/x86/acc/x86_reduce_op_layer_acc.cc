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

#include "x86_reduce_op_layer_acc.h"
#include "tnn/device/x86/acc/compute/x86_compute.h"

namespace TNN_NS {

X86ReduceOpLayerAcc::~X86ReduceOpLayerAcc() {}

Status X86CalculateReduceDims(Blob *input_blob, ReduceLayerParam *layer_param,
                           std::vector<std::tuple<int, int, int>> &reduce_dims) {
    auto input_dims = input_blob->GetBlobDesc().dims;
    auto axes       = layer_param->axis;
    std::sort(axes.begin(), axes.end());
    reduce_dims.clear();
    for (const auto &axis : axes) {
        int outer_count   = DimsVectorUtils::Count(input_dims, 0, axis);
        int reducer_count = input_dims[axis];
        int inner_count   = DimsVectorUtils::Count(input_dims, axis + 1);
        inner_count       = inner_count == 0 ? 1 : inner_count;
        reduce_dims.emplace_back(std::make_tuple(outer_count, reducer_count, inner_count));
        input_dims[axis] = 1;
    }
    return TNN_OK;
}

Status X86ReduceOpLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    
    auto input_blob = inputs[0];
    auto output_blob = outputs[0];

    auto input_dim = input_blob->GetBlobDesc().dims;
    auto output_dim = output_blob->GetBlobDesc().dims;

    auto layer_param = dynamic_cast<ReduceLayerParam*>(param_);

    size_t workspace_size = DimsVectorUtils::Count(input_dim) * 2 * sizeof(float);
    float *workspace = reinterpret_cast<float *>(context_->GetSharedWorkSpace(workspace_size));

    std::vector<std::tuple<int, int, int>> reduce_dims;
    X86CalculateReduceDims(input_blob, layer_param, reduce_dims);

    X86_REDUCE_CALCULATE(handle_ptr<float *>(input_blob->GetHandle()),
                         handle_ptr<float *>(output_blob->GetHandle()),
                         workspace, reduce_dims, input_dim, output_dim, op_type_);

    return TNN_OK;
}

}