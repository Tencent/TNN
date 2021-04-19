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

#include <tnn/utils/data_type_utils.h>

#include "tnn/device/cpu/acc/compute/compute_elewise.h"
#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

DECLARE_CPU_ACC(Einsum, LAYER_EINSUM);

inline int count(std::vector<int> dimes, int start_axis) {
    const int end_axis = int(dimes.size());
    ASSERT(start_axis <= end_axis);
    ASSERT(start_axis >= 0);
    ASSERT(end_axis >= 0);
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
        count *= dimes[i];
    }
    return count;
};

std::shared_ptr<Blob> Permute(Blob *input_blob, const std::vector<int> &orders) {
    auto output_blob_ptr = std::make_shared<Blob>(input_blob->GetBlobDesc(), true);
    auto *output_blob    = output_blob_ptr.get();
    auto input_dims      = input_blob->GetBlobDesc().dims;
    auto output_dims     = input_blob->GetBlobDesc().dims;
    const int dims_size  = input_dims.size();
    for (int i = 0; i < dims_size; i++) {
        output_dims[i] = input_dims[orders[i]];
    }
    output_blob->GetBlobDesc().dims = output_dims;

    std::vector<int> input_step;
    std::vector<int> output_step;
    int num_dims = int(input_dims.size());
    ASSERT(input_dims.size() == output_dims.size());
    for (int i = 0; i < input_dims.size(); ++i) {
        input_step.push_back(count(input_dims, i + 1));
        output_step.push_back(count(output_dims, i + 1));
    }

    float *input_data      = static_cast<float *>(input_blob->GetHandle().base);
    float *output_data     = static_cast<float *>(output_blob->GetHandle().base);
    const int output_count = DimsVectorUtils::Count(output_dims);
    NaivePermute<float>(output_count, output_dims, input_data, orders, input_step, output_step, num_dims, output_data);

    return output_blob_ptr;
}

void Squeeze(Blob *input_blob, const int axis) {
    auto output_dims = input_blob->GetBlobDesc().dims;
    output_dims.erase(output_dims.begin() + axis);
    input_blob->GetBlobDesc().dims = output_dims;
}

std::shared_ptr<Blob> Sum(Blob *input_blob, const int axis) {
    const auto input_desc = input_blob->GetBlobDesc();
    auto output_desc      = input_desc;
    auto output_dims      = output_desc.dims;
    output_dims.erase(output_dims.begin() + axis);
    output_desc.dims     = output_dims;
    auto output_blob_ptr = std::make_shared<Blob>(output_desc, true);
    auto *output_blob    = output_blob_ptr.get();

    auto input_dims   = input_blob->GetBlobDesc().dims;
    int outer_count   = DimsVectorUtils::Count(input_dims, 0, axis);
    int reducer_count = input_dims[axis];
    int inner_count   = DimsVectorUtils::Count(input_dims, axis + 1);
    inner_count       = inner_count == 0 ? 1 : inner_count;
    input_dims[axis]  = 1;

    float *input_data      = static_cast<float *>(input_blob->GetHandle().base);
    float *output_data     = static_cast<float *>(output_blob->GetHandle().base);
    const int output_count = DimsVectorUtils::Count(output_dims);
    memset(output_data, 0, output_count * sizeof(0));

    for (int oc = 0; oc < outer_count; oc++) {
        for (int c = 0; c < reducer_count; c++) {
            for (int ic = 0; ic < inner_count; ic++) {
                output_data[ic] += input_data[ic];
            }
            input_data += inner_count;
        }
        output_data += inner_count;
    }

    return output_blob_ptr;
}

std::shared_ptr<Blob> Mul(Blob *a, Blob *b) {
    std::vector<void *> input_ptrs       = {a->GetHandle().base, b->GetHandle().base};
    std::vector<DimsVector> input_shapes = {a->GetBlobDesc().dims, b->GetBlobDesc().dims};
    auto output_dims                     = DimsVectorUtils::Max(a->GetBlobDesc().dims, b->GetBlobDesc().dims);
    auto output_desc                     = a->GetBlobDesc();
    output_desc.dims                     = output_dims;
    auto output_ptr                      = std::make_shared<Blob>(output_desc, true);
    auto *output                         = output_ptr.get();
    CPU_MUL(input_ptrs, input_shapes, output->GetHandle().base, output->GetBlobDesc().dims);

    return output_ptr;
}

void Flatten(Blob *input_blob) {
    const int output_dims_size     = DimsVectorUtils::Count(input_blob->GetBlobDesc().dims);
    input_blob->GetBlobDesc().dims = {output_dims_size};
}

std::shared_ptr<Blob> Dot(Blob *a, Blob *b) {
    auto output_blob_desc = a->GetBlobDesc();
    output_blob_desc.dims = {1};
    auto output_blob_ptr  = std::make_shared<Blob>(output_blob_desc, true);
    auto *output_blob     = output_blob_ptr.get();
    const int data_size   = a->GetBlobDesc().dims[0];
    float *a_data         = static_cast<float *>(a->GetHandle().base);
    float *b_data         = static_cast<float *>(b->GetHandle().base);
    float *output_data    = static_cast<float *>(output_blob->GetHandle().base);
    float sum             = 0;
    for (int i = 0; i < data_size; i++) {
        sum += a_data[i] * b_data[i];
    }
    output_data[0] = sum;

    return output_blob_ptr;
}

Status CpuEinsumLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuEinsumLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<EinsumLayerParam *>(param_);
    if (!param) {
        return Status(TNNERR_MODEL_ERR, "Error: EinsumLayerParam is nil");
    }

    std::vector<std::shared_ptr<Blob>> permuted_operands;
    const int num_ops = inputs.size();
    for (int i = 0; i < num_ops; i++) {
        auto operand_ptr            = std::make_shared<Blob>(inputs[i]->GetBlobDesc(), inputs[i]->GetHandle());
        auto *operand               = operand_ptr.get();
        operand->GetBlobDesc().dims = param->operand_dims[i];
        permuted_operands.push_back(Permute(operand, param->perm_shapes[i]));
    }

    int out_size     = param->out_size;
    int perm_index   = param->dim_last_op.size();
    auto dim_last_op = param->dim_last_op;
    auto result      = permuted_operands[0];

    if (param->has_zero_size_dim) {
        std::vector<int> out_shape(out_size);
        int output_shape_count = 1;
        for (int i = 0; i < out_size; i++) {
            out_shape[i] = permuted_operands[dim_last_op[i]].get()->GetBlobDesc().dims[i];
            output_shape_count *= out_shape[i];
        }
        float *output_ptr = static_cast<float *>(outputs[0]->GetHandle().base);
        memset(output_ptr, 0, sizeof(float) * output_shape_count);

        return TNN_OK;
    }

    int dim = out_size;
    for (int i = dim; i < perm_index; ++i, ++dim) {
        if (dim_last_op[i] == 0) {
            if (result.get()->GetBlobDesc().dims[dim] == 1) {
                Squeeze(result.get(), dim--);
            } else {
                result = Sum(result.get(), dim--);
            }
        }
    }

    auto operand = permuted_operands[1];
    std::vector<int> sum_dims;

    dim = out_size;
    for (int j = dim; j < perm_index; ++j, ++dim) {
        if (dim_last_op[j] < 1) {
            Squeeze(operand.get(), dim);
            --dim;
        } else if (dim_last_op[j] == 1) {
            if (result.get()->GetBlobDesc().dims[dim] == 1) {
                operand = Sum(operand.get(), dim);
                Squeeze(result.get(), dim);
                --dim;
            } else {
                sum_dims.push_back(dim);
            }
        }
    }

    if (sum_dims.empty()) {
        result = Mul(result.get(), operand.get());
    } else if (sum_dims.size() == result.get()->GetBlobDesc().dims.size()) {
        Flatten(result.get());
        Flatten(operand.get());
        result = Dot(result.get(), operand.get());
    } else {
        result = Mul(result.get(), operand.get());
        for (const auto axis : sum_dims) {
            result = Sum(result.get(), axis);
        }
    }

    const int data_count = DimsVectorUtils::Count(result.get()->GetBlobDesc().dims);
    memcpy(outputs[0]->GetHandle().base, result.get()->GetHandle().base, data_count * sizeof(float));

    return TNN_OK;
}

REGISTER_CPU_ACC(Einsum, LAYER_EINSUM);

}  // namespace TNN_NS
