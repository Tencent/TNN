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

#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/dims_utils.h"
#include <queue>
#include <vector>

namespace TNN_NS {

DECLARE_CPU_ACC_WITH_FUNC(TopK, LAYER_TOPK,
                          virtual Status InferRuntimeOutputShape(const std::vector<Blob *> &inputs,
                                                                 const std::vector<Blob *> &outputs););

template <typename T>
struct topk_record {
    int index;
    T value;
    bool operator<(const topk_record &rhs) const
    {
            return value < rhs.value;
    }
    bool operator>(const topk_record &rhs) const
    {
            return value > rhs.value;
    }
};

template <typename T>
bool compare_greater(topk_record<T> a, topk_record<T> b) {
    return a.value > b.value;
}

template <typename T>
bool compare_less(topk_record<T> a, topk_record<T> b) {
    return a.value < b.value;
}

template <typename T>
void CPU_TOPK(const T * input_data, T * output_data, int * output_index,
              DimsVector input_dims, int topk, int axis, int largest, int sort) {

    auto topk_heap = new std::priority_queue<topk_record<T>, 
                        std::vector<topk_record<T> >,
                        bool (*)(topk_record<T>, topk_record<T>)> (compare_greater<T>);
    if (!largest) {
        delete topk_heap;
        topk_heap = new std::priority_queue<topk_record<T>, 
                        std::vector<topk_record<T> >,
                        bool (*)(topk_record<T>, topk_record<T>)> (compare_less<T>);
    }

    int topk_dim_size = input_dims[axis];
    int inner_size    = DimsVectorUtils::Count(input_dims, axis + 1);
    int outer_size    = DimsVectorUtils::Count(input_dims, 0, axis);
    int outer_stride  = DimsVectorUtils::Count(input_dims, axis);

    for (int o = 0; o < outer_size; o++) {
        auto in_o_ptr     = input_data + o * outer_stride;
        auto ou_o_ptr     = output_data + o * topk * inner_size;
        auto ou_idx_o_ptr = output_index + o * topk * inner_size;

        for (int i = 0; i < inner_size; i++) {
            auto in_i_ptr = in_o_ptr + i;

            for (int k = 0; k < topk_dim_size; k++) {
                topk_record<T> tmp;
                tmp.index = k;
                tmp.value = in_i_ptr[k * inner_size];
                topk_heap->push(tmp);
                if (topk_heap->size() > topk) {
                    topk_heap->pop();
                }
            }

            auto ou_i_ptr     = ou_o_ptr + i;
            auto ou_idx_i_ptr = ou_idx_o_ptr + i;

            if (sort) {
                std::vector<topk_record<T> > sort_result;
                sort_result.reserve(topk_heap->size());
                while (!topk_heap->empty()) {
                    sort_result.emplace_back(topk_heap->top());
                    topk_heap->pop();
                }
                if (largest) {
                    std::sort(sort_result.begin(), sort_result.end(), std::greater<topk_record<T> >());
                } else {
                    std::sort(sort_result.begin(), sort_result.end(), std::less<topk_record<T> >());
                }

                for (int k = 0; k < topk; k++) {
                    topk_record<T> tmp = sort_result[k];
                    ou_i_ptr[k * inner_size] = tmp.value;
                    ou_idx_i_ptr[k * inner_size] = tmp.index;
                }
            } else {
                while (!topk_heap->empty()) {
                    topk_record<T> tmp   = topk_heap->top();
                    *(ou_i_ptr)     = tmp.value;
                    *(ou_idx_i_ptr) = tmp.index;
                    topk_heap->pop();
                    ou_i_ptr += inner_size;
                    ou_idx_i_ptr += inner_size;
                }
            }
        }
    }

    delete topk_heap;
}

Status CpuTopKLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuTopKLayerAcc::InferRuntimeOutputShape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto *layer_param = dynamic_cast<TopKLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);

    Status status = TNN_OK;
    auto input_dims = inputs[0]->GetBlobDesc().dims;
    if (inputs.size() >= 2) {
        if (inputs[1]->GetBlobDesc().data_type != DATA_TYPE_INT32) {
            return Status(TNNERR_PARAM_ERR, "TopK input(shape) has invalid data type");
        }

        auto dim_count = DimsVectorUtils::Count(inputs[1]->GetBlobDesc().dims);
        auto dim_data = (int *)((char *)inputs[1]->GetHandle().base + inputs[1]->GetHandle().bytes_offset);
        ASSERT(dim_count == 1);

        layer_param->k = dim_data[0];
    }

    auto k = layer_param->k;
    auto output_dims = input_dims;
    int axis = layer_param->axis;

    if (layer_param->k > 0) {
        output_dims[layer_param->axis] = std::min(layer_param->k, input_dims[layer_param->axis]);
    }

    if (outputs.size() != 2) {
        return Status(TNNERR_PARAM_ERR, "TopKLayer output blobs size != 2");
    }

    outputs[0]->GetBlobDesc().dims = output_dims;
    outputs[1]->GetBlobDesc().dims = output_dims;

    return TNN_OK;
}

Status CpuTopKLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<TopKLayerParam *>(param_);
    if (!param) {
        LOGE("Error: TopKLayerParam is nil\n");
        return Status(TNNERR_PARAM_ERR, "Error: TopKLayerParam is nil");
    }

    if (outputs.size() != 2) {
        LOGE("Error: TopKLayer must have 2 output blobs\n");
        return Status(TNNERR_PARAM_ERR, "Error: TopKLayer must have 2 output blobs");
    }

    auto input_dims = inputs[0]->GetBlobDesc().dims;
    if (param->axis >= input_dims.size()) {
        LOGE("Error: TopKLayer the axis exceeds input dims\n");
        return Status(TNNERR_PARAM_ERR, "Error: TopKLayer the axis exceeds input dims");
    }

    void *input_data = inputs[0]->GetHandle().base;
    void *output_data = outputs[0]->GetHandle().base;
    void *output_index_data = outputs[1]->GetHandle().base;

    if (param->k <= 0) {
        LOGE("Error: TopKLayer k <= 0\n");
        return Status(TNNERR_PARAM_ERR, "Error: TopKLayer k <= 0");
    }

    int topk = MIN(param->k, input_dims[param->axis]);

    auto data_type = inputs[0]->GetBlobDesc().data_type;
    if (data_type == DATA_TYPE_FLOAT) {
        CPU_TOPK<float>(static_cast<const float*>(input_data),
                 static_cast<float*>(output_data),
                 static_cast<int*>(output_index_data),
                 input_dims, topk, param->axis,
                 param->largest, param->sorted);
    } else if (data_type == DATA_TYPE_INT32) {
        CPU_TOPK<int>(static_cast<const int*>(input_data),
                 static_cast<int*>(output_data),
                 static_cast<int*>(output_index_data),
                 input_dims, topk, param->axis,
                 param->largest, param->sorted);
    } else {
        LOGE("Error: CpuTopKLayerAcc don't support data type: %d\n", data_type);
        return Status(TNNERR_MODEL_ERR, "Error: CpuTopKLayerAcc don't support data type");
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(TopK, LAYER_TOPK);

}  // namespace TNN_NS
