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

#include "tnn/device/cpu/acc/cpu_reduce_layer_acc.h"

#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

CpuReduceLayerAcc::~CpuReduceLayerAcc() {}

Status CalculateReduceDims(Blob *input_blob, ReduceLayerParam *layer_param,
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

template <typename T>
Status CpuReduceLayerAcc::ProcessReduce(Blob *input_blob, Blob *output_blob,
                     const std::vector<std::tuple<int, int, int>> &reduce_dims) {
    T *input_data  = static_cast<T *>(input_blob->GetHandle().base);
    T *output_data = static_cast<T *>(output_blob->GetHandle().base);
    auto input_dims = input_blob->GetBlobDesc().dims;
    auto output_count = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);

    int input_count            = DimsVectorUtils::Count(input_dims);
    T* pre_cal_reduce_result = new T[input_count];
    PreCalculateReduce(pre_cal_reduce_result, input_data, input_count);

    T *src       = pre_cal_reduce_result;
    T *tmp_ptr   = nullptr;
    bool release_mem = false;
    for (int i = 0; i < reduce_dims.size(); ++i) {
        auto reduce_dim   = reduce_dims[i];
        auto outer_count  = std::get<0>(reduce_dim);
        auto reduce_count = std::get<1>(reduce_dim);
        auto inner_count  = std::get<2>(reduce_dim);
        if (tmp_ptr != nullptr) {
            release_mem = true;
        }
        tmp_ptr = new T[inner_count * outer_count]();
        CalculateReduce(tmp_ptr, src, outer_count, reduce_count, inner_count);
        if (release_mem) {
            delete[] src;
        }
        src = tmp_ptr;
    }
    PostCalculateReduce(output_data, src, output_count);
    if (release_mem || reduce_dims.size() == 1) {
        delete[] src;
    }
    delete[] pre_cal_reduce_result;

    return TNN_OK;
}

Status CpuReduceLayerAcc::PreCalculateReduce(float *dst, float *src, int count) {
    ::memcpy(dst, src, count * sizeof(float));
    return TNN_OK;
}

Status CpuReduceLayerAcc::PreCalculateReduce(int32_t *dst, int32_t *src, int count) {
    ::memcpy(dst, src, count * sizeof(int32_t));
    return TNN_OK;
}

Status CpuReduceLayerAcc::PostCalculateReduce(float *dst, float *src, int count) {
    ::memcpy(dst, src, count * sizeof(float));
    return TNN_OK;
}

Status CpuReduceLayerAcc::PostCalculateReduce(int32_t *dst, int32_t *src, int count) {
    ::memcpy(dst, src, count * sizeof(int32_t));
    return TNN_OK;
}

Status CpuReduceLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuReduceLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (inputs.empty()) {
        LOGE("Error: invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "layer's inputs size must >= 2");
    }
    auto layer_param = dynamic_cast<ReduceLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is invalid\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is invalid");
    }

    Blob *input_blob  = inputs[0];
    auto input_dims   = input_blob->GetBlobDesc().dims;
    Blob *output_blob = outputs[0];
    auto output_count = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    // <outer count, reduce count, inner count>
    std::vector<std::tuple<int, int, int>> reduce_dims;
    Status status = CalculateReduceDims(input_blob, layer_param, reduce_dims);
    if (status != TNN_OK) {
        LOGE("CpuReduceLayerAcc: Calculate reduce dims failed\n");
        return status;
    }
    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        status = ProcessReduce<float>(input_blob, output_blob, reduce_dims);
        if (status != TNN_OK) {
            LOGE("CpuReduceLayerAcc: Process Reduce failed\n");
            return status;
        }
    } else if (output_blob->GetBlobDesc().data_type == DATA_TYPE_INT32) {
        status = ProcessReduce<int32_t>(input_blob, output_blob, reduce_dims);
        if (status != TNN_OK) {
            LOGE("CpuReduceLayerAcc: Process Reduce failed\n");
            return status;
        };
    } else if (output_blob->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        LOGE("Error: CpuReduceLayerAcc layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: CpuReduceLayerAcc layer acc dont support datatype");
    } else {
        LOGE("Error: CpuReduceLayerAcc layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: CpuReduceLayerAcc layer acc dont support datatype");
    }

    return TNN_OK;
}

}  // namespace TNN_NS
