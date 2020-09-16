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
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

CpuReduceLayerAcc::~CpuReduceLayerAcc() {}

Status GetReduceDims(std::vector<std::tuple<int, int, int>>& reduce_dims, Blob* input_blob, ReduceLayerParam* layer_param) {
    auto input_dims   = input_blob->GetBlobDesc().dims;
    auto reduce_axises = layer_param->axis;
    for(int i = 0; i < reduce_axises.size(); i++) {
        int axis = reduce_axises[i];
        axis = axis >= 0 ? axis : axis + (int)input_dims.size();
        if (axis < 0 || axis >= input_dims.size()) {
            LOGE("Error: layer param axis is invalid\n");
            return Status(TNNERR_MODEL_ERR, "Error: layer param axis is invalid");
        }
        reduce_axises[i] = axis;
    }
    int dimension = (int)input_dims.size();
    int element_size = DimsVectorUtils::Count(input_dims, 0, dimension);
    std::vector<int> dims(dimension);
    for (int i = 0; i < dims.size(); ++i) {
        dims[i] = input_dims[i];
    }
    std::sort(reduce_axises.begin(), reduce_axises.end());
    std::vector<std::pair<int, int>> pair_axises;
    int head = reduce_axises[0], tail = reduce_axises[0];
    int length = 1;
    for (int i = 1; i < reduce_axises.size(); ++i) {
        int cur = reduce_axises[i];
        if (cur - tail == 1) {
            length++;
        } else {
            pair_axises.emplace_back(std::make_pair(head, length));
            head = reduce_axises[i];
            length = 1;
        }
        tail = reduce_axises[i];
    }
    pair_axises.emplace_back(std::make_pair(head, length));

    for (int i = 0; i < pair_axises.size(); ++i) {
        int outer_dim = 1, inner_dim = 1, channels = 1;
        auto head = pair_axises[i].first;
        auto length = pair_axises[i].second;
        for (int j = 0; j < head; ++j) {
            outer_dim *= dims[j];
        }
        for (int j = head; j < head + length; ++j) {
            channels *= dims[j];
            dims[j] = 1;
        }
        for (int j = head + length; j < dims.size(); ++j) {
            inner_dim *= dims[j];
        }
        if (1 == channels) {
            continue;
        }
        reduce_dims.emplace_back(std::make_tuple(outer_dim, channels, inner_dim));
    }
    if (reduce_dims.empty()) {
        reduce_dims.emplace_back(std::make_tuple(1, 1, element_size));
    }
    return TNN_OK;
}

Status CpuReduceLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuReduceLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (inputs.size() < 1) {
        LOGE("Error: invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "layer's inputs size must >= 2");
    }
    auto layer_param = dynamic_cast<ReduceLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is invalid\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is invalid");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    auto input_dims   = input_blob->GetBlobDesc().dims;

    std::vector<std::tuple<int, int, int>> reduce_dims;
    auto get_reduce_dims = GetReduceDims(reduce_dims, input_blob, layer_param);
    if(get_reduce_dims != TNN_OK) {
        return get_reduce_dims;
    }

    int output_size = std::get<0>(reduce_dims[reduce_dims.size() - 1]) * std::get<2>(reduce_dims[reduce_dims.size() - 1]);

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
        float *output_data = static_cast<float *>(output_blob->GetHandle().base);

        memset(output_data, 0, output_size * sizeof(float));
        float* src = input_data;
        for(int i = 0; i < reduce_dims.size() - 1; ++i) {
            auto reduce_dim = reduce_dims[i];
            auto inner_dim = std::get<2>(reduce_dim);
            auto outer_dim = std::get<0>(reduce_dim);
            auto channels = std::get<1>(reduce_dim);
            std::unique_ptr<float> dst (new float[inner_dim * outer_dim]);
            CalculateReduce(dst.get(), src, outer_dim, channels, inner_dim);
            src = dst.get();
        }

        auto reduce_dim = reduce_dims[reduce_dims.size() - 1];
        auto inner_dim = std::get<2>(reduce_dim);
        auto outer_dim = std::get<0>(reduce_dim);
        auto channels = std::get<1>(reduce_dim);
        auto dst = output_data;
        CalculateReduce(dst, src, outer_dim, channels, inner_dim);
    } else if (output_blob->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }

    return TNN_OK;
}

}  // namespace TNN_NS
