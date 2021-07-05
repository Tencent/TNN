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

#ifndef TNN_SOURCE_TNN_DEVICE_RK_NPU_RKNPU_UTILS_H_
#define TNN_SOURCE_TNN_DEVICE_RK_NPU_RKNPU_UTILS_H_

#include <tnn/core/blob.h>
#include <tnn/interpreter/layer_resource.h>
#include <tnn/interpreter/raw_buffer.h>

#include "rknpu/rknpu_pub.h"
#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/layer_param.h"

namespace TNN_NS {

class RknpuUtils {
public:
    static std::shared_ptr<rk::nn::Tensor> CreateRknnTensor(
        rk::nn::Graph *graph, const std::string &name, const std::vector<int> &dims, const void *data = NULL,
        const rk::nn::TensorRole role = rk::nn::TensorRole::VAR, const DataType type = DATA_TYPE_FLOAT,
        const rk::nn::DataLayoutType layout    = rk::nn::DataLayoutType::NCHW,
        const rk::nn::QuantizationType qntType = rk::nn::QuantizationType::NONE, const uint8_t bits = 8,
        const float scale = 1.0, const uint32_t zero_point = 0, const int8_t fl = 0);

    static Status GetPadType(rk::nn::PadType &rk_pad_type, int pad_type);

    static uint32_t CalcSize(rk::nn::PrecisionType type, std::vector<int32_t> dims);
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_RK_NPU_RKNPU_UTILS_H_
