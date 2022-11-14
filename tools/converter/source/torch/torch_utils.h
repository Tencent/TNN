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

#ifndef TNN_TOOLS_CONVERTER_SOURCE_TORCH_TORCH_UTILS_H_
#define TNN_TOOLS_CONVERTER_SOURCE_TORCH_TORCH_UTILS_H_
#include <torch/csrc/jit/ir/ir.h>

#include "tnn/core/common.h"
#include "tnn/core/macro.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_CONVERTER {

TNN_NS::DataType TorchDataType2TnnDataType(at::ScalarType scalar_type);

bool DealPrime(const torch::jit::Node* node);

int GetDataByteSize(const torch::jit::Value* value);

TNN_NS::RawBuffer CreateRawBufferFromValue(const torch::jit::Value* value);

TNN_NS::RawBuffer CreateRawBufferFromTensor(const at::Tensor& tensor);

TNN_NS::DimsVector GetDimsFromValue(const torch::jit::Value* value);

TNN_NS::DimsVector GetDimsFromTensor(const at::Tensor& tensor);

std::vector<torch::jit::Value*> GetEffectiveInputValues(const torch::jit::Node* node);

TNN_NS::RawBuffer ConvertRawBufferToZero(TNN_NS::RawBuffer& src_buffer);

template <typename T>
static inline T GetValue(const torch::jit::Value* value) {
    auto optional_ivalue = toIValue(value);
    T res;
    if (!optional_ivalue) {
        LOGE("GetValue: must Constant Node.\n");
        return res;
    }
    c10::IValue& val  = optional_ivalue.value();
    auto optional_res = val.toOptional<T>();
    if (!optional_res) {
        LOGE("GetValue: value is None.");
        return res;
    }
    return optional_res.value();
}

template <typename T>
static std::vector<T> GetValue(const torch::jit::Value* value, std::vector<int>& shape) {
    std::vector<T> data;
    const auto tensor = GetValue<at::Tensor>(value);
    int size          = tensor.numel();
    if (!size) {
        return data;
    }
    auto shapes  = tensor.sizes().vec();
    auto strides = tensor.strides().vec();
    if (shapes.empty()) {
        shapes.push_back(size);
        strides.push_back(1);
    }
    shape.resize(shapes.size());
    for (int i = 0; i < shapes.size(); i++) {
        shape[i] = static_cast<int>(shapes[i]);
    }
    data.resize(size);
    int idx                                = 0;
    std::function<void(int, int)> copyData = [&](int dim, int offset) {
        if (dim == shapes.size() - 1) {
            for (int i = 0; i < shapes[dim]; i++) {
                data[idx++] = tensor.data_ptr<T>()[offset + i * strides[dim]];
            }
        } else {
            for (int i = 0; i < shapes[dim]; i++) {
                copyData(dim + 1, offset + i * strides[dim]);
            }
        }
    };
    copyData(0, 0);
    return data;
}
template <typename T>
static std::vector<T> GetValue(const at::Tensor& tensor, std::vector<int>& shape) {
    std::vector<T> data;
    int size = tensor.numel();
    if (!size) {
        return data;
    }
    const auto shapes  = tensor.sizes().vec();
    const auto strides = tensor.strides().vec();
    shape.resize(shapes.size());
    for (int i = 0; i < shapes.size(); i++) {
        shape[i] = static_cast<int>(shapes[i]);
    }
    data.resize(size);
    int idx                                = 0;
    std::function<void(int, int)> copyData = [&](int dim, int offset) {
        if (dim == shapes.size() - 1) {
            for (int i = 0; i < shapes[dim]; i++) {
                data[idx++] = tensor.data_ptr<T>()[offset + i * strides[dim]];
            }
        } else {
            for (int i = 0; i < shapes[dim]; i++) {
                copyData(dim + 1, offset + i * strides[dim]);
            }
        }
    };
    copyData(0, 0);
    return data;
}
}  // namespace TNN_CONVERTER

#endif  // TNN_TOOLS_CONVERTER_SOURCE_TORCH_TORCH_UTILS_H_
