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

#ifndef TNN_TOOLS_CONVERTER_SOURCE_TFLITE_TFLITE_UTILS_H_
#define TNN_TOOLS_CONVERTER_SOURCE_TFLITE_TFLITE_UTILS_H_
#include <cstdint>
#include <map>
#include <vector>

#include "tflite-schema/schema_generated.h"

namespace TNN_CONVERTER {

bool TFLiteConvertOHWI2OIHW(const float* src, float* dst, int CO, int KH, int KW, int CI);

bool TFLiteConvertOHWI2IOHW(const float* src, float* dst, int CO, int KH, int KW, int CI);

bool ConvertShapeFormatTFLite(std::vector<int32_t>& shape);

bool ConvertPermFormatTFLite(std::vector<int32_t>& perm);

// template <typename T>
bool ConvertConstFormatTFLite(int32_t const* dst, int32_t const* src, std::vector<int32_t> shape);

int ConvertAxisFormatTFLite(int axis, int input_shape_size = 4);

int Count(std::vector<int> shape);

int SizeofTFLiteTensorData(tflite::TensorType type);

void Mask(std::vector<int> shape, int mask, int upper, std::vector<int>& v);

}  // namespace TNN_CONVERTER
#endif  // TNN_TOOLS_CONVERTER_SOURCE_TFLITE_TFLITE_UTILS_H_
