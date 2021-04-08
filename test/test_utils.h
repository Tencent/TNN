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

#ifndef TNN_TEST_TEST_UTILS_H_
#define TNN_TEST_TEST_UTILS_H_

#include "tnn/core/common.h"
#include "tnn/core/macro.h"
#include "tnn/utils/bfp16.h"

namespace TNN_NS {

DeviceType ConvertDeviceType(std::string device_type);

ModelType ConvertModelType(std::string model_type);

NetworkType ConvertNetworkType(std::string network_type);

Precision ConvertPrecision(std::string precision);

Precision SetPrecision(DeviceType dev, DataType dtype);

int CompareData(const float* ref_data, const float* result_data, size_t n, float ep);
int CompareData(const float* ref_data, const float* result_data, size_t n, float ep, float dp);
int CompareData(const bfp16_t* ref_data, const bfp16_t* result_data, size_t n, float ep);
int CompareData(const int8_t* ref_data, const int8_t* result_data, size_t n);
int CompareData(const int* ref_data, const int* result_data, size_t n);

int CompareData(const uint8_t* ref_data, const uint8_t* result_data, int mat_channel, int channel, size_t n);

}  // namespace TNN_NS

#endif  // TNN_TEST_TEST_UTILS_H_
