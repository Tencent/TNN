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

#include "tnn/core/common.h"
#include "pytnn/pytnn.h"

namespace py = pybind11;


namespace TNN_NS {

void InitCommonPy(py::module &m) {
    //DataType
    py::enum_<DataType>(m, "DataType")
    .value("DATA_TYPE_AUTO", DataType::DATA_TYPE_AUTO)
    .value("DATA_TYPE_FLOAT", DataType::DATA_TYPE_FLOAT)
    .value("DATA_TYPE_HALF", DataType::DATA_TYPE_HALF)
    .value("DATA_TYPE_INT8", DataType::DATA_TYPE_INT8)
    .value("DATA_TYPE_INT32", DataType::DATA_TYPE_INT32)
    .value("DATA_TYPE_BFP16", DataType::DATA_TYPE_BFP16)
    .value("DATA_TYPE_INT64", DataType::DATA_TYPE_INT64)
    .value("DATA_TYPE_UINT32", DataType::DATA_TYPE_UINT32)
    .export_values();

    //DeviceType
    py::enum_<DeviceType>(m, "DeviceType")
    .value("DEVICE_NAIVE", DeviceType::DEVICE_NAIVE)
    .value("DEVICE_X86", DeviceType::DEVICE_X86)
    .value("DEVICE_ARM", DeviceType::DEVICE_ARM)
    .value("DEVICE_OPENCL", DeviceType::DEVICE_OPENCL)
    .value("DEVICE_METAL", DeviceType::DEVICE_METAL)
    .value("DEVICE_CUDA", DeviceType::DEVICE_CUDA)
    .export_values();
}

}  // namespace TNN_NS
