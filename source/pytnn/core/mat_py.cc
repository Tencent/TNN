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

#include "pytnn/pytnn.h"
#include "tnn/core/mat.h"

namespace py = pybind11;

namespace TNN_NS {

    void InitMatPy(py::module& m) {
    // MatType
    py::enum_<MatType>(m, "MatType")
        .value("INVALID", MatType::INVALID)
        .value("N8UC3", MatType::N8UC3)
        .value("N8UC4", MatType::N8UC4)
        .value("NGRAY", MatType::NGRAY)
        .value("NNV21", MatType::NNV21)
        .value("NNV12", MatType::NNV12)
        .value("NCHW_FLOAT", MatType::NCHW_FLOAT)
        .value("NC_INT32", MatType::NC_INT32)
        .value("RESERVED_BFP16_TEST", MatType::RESERVED_BFP16_TEST)
        .value("RESERVED_FP16_TEST", MatType::RESERVED_FP16_TEST)
        .value("RESERVED_INT8_TEST", MatType::RESERVED_INT8_TEST)
        .export_values();

    py::class_<Mat, std::shared_ptr<Mat>>(m, "Mat")
        .def(py::init<DeviceType, MatType, std::vector<int>>())
        .def(py::init<DeviceType, MatType, std::vector<int>, char*>())
        .def(py::init<DeviceType, MatType>())
        .def("GetDeviceType", &Mat::GetDeviceType)
        .def("GetMatType", &Mat::GetMatType)
        .def("GetData", &Mat::GetData)
        .def("GetBatch", &Mat::GetBatch)
        .def("GetChannel", &Mat::GetChannel)
        .def("GetHeight", &Mat::GetHeight)
        .def("GetWidth", &Mat::GetWidth)
        .def("GetDim", &Mat::GetDim)
        .def("GetDims", &Mat::GetDims);
}

}  // namespace TNN_NS
