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

#ifndef TNN_SOURCE_PYTNN_PYTNN_H_
#define TNN_SOURCE_PYTNN_PYTNN_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include <tnn/core/tnn.h>
#include <tnn/core/instance.h>
#include <tnn/core/macro.h>
#include <tnn/core/status.h>

#pragma warning(push)
#pragma warning(disable:4251)

namespace py = pybind11;

namespace TNN_NS {

void InitStatusPy(py::module &m);
void InitCommonPy(py::module &m);
void InitMatPy(py::module &m);
void InitBlobPy(py::module& m);
void InitTNNPy(py::module &m);
void InitInstancePy(py::module &m);

void InitBFP16UtilsPy(py::module &m);
void InitBlobConverterPy(py::module &m);
void InitCpuUtilsPy(py::module &m);
void InitDataTypeUtilsPy(py::module &m);
void InitDeviceUtilsPy(py::module &m);
void InitDimsVectorUtilsPy(py::module &m);
void InitHalfUtilsPy(py::module &m);
void InitMatUtilsPy(py::module& m);
void InitStringUtilsPy(py::module& m);

}  // namespace TNN_NS

#pragma warning(pop)

#endif  // TNN_SOURCE_PYTNN_PYTNN_H_
