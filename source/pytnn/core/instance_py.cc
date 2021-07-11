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
#include "tnn/core/instance.h"

namespace py = pybind11;

namespace TNN_NS {

    void InitInstancePy(py::module &m){
        py::class_<Instance, std::shared_ptr<Instance>>(m, "Instance")
            .def("Forward", &Instance::Forward)
            .def("GetForwardMemorySize", &Instance::GetForwardMemorySize)
            .def("SetForwardMemory", &Instance::SetForwardMemory)
            .def("Reshape", &Instance::Reshape)
	    .def("ShareCommandQueue", &Instance::ShareCommandQueue)
	    .def("SetCpuNumThreads", &Instance::SetCpuNumThreads)
            .def("SetInputMat", &Instance::SetInputMat, py::arg("mat"), py::arg("param"), py::arg("input_name") = "")
            .def("GetOutputMat", &Instance::GetOutputMat, py::arg("mat"), py::arg("param")=MatConvertParam(), py::arg("output_name")="", py::arg("device")=DEVICE_ARM, py::arg("mat_type")=NCHW_FLOAT);
    }

}  // namespace TNN_NS
