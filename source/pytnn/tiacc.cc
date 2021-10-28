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

#include "torch/csrc/jit/python/pybind_utils.h"
#include "torch/custom_class.h"
#include "torch/script.h"
#include "torch/torch.h"

#include "tnn/network/torch/torch_compile.h"

namespace py = pybind11; 

namespace TNN_NS {

void InitTIACCPy(py::module &m) {
    m.def("CompileTorch", &CompileTorch, py::arg("mod"), py::arg("input_shape"), py::arg("config"), py::arg("forward_func_name") = "forward"); 
}

} // TNN_NS
