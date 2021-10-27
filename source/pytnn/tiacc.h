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

#ifndef TNN_SOURCE_TIACC_TIACC_H_
#define TNN_SOURCE_TIACC_TIACC_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#pragma warning(push)
#pragma warning(disable:4251)

namespace py = pybind11;

namespace TNN_NS {

void InitTIACCPy(py::module &m);

}  // namespace TNN_NS

#pragma warning(pop)

#endif  // TNN_SOURCE_TIACC_TIACC_H_
