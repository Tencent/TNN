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
#include "tnn/utils/data_type_utils.h"

namespace py = pybind11;

namespace TNN_NS {

void InitDataTypeUtilsPy(py::module& m) {
    py::class_<DataTypeUtils>(m, "DataTypeUtils")
    .def_static("GetBytesSize", &DataTypeUtils::GetBytesSize)
    .def_static("GetDataTypeString", &DataTypeUtils::GetDataTypeString)
    .def_static("SaturateCast", &DataTypeUtils::SaturateCast)
    ;
}

}  // namespace TNN_NS
