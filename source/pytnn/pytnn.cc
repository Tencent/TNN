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

#include <tnn/core/tnn.h>
#include <tnn/core/mat.h>
#include <pytnn/pytnn.h>
#include <pytnn/tiacc.h>

namespace py = pybind11; 

namespace TNN_NS {

PYBIND11_MODULE(_pytnn, m) {
    m.doc() = "pybind11 tnn torch plugin"; // optional module docstring

    InitStatusPy(m);
    InitCommonPy(m);
    InitMatPy(m);
    InitBlobPy(m);
    InitBlobConverterPy(m);

    InitTNNPy(m);
    InitInstancePy(m);

    InitBFP16UtilsPy(m);
    InitCpuUtilsPy(m);
    InitDataTypeUtilsPy(m);
    InitDeviceUtilsPy(m);
    InitDimsVectorUtilsPy(m);
    InitHalfUtilsPy(m);
    InitMatUtilsPy(m);
    InitStringUtilsPy(m);

    //tiacc
    InitTIACCPy(m);
}

} // TNN_NS
