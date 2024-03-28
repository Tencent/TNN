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
#include "tnn/core/blob.h"

namespace py = pybind11;

namespace TNN_NS {

void InitBlobPy(py::module& m) {
    py::class_<BlobDesc>(m, "BlobDesc")
        .def(py::init<>())
        .def_readwrite("device_type", &BlobDesc::device_type)
        .def_readwrite("data_type", &BlobDesc::data_type)
        .def_readwrite("data_format", &BlobDesc::data_format)
        .def_readwrite("dims", &BlobDesc::dims)
        .def_readwrite("name", &BlobDesc::name);

    py::class_<BlobHandle>(m, "BlobHandle")
        .def(py::init<>())
        .def_readwrite("base", &BlobHandle::base)
        .def_readwrite("bytes_offset", &BlobHandle::bytes_offset); 

    py::class_<Blob>(m, "Blob")
	    .def(py::init<BlobDesc>())
        .def(py::init<BlobDesc, bool>())
        .def(py::init<BlobDesc, BlobHandle>())
        .def("GetBlobDesc", &Blob::GetBlobDesc)
        .def("SetBlobDesc", &Blob::SetBlobDesc)
        .def("GetHandle", &Blob::GetHandle)
        .def("SetHandle", &Blob::SetHandle);
}

}  // namespace TNN_NS
