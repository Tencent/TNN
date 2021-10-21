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

    py::class_<Mat, std::shared_ptr<Mat>>(m, "Mat", py::buffer_protocol())
    	.def(py::init<DeviceType, MatType, std::vector<int>>())
        .def(py::init<DeviceType, MatType, std::vector<int>, char*>())
        .def(py::init<DeviceType, MatType>())
        .def(py::init([](py::buffer input) {
            py::buffer_info input_info = input.request();
            void *input_ptr = static_cast<void *>(input_info.ptr);
            auto format = input_info.format;
            auto shape = input_info.shape;
            if(shape.size() != 4) {
                return TNN_NS::Mat(DEVICE_NAIVE, INVALID);
            }
            if(format == py::format_descriptor<unsigned char>::format()) {
                DimsVector input_dims = {(int)shape[0], (int)shape[3], (int)shape[1], (int)shape[2]};
                if(shape[3] == 1) {
                    return TNN_NS::Mat(DEVICE_NAIVE, NGRAY, input_dims, input_ptr);
                } else if(shape[3] == 3) {
                    return TNN_NS::Mat(DEVICE_NAIVE, N8UC3, input_dims, input_ptr);
                } else if(shape[4] == 4) {
                    return TNN_NS::Mat(DEVICE_NAIVE, N8UC4, input_dims, input_ptr);
                } else {
                    return TNN_NS::Mat(DEVICE_NAIVE, INVALID); 
		}
            } else {
                DimsVector input_dims;
                for(auto dim : input_info.shape) {
                    input_dims.push_back(dim);
                }
		if(format == py::format_descriptor<float>::format()) {
                    return TNN_NS::Mat(DEVICE_NAIVE, NCHW_FLOAT, input_dims, input_ptr);
                } else if(format == py::format_descriptor<int>::format()) {
                    return TNN_NS::Mat(DEVICE_NAIVE, NC_INT32, input_dims, input_ptr);
                } else if(format == py::format_descriptor<long long>::format()) {
                    return TNN_NS::Mat(DEVICE_NAIVE, NC_INT64, input_dims, input_ptr);
		} else {
                    return TNN_NS::Mat(DEVICE_NAIVE, INVALID);
                }
	    }
        }))
	.def("GetDeviceType", &Mat::GetDeviceType)
        .def("GetMatType", &Mat::GetMatType)
        .def("GetData", &Mat::GetData)
        .def("GetBatch", &Mat::GetBatch)
        .def("GetChannel", &Mat::GetChannel)
        .def("GetHeight", &Mat::GetHeight)
        .def("GetWidth", &Mat::GetWidth)
        .def("GetDim", &Mat::GetDim)
        .def("GetDims", &Mat::GetDims)
        .def_buffer([](Mat &mat) -> py::buffer_info {
            auto output_dims = mat.GetDims();
            auto mat_type = mat.GetMatType();
            auto device_type = mat.GetDeviceType();
	    int item_size = 0;
	    std::string format;
	    if(mat_type == NGRAY || mat_type == N8UC3 || mat_type == N8UC4) {
                item_size = 1;
		format = py::format_descriptor<unsigned char>::format();
            } else if(mat_type == NCHW_FLOAT) {
	        item_size = 4;
		format = py::format_descriptor<float>::format();
	    } else if(mat_type == NC_INT32) {
                item_size = 4;
		format = py::format_descriptor<int>::format();
	    } else if(mat_type == NC_INT64) {
                item_size = 8;
		format = py::format_descriptor<long long>::format();
	    }
            std::vector<size_t> shape;
            if(item_size == 1) {
                shape = {(size_t)output_dims[0], (size_t)output_dims[2], (size_t)output_dims[3], (size_t)output_dims[1]};
	    } else {
                for(auto dim : output_dims) {
                    shape.push_back(dim);
                }
	    }
            int stride = item_size;
            std::vector<size_t> strides(shape.size());
            for(int i = shape.size() - 1; i >=0; --i) {
                strides[i] = stride;
                stride *= shape[i];
            }
            return  py::buffer_info(
                    mat.GetData(),
                    item_size,
                    format,
                    shape.size(),
                    shape,
                    strides);
        });
}

}  // namespace TNN_NS
