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

namespace py = pybind11; 

namespace TNN_NS {

std::shared_ptr<Mat> ConvertNumpyToMat(py::array_t<float> input) {
    py::buffer_info input_info = input.request();
    float *input_ptr = static_cast<float *>(input_info.ptr);
    DimsVector input_dims;
    for(auto dim : input_info.shape) {
        input_dims.push_back(dim);
    }
    auto input_mat = std::make_shared<TNN_NS::Mat>(DEVICE_NAIVE, NCHW_FLOAT, input_dims, input_ptr);
    return input_mat;
}

py::array_t<float> ConvertMatToNumpy(std::shared_ptr<Mat> output_mat) {
    auto output_dims = output_mat->GetDims();
    std::vector<size_t> shape;
    for(auto dim : output_dims) {
	shape.push_back(dim);
    }
    int stride = sizeof(float);
    std::vector<size_t> strides(output_dims.size());
    for(int i = output_dims.size() - 1; i >=0; --i) {
     	strides[i] = stride;
        stride *= output_dims[i];
    }
    py::array_t<float> output =  py::array_t<float>(
            py::buffer_info(
                output_mat->GetData(),
                sizeof(float), //itemsize
                py::format_descriptor<float>::format(),
                shape.size(), // ndim
		shape, // shape
		strides // strides
           )
    );
    return output;
}

PYBIND11_MODULE(pytnn, m) {
    m.doc() = "pybind11 tnn torch plugin"; // optional module docstring

    InitStatusPy(m);
    InitCommonPy(m);
    InitMatPy(m);
    InitBlobPy(m);
    InitBlobConverterPy(m);

    InitTNNPy(m);
    InitInstancePy(m);

    m.def("convert_mat_to_numpy", &ConvertMatToNumpy, "convert mat to numpy");
    m.def("convert_numpy_to_mat", &ConvertNumpyToMat, "convert numpy to mat");
}

} // TNN_NS
