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
#include "tnn/utils/mat_utils.h"

namespace py = pybind11;

namespace TNN_NS {

void SetTransform(WarpAffineParam* param, std::vector<std::vector<int>> transform) {
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 3; ++j) {
            param->transform[i][j] = transform[i][j];
        }
    }	
}

void InitMatUtilsPy(py::module& m) {

    py::enum_<InterpType>(m, "InterpType")
        .value("INTERP_TYPE_NEAREST", InterpType::INTERP_TYPE_NEAREST)
        .value("INTERP_TYPE_LINEAR", InterpType::INTERP_TYPE_LINEAR)
        .export_values();

    py::enum_<BorderType>(m, "BorderType")
        .value("BORDER_TYPE_CONSTANT", BorderType::BORDER_TYPE_CONSTANT)
        .value("BORDER_TYPE_REFLECT", BorderType::BORDER_TYPE_REFLECT)
        .value("BORDER_TYPE_EDGE", BorderType::BORDER_TYPE_EDGE)
        .export_values();

    py::enum_<ColorConversionType>(m, "ColorConversionType")
        .value("COLOR_CONVERT_NV12TOBGR", ColorConversionType::COLOR_CONVERT_NV12TOBGR)
        .value("COLOR_CONVERT_NV12TOBGRA", ColorConversionType::COLOR_CONVERT_NV12TOBGRA)
        .value("COLOR_CONVERT_NV21TOBGR", ColorConversionType::COLOR_CONVERT_NV21TOBGR)
        .value("COLOR_CONVERT_NV21TOBGRA", ColorConversionType::COLOR_CONVERT_NV21TOBGRA)
        .value("COLOR_CONVERT_BGRTOGRAY", ColorConversionType::COLOR_CONVERT_BGRTOGRAY)
        .value("COLOR_CONVERT_BGRATOGRAY", ColorConversionType::COLOR_CONVERT_BGRATOGRAY)
        .value("COLOR_CONVERT_RGBTOGRAY", ColorConversionType::COLOR_CONVERT_RGBTOGRAY)
        .value("COLOR_CONVERT_RGBATOGRAY", ColorConversionType::COLOR_CONVERT_RGBATOGRAY)
        .export_values();

    py::class_<ResizeParam>(m, "ResizeParam")
    .def(py::init<>())
    .def_readwrite("scale_w", &ResizeParam::scale_w)
    .def_readwrite("scale_h", &ResizeParam::scale_h)
    .def_readwrite("type", &ResizeParam::type);

    py::class_<CropParam>(m, "CropParam")
    .def(py::init<>())
    .def_readwrite("top_left_x", &CropParam::top_left_x)
    .def_readwrite("top_left_y", &CropParam::top_left_y)
    .def_readwrite("width", &CropParam::width)
    .def_readwrite("height", &CropParam::height)
    ;

    py::class_<WarpAffineParam>(m, "WarpAffineParam")
    .def(py::init<>())
    .def("SetTransform", &SetTransform)
    .def_readwrite("interp_type", &WarpAffineParam::interp_type)
    .def_readwrite("border_type", &WarpAffineParam::border_type)
    .def_readwrite("border_val", &WarpAffineParam::border_val)
    ;

    py::class_<CopyMakeBorderParam>(m, "CopyMakeBorderParam")
    .def(py::init<>())
    .def_readwrite("top", &CopyMakeBorderParam::top)
    .def_readwrite("bottom", &CopyMakeBorderParam::bottom)
    .def_readwrite("left", &CopyMakeBorderParam::left)
    .def_readwrite("right", &CopyMakeBorderParam::right)
    .def_readwrite("border_type", &CopyMakeBorderParam::border_type)
    .def_readwrite("border_val", &CopyMakeBorderParam::border_val)
    ;

    py::class_<MatUtils>(m, "MatUtils")
    .def_static("Copy", &MatUtils::Copy)
    .def_static("Resize", &MatUtils::Resize)
    .def_static("Crop", &MatUtils::Crop)
    .def_static("CvtColor", &MatUtils::CvtColor)
    .def_static("CopyMakeBorder", &MatUtils::CopyMakeBorder)
    .def_static("WarpAffine", &MatUtils::WarpAffine)
    ;

}

}  // namespace TNN_NS
