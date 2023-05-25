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
#include "tnn/core/common.h"

namespace py = pybind11;

namespace TNN_NS {

void InitCommonPy(py::module &m) {

    // DataType
    py::enum_<DataType>(m, "DataType")
        .value("DATA_TYPE_AUTO", DataType::DATA_TYPE_AUTO)
        .value("DATA_TYPE_FLOAT", DataType::DATA_TYPE_FLOAT)
        .value("DATA_TYPE_HALF", DataType::DATA_TYPE_HALF)
        .value("DATA_TYPE_INT8", DataType::DATA_TYPE_INT8)
        .value("DATA_TYPE_INT32", DataType::DATA_TYPE_INT32)
        .value("DATA_TYPE_BFP16", DataType::DATA_TYPE_BFP16)
        .value("DATA_TYPE_INT64", DataType::DATA_TYPE_INT64)
        .value("DATA_TYPE_UINT32", DataType::DATA_TYPE_UINT32)
        .export_values();

    // DataFormat
    py::enum_<DataFormat>(m, "DataFormat")
        .value("DATA_FORMAT_AUTO", DataFormat::DATA_FORMAT_AUTO)
        .value("DATA_FORMAT_NCHW", DataFormat::DATA_FORMAT_NCHW)
        .value("DATA_FORMAT_NHWC", DataFormat::DATA_FORMAT_NHWC)
        .value("DATA_FORMAT_NHWC4", DataFormat::DATA_FORMAT_NHWC4)
        .value("DATA_FORMAT_NC2HW2", DataFormat::DATA_FORMAT_NC2HW2)
        .value("DATA_FORMAT_NC4HW4", DataFormat::DATA_FORMAT_NC4HW4)
        .value("DATA_FORMAT_NC8HW8", DataFormat::DATA_FORMAT_NC8HW8)
        .value("DATA_FORMAT_NC16HW16", DataFormat::DATA_FORMAT_NC16HW16)
        .value("DATA_FORMAT_NCDHW", DataFormat::DATA_FORMAT_NCDHW)
        .value("DATA_FORMAT_NHC4W4", DataFormat::DATA_FORMAT_NHC4W4)
        .value("DATA_FORMAT_CNH4", DataFormat::DATA_FORMAT_CNH4)
        .export_values();

    // Precison
    py::enum_<Precision>(m, "Precision")
        .value("PRECISION_AUTO", Precision::PRECISION_AUTO)
        .value("PRECISION_NORMAL", Precision::PRECISION_NORMAL)
        .value("PRECISION_HIGH", Precision::PRECISION_HIGH)
        .value("PRECISION_LOW", Precision::PRECISION_LOW)
        .export_values();

    // NetworkType
    py::enum_<NetworkType>(m, "NetworkType")
        .value("NETWORK_TYPE_AUTO", NetworkType::NETWORK_TYPE_AUTO)
        .value("NETWORK_TYPE_DEFAULT", NetworkType::NETWORK_TYPE_DEFAULT)
        .value("NETWORK_TYPE_OPENVINO", NetworkType::NETWORK_TYPE_OPENVINO)
        .value("NETWORK_TYPE_COREML", NetworkType::NETWORK_TYPE_COREML)
        .value("NETWORK_TYPE_SNPE", NetworkType::NETWORK_TYPE_SNPE)
        .value("NETWORK_TYPE_HIAI", NetworkType::NETWORK_TYPE_HIAI)
        .value("NETWORK_TYPE_ATLAS", NetworkType::NETWORK_TYPE_ATLAS)
        .value("NETWORK_TYPE_HUAWEI_NPU", NetworkType::NETWORK_TYPE_HUAWEI_NPU)
        .value("NETWORK_TYPE_RK_NPU", NetworkType::NETWORK_TYPE_RK_NPU)
        .value("NETWORK_TYPE_TENSORRT", NetworkType::NETWORK_TYPE_TENSORRT)
        .value("NETWORK_TYPE_TNNTORCH", NetworkType::NETWORK_TYPE_TNNTORCH)
        .export_values();

    // DeviceType
    py::enum_<DeviceType>(m, "DeviceType")
        .value("DEVICE_NAIVE", DeviceType::DEVICE_NAIVE)
        .value("DEVICE_X86", DeviceType::DEVICE_X86)
        .value("DEVICE_ARM", DeviceType::DEVICE_ARM)
        .value("DEVICE_OPENCL", DeviceType::DEVICE_OPENCL)
        .value("DEVICE_METAL", DeviceType::DEVICE_METAL)
        .value("DEVICE_CUDA", DeviceType::DEVICE_CUDA)
        .value("DEVICE_DSP", DeviceType::DEVICE_DSP)
        .value("DEVICE_ATLAS", DeviceType::DEVICE_ATLAS)
        .value("DEVICE_HUAWEI_NPU", DeviceType::DEVICE_HUAWEI_NPU)
        .value("DEVICE_RK_NPU", DeviceType::DEVICE_RK_NPU)
        .export_values();

    // ShareMemoryMode
    py::enum_<ShareMemoryMode>(m, "ShareMemoryMode")
        .value("SHARE_MEMORY_MODE_DEFAULT", ShareMemoryMode::SHARE_MEMORY_MODE_DEFAULT)
        .value("SHARE_MEMORY_MODE_SHARE_ONE_THREAD", ShareMemoryMode::SHARE_MEMORY_MODE_SHARE_ONE_THREAD)
        .value("SHARE_MEMORY_MODE_SET_FROM_EXTERNAL", ShareMemoryMode::SHARE_MEMORY_MODE_SET_FROM_EXTERNAL)
        .export_values();

    // ModelType
    py::enum_<ModelType>(m, "ModelType")
        .value("MODEL_TYPE_TNN", ModelType::MODEL_TYPE_TNN)
        .value("MODEL_TYPE_NCNN", ModelType::MODEL_TYPE_NCNN)
        .value("MODEL_TYPE_OPENVINO", ModelType::MODEL_TYPE_OPENVINO)
        .value("MODEL_TYPE_COREML", ModelType::MODEL_TYPE_COREML)
        .value("MODEL_TYPE_SNPE", ModelType::MODEL_TYPE_SNPE)
        .value("MODEL_TYPE_HIAI", ModelType::MODEL_TYPE_HIAI)
        .value("MODEL_TYPE_ATLAS", ModelType::MODEL_TYPE_ATLAS)
        .value("MODEL_TYPE_RKCACHE", ModelType::MODEL_TYPE_RKCACHE)
        .value("MODEL_TYPE_TORCHSCRIPT", ModelType::MODEL_TYPE_TORCHSCRIPT)
	.export_values();

    py::class_<NetworkConfig>(m, "NetworkConfig")
        .def(py::init<>())
    	.def_readwrite("device_type", &NetworkConfig::device_type)
        .def_readwrite("device_id", &NetworkConfig::device_id)
        .def_readwrite("data_format", &NetworkConfig::data_format)
        .def_readwrite("network_type", &NetworkConfig::network_type)
	.def_readwrite("share_memory_mode", &NetworkConfig::share_memory_mode)
        .def_readwrite("library_path", &NetworkConfig::library_path)
        .def_readwrite("precision", &NetworkConfig::precision)
        .def_readwrite("cache_path", &NetworkConfig::cache_path)
        .def_readwrite("enable_tune_kernel", &NetworkConfig::enable_tune_kernel);

    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
    	.def_readwrite("model_type", &ModelConfig::model_type)
        .def_readwrite("params", &ModelConfig::params);
};

}  // namespace TNN_NS
