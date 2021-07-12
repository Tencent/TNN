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
   
    /*
    Status GetOutputMat(Instance* instance, std::shared_ptr<Mat>& mat,
                        MatConvertParam param = MatConvertParam(),
                        std::string output_name = "",
                        DeviceType device = DEVICE_ARM, MatType mat_type = NCHW_FLOAT) {
	std::shared_ptr<Mat> tmp_mat;
        instance->GetOutputMat(tmp_mat, param, output_name, device, mat_type);
     	using HolderType = std::shared_ptr<Mat>;
        auto * mat_inst = reinterpret_cast<py::detail::instance *>(py::cast(mat).ptr());
        auto * type_info = py::detail::get_type_info(typeid(HolderType));
	auto v_h = mat_inst->get_value_and_holder(type_info);
	py::detail::deregister_instance(mat_inst, v_h.value_ptr(), type_info);
        Mat* p_new = new Mat(tmp_mat->GetDeviceType(), tmp_mat->GetMatType(), tmp_mat->GetDims(), tmp_mat->GetData());
        v_h.value_ptr<Mat>() = p_new;
	if(v_h.holder_constructed()) {
            v_h.holder<HolderType>().reset(p_new);
        } else {
            new (&v_h.holder<HolderType>()) std::shared_ptr<Mat>(p_new);
	    v_h.set_holder_constructed();
        }
        py::detail::register_instance(mat_inst, v_h.value_ptr(), type_info);
	return TNN_OK;
    }
    */

    std::shared_ptr<Mat> GetOutputMat(Instance* instance, 
                        MatConvertParam param = MatConvertParam(),
                        std::string output_name = "",
                        DeviceType device = DEVICE_ARM, MatType mat_type = NCHW_FLOAT) {
	std::shared_ptr<Mat> output_mat;
        instance->GetOutputMat(output_mat, param, output_name, device, mat_type);
	return output_mat;
    }
 


    void InitInstancePy(py::module &m){
        py::class_<Instance, std::shared_ptr<Instance>>(m, "Instance")
            .def("Forward", &Instance::Forward)
            .def("GetForwardMemorySize", &Instance::GetForwardMemorySize)
            .def("SetForwardMemory", &Instance::SetForwardMemory)
            .def("Reshape", &Instance::Reshape)
	    .def("ShareCommandQueue", &Instance::ShareCommandQueue)
	    .def("SetCpuNumThreads", &Instance::SetCpuNumThreads)
            .def("SetInputMat", &Instance::SetInputMat, py::arg("mat"), py::arg("param"), py::arg("input_name") = "")
            .def("GetOutputMat", GetOutputMat, py::arg("param")=MatConvertParam(), py::arg("output_name")="", py::arg("device")=DEVICE_ARM, py::arg("mat_type")=NCHW_FLOAT);
//            .def("GetOutputMat", &Instance::GetOutputMat, py::arg("mat"), py::arg("param")=MatConvertParam(), py::arg("output_name")="", py::arg("device")=DEVICE_ARM, py::arg("mat_type")=NCHW_FLOAT);
    }

}  // namespace TNN_NS
