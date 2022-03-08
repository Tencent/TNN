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

#include "tnn/device/opencl/acc/opencl_unary_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/device/opencl/opencl_utils.h"

namespace TNN_NS {

Status OpenCLUnaryLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Unary Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_         = true;
    std::string kernel_name = "Unary";

    std::set<std::string> build_options = CreateBuildOptions();
    build_options.insert(build_options_.begin(), build_options_.end());

    ret = CreateExecuteUnit(execute_units_[0], "unary", kernel_name, build_options);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return ret;
}

OpenCLUnaryLayerAcc::~OpenCLUnaryLayerAcc() {
}

Status OpenCLUnaryLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Unary Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto output_dims = outputs[0]->GetBlobDesc().dims;
    uint32_t idx = SetExecuteUnit3DSizeInfoDefault(execute_units_[0], output_dims);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    return TNN_OK;
}

std::vector<DataFormat> OpenCLUnaryLayerAcc::SupportDataFormat(DataType data_type,
                                                               int dims_size,
                                                               BlobType blob_type) {
    std::vector<DataFormat> support_list;
    if (dims_size >= 2 && dims_size <= 6) { // only support up to 6 dims
        support_list.push_back(DATA_FORMAT_NHC4W4);
    }
    return support_list;
}

}  // namespace TNN_NS
