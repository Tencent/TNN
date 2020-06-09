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

#include "tnn/device/opencl/acc/opencl_reshape_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

Status OpenCLReshapeLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                   const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Reshape Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "Reshape";

    execute_units_.resize(2);
    // image->buffer
    {
        ret = CreateExecuteUnit(execute_units_[0], "image_to_buffer", "ImageToNCHWBuffer");
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    // buffer->image
    {
        ret = CreateExecuteUnit(execute_units_[1], "buffer_to_image", "NCHWBufferToImage");
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    return TNN_OK;
}

OpenCLReshapeLayerAcc::~OpenCLReshapeLayerAcc() {}

Status OpenCLReshapeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Reshape Acc Reshape\n");
    auto input  = inputs[0];
    auto output = outputs[0];

    auto input_dims  = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;

    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    int blob_size                 = sizeof(float) * DimsVectorUtils::Count(input_dims);

    inter_buffer_ = std::make_shared<cl::Buffer>(*opencl_runtime->Context(), CL_MEM_READ_WRITE, blob_size);

    // image->buffer
    {
        uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], input_dims);
        execute_units_[0].ocl_kernel.setArg(idx++, *inter_buffer_.get());
        execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(input_dims[2]));
        execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(input_dims[3]));
        execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(input_dims[1]));
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
    }

    // buffer->image
    {
        uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[1], output_dims);
        execute_units_[1].ocl_kernel.setArg(idx++, *inter_buffer_.get());
        execute_units_[1].ocl_kernel.setArg(idx++, static_cast<uint32_t>(output_dims[2]));
        execute_units_[1].ocl_kernel.setArg(idx++, static_cast<uint32_t>(output_dims[3]));
        execute_units_[1].ocl_kernel.setArg(idx++, static_cast<uint32_t>(output_dims[1]));
        execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
    }

    return TNN_OK;
}

REGISTER_OPENCL_ACC(Reshape, LAYER_RESHAPE)
REGISTER_OPENCL_ACC(Reshape, LAYER_FLATTEN)

}  // namespace TNN_NS
