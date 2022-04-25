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

#include <algorithm>
#include "tnn/device/opencl/acc/opencl_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/device/opencl/opencl_context.h"
#include "tnn/device/opencl/opencl_runtime.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

class OpenCLReorgLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLReorgLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    std::shared_ptr<cl::Buffer> input_buffer_ = nullptr;
    std::shared_ptr<cl::Buffer> output_buffer_ = nullptr;
    int stride_     = 0;
    int forward_    = 0;
    int mode_       = 0;
};

Status OpenCLReorgLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Reorg Acc \n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "Reorg";

    auto layer_param = dynamic_cast<ReorgLayerParam *>(param);
    if (layer_param == nullptr) {
        LOGE("ReorgLayerParam is null!\n");
        return Status(TNNERR_MODEL_ERR, "ReorgLayerParam is null");
    }

    forward_    = layer_param->forward;
    stride_     = layer_param->stride;
    mode_       = layer_param->mode;

    std::string program_name, kernel_name;
    execute_units_.resize(3);
    program_name = "image_to_buffer";
    kernel_name  = "ImageToNCHWBufferFLOAT";
    ret          = CreateExecuteUnit(execute_units_[0], program_name, kernel_name, build_options_);
    if (ret != TNN_OK) {
        return ret;
    }

    program_name = "reorg";
    kernel_name  = "Reorg";
    ret          = CreateExecuteUnit(execute_units_[1], program_name, kernel_name, build_options_);
    if (ret != TNN_OK) {
        return ret;
    }

    program_name = "buffer_to_image";
    kernel_name  = "NCHWBufferToImageFLOAT";
    ret          = CreateExecuteUnit(execute_units_[2], program_name, kernel_name, build_options_);
    if (ret != TNN_OK) {
        return ret;
    }

    return TNN_OK;
}

OpenCLReorgLayerAcc::~OpenCLReorgLayerAcc() {}

Status OpenCLReorgLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Reorg Acc Reshape\n");
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto input  = inputs[0];
    auto output = outputs[0];

    auto input_dims  = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;

    int type_size = sizeof(float);
    if (opencl_runtime->GetPrecision() != PRECISION_HIGH) {
        type_size = 2;
    }
    int dims_count = DimsVectorUtils::Count(input->GetBlobDesc().dims);
    input_buffer_ = std::make_shared<cl::Buffer>(*opencl_runtime->Context(), CL_MEM_READ_WRITE, dims_count * type_size);
    output_buffer_ =
        std::make_shared<cl::Buffer>(*opencl_runtime->Context(), CL_MEM_READ_WRITE, dims_count * type_size);

    auto &unit0              = execute_units_[0];
    uint32_t idx = SetExecuteUnit2DSizeInfoDefault(unit0, input_dims);
    unit0.ocl_kernel.setArg(idx++, *input_buffer_);
    unit0.ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 2))); //input height
    unit0.ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 3))); //input width
    unit0.ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 1))); //input channel
    unit0.ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));

    idx = 0;
    auto &unit1            = execute_units_[1];
    unit1.global_work_size = {(uint32_t) DimsVectorUtils::Count(input_dims)};
    unit1.local_work_size  = {unit1.workgroupsize_max};
    unit1.ocl_kernel.setArg(idx++, unit1.global_work_size[0]);
    unit1.ocl_kernel.setArg(idx++, *input_buffer_);
    unit1.ocl_kernel.setArg(idx++, *output_buffer_);
    unit1.ocl_kernel.setArg(idx++, forward_ ? DimsFunctionUtils::GetDim(input_dims, 3) : DimsFunctionUtils::GetDim(output_dims, 3)); // input width
    unit1.ocl_kernel.setArg(idx++, forward_ ? DimsFunctionUtils::GetDim(input_dims, 2) : DimsFunctionUtils::GetDim(output_dims, 2)); // input height
    unit1.ocl_kernel.setArg(idx++, forward_ ? DimsFunctionUtils::GetDim(input_dims, 1) : DimsFunctionUtils::GetDim(output_dims, 1)); // input channel
    unit1.ocl_kernel.setArg(idx++, forward_ ? DimsFunctionUtils::GetDim(input_dims, 0) : DimsFunctionUtils::GetDim(output_dims, 0)); // batch
    unit1.ocl_kernel.setArg(idx++, stride_);
    unit1.ocl_kernel.setArg(idx++, stride_ * stride_);
    unit1.ocl_kernel.setArg(idx++, forward_);
    unit1.ocl_kernel.setArg(idx++, mode_);

    auto &unit2            = execute_units_[2];
    idx = SetExecuteUnit2DSizeInfoDefault(unit2, output_dims);
    unit2.ocl_kernel.setArg(idx++, *output_buffer_);
    unit2.ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 2)));   // output height
    unit2.ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 3)));    // output width
    unit2.ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 1)));  // output channel
    unit2.ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));

    return TNN_OK;
}

REGISTER_OPENCL_ACC(Reorg, LAYER_REORG)
REGISTER_OPENCL_LAYOUT(LAYER_REORG, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
