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

#include "tnn/device/opencl/acc/opencl_layer_acc.h"

namespace TNN_NS {
class OpenCLSqueezeLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLSqueezeLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    std::shared_ptr<cl::Buffer> inter_buffer_ = nullptr;
};

Status OpenCLSqueezeLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                   const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Squeeze Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "Squeeze";

    const int input_dims_size  = inputs[0]->GetBlobDesc().dims.size();
    const int output_dims_size = outputs[0]->GetBlobDesc().dims.size();

    execute_units_.resize(2);
    // image->buffer
    {
        std::string program_name = input_dims_size == 6   ? "image_6d_to_buffer"
                                   : input_dims_size == 5 ? "image_5d_to_buffer"
                                                          : "image_to_buffer";
        std::string kernel_name  = input_dims_size == 6   ? "Image6DToNCHWBuffer"
                                   : input_dims_size == 5 ? "Image5DToNCHWBuffer"
                                                          : "ImageToNCHWBuffer";
        ret                      = CreateExecuteUnit(execute_units_[0], program_name, kernel_name);
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    // buffer->image
    {
        std::string program_name = output_dims_size == 6   ? "buffer_to_image_6d"
                                   : output_dims_size == 5 ? "buffer_to_image_5d"
                                                           : "buffer_to_image";
        std::string kernel_name  = output_dims_size == 6   ? "NCHWBufferToImage6D"
                                   : output_dims_size == 5 ? "NCHWBufferToImage5D"
                                                           : "NCHWBufferToImage";
        ret                      = CreateExecuteUnit(execute_units_[1], program_name, kernel_name);
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    return TNN_OK;
}

OpenCLSqueezeLayerAcc::~OpenCLSqueezeLayerAcc() {}

Status OpenCLSqueezeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Squeeze Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto input  = inputs[0];
    auto output = outputs[0];

    auto input_dims             = input->GetBlobDesc().dims;
    auto output_dims            = output->GetBlobDesc().dims;
    const auto input_dims_size  = input_dims.size();
    const auto output_dims_size = output_dims.size();

    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();

    int size0 = UP_DIV(DimsFunctionUtils::GetDim(output_dims, 1), 4) * 4 * DimsFunctionUtils::GetDim(output_dims, 0);
    for (int i = 2; i < output_dims_size; i++) {
        size0 *= DimsFunctionUtils::GetDim(output_dims, i);
    }
    int size1 = UP_DIV(DimsFunctionUtils::GetDim(input_dims, 1), 4) * 4 * DimsFunctionUtils::GetDim(input_dims, 0);
    for (int i = 2; i < input_dims_size; i++) {
        size1 *= DimsFunctionUtils::GetDim(input_dims, i);
    }
    int blob_size = std::max(size0, size1) * sizeof(float);

    inter_buffer_ = std::make_shared<cl::Buffer>(*opencl_runtime->Context(), CL_MEM_READ_WRITE, blob_size);

    // image->buffer
    {
        uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], input_dims);
        execute_units_[0].ocl_kernel.setArg(idx++, *inter_buffer_.get());
        if (input_dims_size > 4) {
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 1)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 2)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 3)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 4)));
            if (input_dims_size == 6) {
                execute_units_[0].ocl_kernel.setArg(idx++,
                                                    static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 5)));
            }
        } else {
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 2)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 3)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 1)));
        }
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
    }

    // buffer->image
    {
        uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[1], output_dims);
        execute_units_[1].ocl_kernel.setArg(idx++, *inter_buffer_.get());
        if (output_dims_size > 4) {
            execute_units_[1].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 1)));
            execute_units_[1].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 2)));
            execute_units_[1].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 3)));
            execute_units_[1].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 4)));
            if (output_dims_size == 6) {
                execute_units_[1].ocl_kernel.setArg(idx++,
                                                    static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 5)));
            }
        } else {
            execute_units_[1].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 2)));
            execute_units_[1].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 3)));
            execute_units_[1].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 1)));
        }
        execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
    }

    return TNN_OK;
}

REGISTER_OPENCL_ACC(Squeeze, LAYER_SQUEEZE)
REGISTER_OPENCL_ACC(Squeeze, LAYER_UNSQUEEZE)

REGISTER_OPENCL_LAYOUT(LAYER_SQUEEZE, DATA_FORMAT_NHC4W4);
REGISTER_OPENCL_LAYOUT(LAYER_UNSQUEEZE, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
