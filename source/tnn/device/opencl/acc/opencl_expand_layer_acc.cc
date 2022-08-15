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

#include "tnn/device/opencl/acc/opencl_expand_layer_acc.h"

#include <sstream>

#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

Status OpenCLExpandLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                  const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Expand Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "Expand";

    auto input  = inputs[0];
    auto output = outputs[0];

    auto input_dims       = input->GetBlobDesc().dims;
    auto output_dims      = output->GetBlobDesc().dims;
    auto input_dims_size  = input_dims.size();
    auto output_dims_size = output_dims.size();

    std::string src_format = "Image", dst_format = "Image";
    std::string img_to_buf_program_name = "image_to_buffer", buf_to_img_program_name = "buffer_to_image";
    src_format              = input_dims_size == 5 ? "Image5D" : input_dims_size == 6 ? "Image6D" : src_format;
    img_to_buf_program_name = input_dims_size == 5   ? "image_5d_to_buffer"
                              : input_dims_size == 6 ? "image_6d_to_buffer"
                                                     : img_to_buf_program_name;
    dst_format              = output_dims_size == 5 ? "Image5D" : output_dims_size == 6 ? "Image6D" : dst_format;
    buf_to_img_program_name = output_dims_size == 5   ? "buffer_to_image_5d"
                              : output_dims_size == 6 ? "buffer_to_image_6d"
                                                      : buf_to_img_program_name;
    execute_units_.resize(3);
    // image->buffer
    {
        ret =
            CreateExecuteUnit(execute_units_[0], img_to_buf_program_name, src_format + "ToNCHWBuffer", build_options_);
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    // expand
    {
        std::set<std::string> build_options;
        std::ostringstream oss;
        oss << "-DINNER_DIMS=" << output_dims.size();
        build_options.emplace(oss.str());
        build_options.insert(build_options_.begin(), build_options_.end());
        ret = CreateExecuteUnit(execute_units_[1], "expand", "Expand", build_options);
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    // buffer->image
    {
        ret =
            CreateExecuteUnit(execute_units_[2], buf_to_img_program_name, "NCHWBufferTo" + dst_format, build_options_);
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    return TNN_OK;
}

OpenCLExpandLayerAcc::~OpenCLExpandLayerAcc() {}

Status OpenCLExpandLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Expand Acc Expand\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)
    auto input  = inputs[0];
    auto output = outputs[0];

    auto input_dims  = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;

    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    int input_size                = sizeof(float) * DimsVectorUtils::Count(input_dims);
    int output_size               = sizeof(float) * DimsVectorUtils::Count(output_dims);

    src_buffer_ = std::make_shared<cl::Buffer>(*opencl_runtime->Context(), CL_MEM_READ_WRITE, input_size);
    dst_buffer_ = std::make_shared<cl::Buffer>(*opencl_runtime->Context(), CL_MEM_READ_WRITE, output_size);

    // image->buffer
    {
        uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], input_dims);
        execute_units_[0].ocl_kernel.setArg(idx++, *src_buffer_.get());
        if (input_dims.size() <= 4) {
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 2)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 3)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 1)));
        } else if (input_dims.size() == 5) {
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 1)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 2)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 3)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 4)));
        } else if (input_dims.size() == 6) {
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 1)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 2)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 3)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 4)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 5)));
        }
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
    }

    // expand
    {
        auto expanded_input_dims = input_dims;
        while (expanded_input_dims.size() < output_dims.size()) {
            expanded_input_dims.insert(expanded_input_dims.begin(), 1);
        }
        auto expanded_input_step = DimsFunctionUtils::GetDimsStep(expanded_input_dims);

        DimsVector fix_output_dims(6, 0), fix_input_step(6, 0);
        for (int i = output_dims.size() - 1; i >= 0; --i) {
            fix_output_dims[i] = output_dims[i];
            fix_input_step[i]  = expanded_input_step[i];
        }

        uint32_t idx = SetExecuteUnit1DSizeInfoDefault(execute_units_[1], output_dims);
        execute_units_[1].ocl_kernel.setArg(idx++, *src_buffer_.get());
        execute_units_[1].ocl_kernel.setArg(idx++, *dst_buffer_.get());
        execute_units_[1].ocl_kernel.setArg(idx++, fix_output_dims.size() * sizeof(int), fix_output_dims.data());
        execute_units_[1].ocl_kernel.setArg(idx++, expanded_input_dims.size() * sizeof(int),
                                            expanded_input_dims.data());
        execute_units_[1].ocl_kernel.setArg(idx++, fix_input_step.size() * sizeof(int), fix_input_step.data());
    }

    // buffer->image
    {
        uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[2], output_dims);
        execute_units_[2].ocl_kernel.setArg(idx++, *dst_buffer_.get());
        if (output_dims.size() <= 4) {
            execute_units_[2].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 2)));
            execute_units_[2].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 3)));
            execute_units_[2].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 1)));
        } else if (output_dims.size() == 5) {
            execute_units_[2].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 1)));
            execute_units_[2].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 2)));
            execute_units_[2].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 3)));
            execute_units_[2].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 4)));
        } else if (output_dims.size() == 6) {
            execute_units_[2].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 1)));
            execute_units_[2].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 2)));
            execute_units_[2].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 3)));
            execute_units_[2].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 4)));
            execute_units_[2].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 5)));
        }
        execute_units_[2].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
    }

    return TNN_OK;
}

REGISTER_OPENCL_ACC(Expand, LAYER_EXPAND)
REGISTER_OPENCL_LAYOUT(LAYER_EXPAND, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
