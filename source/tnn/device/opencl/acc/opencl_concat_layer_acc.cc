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
#include "tnn/device/opencl/imagebuffer_convertor.h"

#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/string_utils_inner.h"

namespace TNN_NS {

typedef enum { BUFFER_COPY = 0, IMAGE_COPY, TWO_INPUTS_CHANNEL_4X, TWO_INPUTS_CHANNEL_MOD_123 } ConcatKernelType;

class OpenCLConcatLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLConcatLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

#if TNN_PROFILE
    virtual double GetBandwidth() override;
#endif

private:
    Status ReshapeImageConcat(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    Status ReshapeTwoInputsConcat(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    Status ReshapeBufferConcat(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

private:
    std::shared_ptr<cl::Buffer> output_buffer_              = nullptr;
    std::vector<std::shared_ptr<cl::Buffer>> input_buffers_ = {};
    int axis_                                               = 1;
    bool do_image_concat_                                   = true;
    ConcatKernelType concat_type_                           = BUFFER_COPY;
};

bool CheckIsTwoInputs(const size_t input_size, const int axis) {
    return input_size == 2 && axis == 1;
}

Status OpenCLConcatLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                  const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Concat Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "Concat";

    ConcatLayerParam *concat_param = dynamic_cast<ConcatLayerParam *>(param);
    CHECK_PARAM_NULL(concat_param);

    axis_            = concat_param->axis;
    do_image_concat_ = true;
    if (axis_ == 1) {
        for (size_t i = 0; i < inputs.size() - 1; ++i) {
            int channel = inputs[i]->GetBlobDesc().dims[1];
            if (channel % 4 != 0) {
                do_image_concat_ = false;
                break;
            }
        }
    }

    LOGD("do_image_concat: %s\n", do_image_concat_ ? "true" : "false");

    // choose kernel type
    if (CheckIsTwoInputs(inputs.size(), axis_)) {
        if (do_image_concat_) {
            if (gpu_info_.type == ADRENO) {
                concat_type_ = TWO_INPUTS_CHANNEL_4X;
            } else {
                concat_type_ = IMAGE_COPY;
            }
        } else {
            concat_type_ = TWO_INPUTS_CHANNEL_MOD_123;
        }
    } else {
        if (do_image_concat_) {
            concat_type_ = IMAGE_COPY;
        } else {
            concat_type_ = BUFFER_COPY;
        }
    }

    // create kernel
    std::string kernel_name;

    if (IMAGE_COPY == concat_type_) {
        std::string program_name = "copy";
        execute_units_.resize(inputs.size());
        for (size_t i = 0; i < execute_units_.size(); i++) {
            kernel_name = "CopyImage";
            ret         = CreateExecuteUnit(execute_units_[i], program_name, kernel_name);
            if (ret != TNN_OK) {
                return ret;
            }
        }
    } else if (TWO_INPUTS_CHANNEL_4X == concat_type_) {
        std::string program_name = "concat";
        kernel_name              = "ConcatChannel4X";
        execute_units_.resize(1);
        ret = CreateExecuteUnit(execute_units_[0], program_name, kernel_name);
        if (ret != TNN_OK) {
            return ret;
        }
    } else if (TWO_INPUTS_CHANNEL_MOD_123 == concat_type_) {
        std::set<std::string> build_options;
        build_options.emplace("-DCHANNEL0_MOD_4=" + ToString(inputs[0]->GetBlobDesc().dims[1] % 4));
        std::string program_name = "concat";
        kernel_name              = "ConcatChannel";
        execute_units_.resize(1);
        ret = CreateExecuteUnit(execute_units_[0], program_name, kernel_name, build_options);
        if (ret != TNN_OK) {
            return ret;
        }
    } else {
        // default BUFFER_COPY
        std::string program_name = "copy";
        execute_units_.resize(2 * inputs.size() + 1);
        for (size_t i = 0; i < inputs.size(); i++) {
            // Image to Buffer
            kernel_name = "CopyImageToBuffer";
            ret         = CreateExecuteUnit(execute_units_[2 * i], program_name, kernel_name);
            if (ret != TNN_OK) {
                return ret;
            }
            // Merge Buffer to Buffer
            kernel_name = "CopyBuffer";
            ret         = CreateExecuteUnit(execute_units_[2 * i + 1], program_name, kernel_name);
            if (ret != TNN_OK) {
                return ret;
            }
        }
        // Buffer to Image
        kernel_name = "CopyBufferToImage";
        ret         = CreateExecuteUnit(execute_units_[2 * inputs.size()], program_name, kernel_name);
        if (ret != TNN_OK) {
            return ret;
        }
    }

    return TNN_OK;
}

OpenCLConcatLayerAcc::~OpenCLConcatLayerAcc() {}

Status OpenCLConcatLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Concat Acc Reshape\n");

    if (IMAGE_COPY == concat_type_) {
        return ReshapeImageConcat(inputs, outputs);
    } else if (TWO_INPUTS_CHANNEL_4X == concat_type_) {
        return ReshapeTwoInputsConcat(inputs, outputs);
    } else if (TWO_INPUTS_CHANNEL_MOD_123 == concat_type_) {
        return ReshapeTwoInputsConcat(inputs, outputs);
    } else {
        // default BUFFER_COPY
        return ReshapeBufferConcat(inputs, outputs);
    }
}

#if TNN_PROFILE
double OpenCLConcatLayerAcc::GetBandwidth() {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    int data_type_size            = opencl_runtime->GetPrecision() != PRECISION_HIGH ? 2 : 4;
    return DimsVectorUtils::Count(output_dims_) * data_type_size / 1000.0 / 1000.0;
}
#endif

Status OpenCLConcatLayerAcc::ReshapeImageConcat(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto output      = outputs[0];
    auto output_dims = output->GetBlobDesc().dims;

    // output_width, output_height
    int output_wh[]     = {output_dims[3], output_dims[2]};
    int input_offset[]  = {0, 0, 0, 0};
    int output_offset[] = {0, 0, 0, 0};

    for (int i = 0; i < execute_units_.size(); ++i) {
        auto input      = inputs[i];
        auto input_dims = input->GetBlobDesc().dims;
        // input_width, input_height
        int input_wh[] = {input_dims[3], input_dims[2]};
        // batch, input_channel/4, input_height, input_width
        int region[] = {input_dims[0], UP_DIV(input_dims[1], 4), input_dims[2], input_dims[3]};

        auto &unit = execute_units_[i];
        int idx    = SetExecuteUnit2DSizeInfoDefault(unit, input_dims);
        unit.ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
        unit.ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
        unit.ocl_kernel.setArg(idx++, input_offset);
        unit.ocl_kernel.setArg(idx++, output_offset);
        unit.ocl_kernel.setArg(idx++, input_wh);
        unit.ocl_kernel.setArg(idx++, output_wh);
        unit.ocl_kernel.setArg(idx++, input_wh);

        output_offset[axis_] += region[axis_];
    }

    return TNN_OK;
}

Status OpenCLConcatLayerAcc::ReshapeBufferConcat(const std::vector<Blob *> &inputs,
                                                 const std::vector<Blob *> &outputs) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    auto output                   = outputs[0];
    auto output_dims              = output->GetBlobDesc().dims;

    // allocate temp buffers
    int blob_elem_size = opencl_runtime->GetPrecision() != PRECISION_HIGH ? 2 : 4;
    int output_size    = DimsVectorUtils::Count(output->GetBlobDesc().dims);
    output_buffer_ =
        std::make_shared<cl::Buffer>(*opencl_runtime->Context(), CL_MEM_READ_WRITE, output_size * blob_elem_size);
    input_buffers_.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        int input_size = DimsVectorUtils::Count(inputs[i]->GetBlobDesc().dims);
        input_buffers_[i] =
            std::make_shared<cl::Buffer>(*opencl_runtime->Context(), CL_MEM_READ_WRITE, input_size * blob_elem_size);
    }

    // set args
    int input_offset[]  = {0, 0, 0, 0};
    int output_offset[] = {0, 0, 0, 0};
    int output_stride[] = {output_dims[1] * output_dims[3] * output_dims[2], 1, output_dims[3] * output_dims[1],
                           output_dims[1]};
    for (size_t i = 0; i < inputs.size(); i++) {
        auto input      = inputs[i];
        auto input_dims = input->GetBlobDesc().dims;

        const int batch        = input_dims[0];
        const int input_height = input_dims[2];
        const int input_width  = input_dims[3];
        const int channels     = input_dims[1];

        int input_wh[]      = {input_width, input_height};
        int buffer_region[] = {batch, channels, input_height, input_width};
        int input_stride[]  = {input_dims[1] * input_dims[3] * input_dims[2], 1, input_dims[3] * input_dims[1],
                              input_dims[1]};

        // image to buffer (from (NH,C4W4) to NHWC)
        {
            auto &unit = execute_units_[2 * i];
            int idx    = SetExecuteUnit2DSizeInfoDefault(unit, input_dims);
            unit.ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
            unit.ocl_kernel.setArg(idx++, *(input_buffers_[i]));
            unit.ocl_kernel.setArg(idx++, input_offset);
            unit.ocl_kernel.setArg(idx++, input_offset);
            unit.ocl_kernel.setArg(idx++, input_wh);
            unit.ocl_kernel.setArg(idx++, input_stride);
            unit.ocl_kernel.setArg(idx++, 4 * sizeof(int), input_dims.data());
        }

        // buffer to buffer
        {
            auto &unit = execute_units_[2 * i + 1];
            // special process: buffer_region[1] * 4 to get the right global size
            int idx = SetExecuteUnit2DSizeInfoDefault(
                unit, {buffer_region[0], buffer_region[1] * 4, buffer_region[2], buffer_region[3]});
            unit.ocl_kernel.setArg(idx++, *(input_buffers_[i]));
            unit.ocl_kernel.setArg(idx++, *output_buffer_);
            unit.ocl_kernel.setArg(idx++, input_offset);
            unit.ocl_kernel.setArg(idx++, output_offset);
            unit.ocl_kernel.setArg(idx++, input_stride);
            unit.ocl_kernel.setArg(idx++, output_stride);
            unit.ocl_kernel.setArg(idx++, input_wh);
        }
        output_offset[axis_] += buffer_region[axis_];
    }

    // buffer to image (from NHWC to (NH,C4W4))
    {
        int output_wh[] = {output_dims[3], output_dims[2]};
        auto &unit      = execute_units_[2 * inputs.size()];
        int idx         = SetExecuteUnit2DSizeInfoDefault(unit, output_dims);
        unit.ocl_kernel.setArg(idx++, *output_buffer_);
        unit.ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
        unit.ocl_kernel.setArg(idx++, input_offset);
        unit.ocl_kernel.setArg(idx++, input_offset);
        unit.ocl_kernel.setArg(idx++, output_stride);
        unit.ocl_kernel.setArg(idx++, output_wh);
        unit.ocl_kernel.setArg(idx++, output_wh);
        unit.ocl_kernel.setArg(idx++, output_size - 1);
    }

    return TNN_OK;
}

Status OpenCLConcatLayerAcc::ReshapeTwoInputsConcat(const std::vector<Blob *> &inputs,
                                                    const std::vector<Blob *> &outputs) {
    run_3d_ndrange_  = true;
    auto output      = outputs[0];
    auto output_dims = output->GetBlobDesc().dims;
    auto input0      = inputs[0];
    auto input1      = inputs[1];

    // [output_channle/4, output_width, batch * output_height]
    execute_units_[0].global_work_size = {static_cast<uint32_t>(UP_DIV(output_dims[1], 4)),
                                          static_cast<uint32_t>(output_dims[3]),
                                          static_cast<uint32_t>(output_dims[0] * output_dims[2])};
    execute_units_[0].local_work_size  = LocalWS3DDefault(execute_units_[0]);
    int idx                            = 0;
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[2]);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input0->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input1->GetHandle().base));
    // input channel
    execute_units_[0].ocl_kernel.setArg(idx++, input0->GetBlobDesc().dims[1]);
    // output channel
    execute_units_[0].ocl_kernel.setArg(idx++, output_dims[1]);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
    return TNN_OK;
}

REGISTER_OPENCL_ACC(Concat, LAYER_CONCAT)

}  // namespace TNN_NS
