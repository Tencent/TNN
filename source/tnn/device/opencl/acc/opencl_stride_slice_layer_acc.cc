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
#include "tnn/device/opencl/opencl_execute_unit.h"
#include "tnn/device/opencl/opencl_memory.h"
#include "tnn/device/opencl/opencl_runtime.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

typedef enum { STRIDE_SLICE_IMAGE = 0, STRIDE_SLICE_C4_UNITE = 1, STRIDE_SLICE_C4_SEPARATE = 2 } StrideSliceType;

class OpenCLStrideSliceLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLStrideSliceLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    std::vector<int> begins_;
    std::vector<int> strides_;
    std::vector<int> ends_;
    StrideSliceType type_ = STRIDE_SLICE_C4_SEPARATE;
    std::shared_ptr<cl::Buffer> buffer_;
};

Status OpenCLStrideSliceLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                       const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init StrideSlice Acc \n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_  = false;
    op_name_         = "StrideSlice";
    auto layer_param = dynamic_cast<StrideSliceLayerParam *>(param);
    if (layer_param == nullptr) {
        LOGE("StrideSliceLayerParam is null!\n");
        return Status(TNNERR_MODEL_ERR, "StrideSliceLayerParam is null");
    }

    begins_ = layer_param->begins;
    std::reverse(begins_.begin(), begins_.end());
    strides_ = layer_param->strides;
    std::reverse(strides_.begin(), strides_.end());
    ends_ = layer_param->ends;
    std::reverse(ends_.begin(), ends_.end());

    for (int i = 0; i < ends_.size(); ++i) {
        if (ends_[i] == 0) {
            ends_[i] = inputs[0]->GetBlobDesc().dims[i];
        }
    }

    int begin_channel  = begins_[1];
    int end_channel    = ends_[1];
    int channel_stride = strides_[1];
    type_              = STRIDE_SLICE_C4_SEPARATE;
    if (begin_channel % 4 == 0 && channel_stride == 1) {
        type_ = STRIDE_SLICE_IMAGE;
        for (auto value : strides_) {
            if (value > 1) {
                type_ = STRIDE_SLICE_C4_UNITE;
            }
        }
    }

    std::string program_name, kernel_name;
    if (type_ == STRIDE_SLICE_IMAGE) {
        execute_units_.resize(1);
        program_name = "copy";
        kernel_name  = "CopyImage";
        ret          = CreateExecuteUnit(execute_units_[0], program_name, kernel_name);
        if (ret != TNN_OK) {
            return ret;
        }
    } else if (type_ == STRIDE_SLICE_C4_UNITE) {
        execute_units_.resize(1);
        program_name = "stride_slice";
        kernel_name  = "StrideSliceC4Unite";
        ret          = CreateExecuteUnit(execute_units_[0], program_name, kernel_name);
        if (ret != TNN_OK) {
            return ret;
        }
    } else {
        execute_units_.resize(2);
        program_name = "image_to_buffer";
        kernel_name  = "ImageToNCHWBufferFLOAT";
        ret          = CreateExecuteUnit(execute_units_[0], program_name, kernel_name);
        if (ret != TNN_OK) {
            return ret;
        }
        program_name = "stride_slice";
        kernel_name  = "StrideSliceC4Separate";
        ret          = CreateExecuteUnit(execute_units_[1], program_name, kernel_name);
        if (ret != TNN_OK) {
            return ret;
        }
    }

    return TNN_OK;
}

OpenCLStrideSliceLayerAcc::~OpenCLStrideSliceLayerAcc() {}

Status OpenCLStrideSliceLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("StrideSlice Acc Reshape\n");
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    auto input                    = inputs[0];
    auto output                   = outputs[0];
    auto input_dims               = input->GetBlobDesc().dims;
    auto output_dims              = output->GetBlobDesc().dims;
    int inputWH[]                 = {input_dims[3], input_dims[2]};
    int outputWH[]                = {output_dims[3], output_dims[2]};

    if (type_ == STRIDE_SLICE_IMAGE) {
        auto &unit0          = execute_units_[0];
        int inputOffset[]    = {begins_[0], begins_[1] / 4, begins_[2], begins_[3]};
        int outputOffset[]   = {0, 0, 0, 0};
        DimsVector exec_dims = {ends_[0] - begins_[0], ends_[1] - begins_[1], ends_[2] - begins_[2],
                                ends_[3] - begins_[3]};
        int idx = SetExecuteUnit2DSizeInfoDefault(unit0, exec_dims);
        unit0.ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
        unit0.ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
        unit0.ocl_kernel.setArg(idx++, inputOffset);
        unit0.ocl_kernel.setArg(idx++, outputOffset);
        unit0.ocl_kernel.setArg(idx++, inputWH);
        unit0.ocl_kernel.setArg(idx++, outputWH);
        unit0.ocl_kernel.setArg(idx++, outputWH);
    } else if (type_ == STRIDE_SLICE_C4_UNITE) {
        auto &unit0               = execute_units_[0];
        int output_channel_blocks = UP_DIV(output_dims[1], 4);
        // output_width * output_channel_blocks, output_batch * output_height
        unit0.global_work_size = {(uint32_t)output_dims[3] * output_channel_blocks,
                                    (uint32_t)output_dims[0] * output_dims[2]};
        unit0.local_work_size  = LocalWS2DDefault(unit0);
        unit0.ocl_kernel.setArg(0, *((cl::Image *)input->GetHandle().base));
        unit0.ocl_kernel.setArg(1, *((cl::Image *)output->GetHandle().base));
        unit0.ocl_kernel.setArg(2, 4 * sizeof(int), begins_.data());
        unit0.ocl_kernel.setArg(3, 4 * sizeof(int), strides_.data());
        unit0.ocl_kernel.setArg(4, inputWH);
        unit0.ocl_kernel.setArg(5, outputWH);
    } else {
        int dims_count = DimsVectorUtils::Count(input->GetBlobDesc().dims);
        int type_size  = sizeof(float);
        if (opencl_runtime->GetPrecision() != PRECISION_HIGH) {
            type_size = 2;
        }
        buffer_ = std::make_shared<cl::Buffer>(*opencl_runtime->Context(), CL_MEM_READ_WRITE, dims_count * type_size);

        auto &unit0              = execute_units_[0];
        int input_channel_blocks = UP_DIV(input_dims[1], 4);
        // input_width * input_channel_blocks, input_batch * input_height
        unit0.global_work_size = {(uint32_t)input_dims[3] * input_channel_blocks,
                                    (uint32_t)input_dims[0] * input_dims[2]};
        unit0.local_work_size  = LocalWS2DDefault(unit0);
        unit0.ocl_kernel.setArg(0, unit0.global_work_size[0]);
        unit0.ocl_kernel.setArg(1, unit0.global_work_size[1]);
        unit0.ocl_kernel.setArg(2, *buffer_);
        // input height
        unit0.ocl_kernel.setArg(3, static_cast<uint32_t>(input_dims[2]));
        // input width
        unit0.ocl_kernel.setArg(4, static_cast<uint32_t>(input_dims[3]));
        // input channel
        unit0.ocl_kernel.setArg(5, static_cast<uint32_t>(input_dims[1]));
        unit0.ocl_kernel.setArg(6, *((cl::Image *)input->GetHandle().base));

        auto &unit1               = execute_units_[1];
        int output_channel_blocks = UP_DIV(output_dims[1], 4);
        unit1.global_work_size    = {(uint32_t)output_dims[3] * output_channel_blocks,
                                  (uint32_t)output_dims[0] * output_dims[2]};
        unit1.local_work_size     = LocalWS2DDefault(unit1);
        unit1.ocl_kernel.setArg(0, unit1.global_work_size[0]);
        unit1.ocl_kernel.setArg(1, unit1.global_work_size[1]);
        unit1.ocl_kernel.setArg(2, *buffer_);
        unit1.ocl_kernel.setArg(3, *((cl::Image *)output->GetHandle().base));
        unit1.ocl_kernel.setArg(4, 4 * sizeof(int), begins_.data());
        unit1.ocl_kernel.setArg(5, 4 * sizeof(int), strides_.data());
        // input_width
        unit1.ocl_kernel.setArg(6, input_dims[3]);
        // input_width * input_height
        unit1.ocl_kernel.setArg(7, input_dims[3] * input_dims[2]);
        // input_width * input_height * input_channel
        unit1.ocl_kernel.setArg(8, input_dims[3] * input_dims[2] * input_dims[1]);
        // input_channel
        unit1.ocl_kernel.setArg(9, input_dims[1]);
        unit1.ocl_kernel.setArg(10, outputWH);
        // output_channel
        unit1.ocl_kernel.setArg(11, output_dims[1]);
    }
    return TNN_OK;
}

REGISTER_OPENCL_ACC(StrideSlice, LAYER_STRIDED_SLICE)

}  // namespace TNN_NS
