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
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

typedef enum {
    SPLITV_IMAGE = 0,
    SPLITV_UNITE = 1,
} SplitVType;

struct SplitVUnit {
    int axis;
    int begin;
    int end;
    SplitVType type;
};

class OpenCLSplitVLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLSplitVLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    std::vector<SplitVUnit> splitv_units_ = {};
    std::shared_ptr<cl::Buffer> buffer_ = nullptr;
    bool use_buffer_ = false;
};

Status OpenCLSplitVLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                  const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init SplitV Acc \n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "SplitV";

    auto layer_param = dynamic_cast<SplitVLayerParam *>(param);
    if (layer_param == nullptr) {
        LOGE("SplitVLayerParam is null!\n");
        return Status(TNNERR_MODEL_ERR, "SplitVLayerParam is null");
    }

    if (layer_param->axis != 1 && layer_param->axis != 2) {
        LOGE("axis=%d is not support in SplitV yet!\n", layer_param->axis);
        return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "the axis not support");
    }

    std::vector<int> slices = layer_param->slices;

    if (slices.size() != outputs.size()) {
        LOGE("invalid params in SplitV!\n");
        return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "invalid params in SplitV");
    }

    int begin   = 0;
    for (auto output : outputs) {
        auto output_dim = output->GetBlobDesc().dims;

        SplitVUnit unit;
        unit.axis  = layer_param->axis;
        int slice  = output_dim[unit.axis];
        unit.begin = begin;
        unit.end   = begin + slice;

        if (unit.axis == 1 && unit.begin % 4 != 0) {
            unit.type   = SPLITV_UNITE;
            use_buffer_ = true;
        } else {
            unit.type = SPLITV_IMAGE;
        }

        splitv_units_.push_back(unit);

        begin += slice;
    }

    execute_units_.clear();
    if (use_buffer_) {
        // use buffer, convert image to buffer first.
        OpenCLExecuteUnit exec_unit;
        ret = CreateExecuteUnit(exec_unit, "image_to_buffer", "ImageToNCHWBufferFLOAT");
        CHECK_TNN_OK(ret)
        execute_units_.push_back(exec_unit);
    }
    for (auto unit : splitv_units_) {
        OpenCLExecuteUnit exec_unit;
        if (SPLITV_IMAGE == unit.type) {
            ret = CreateExecuteUnit(exec_unit, "copy", "CopyImage");
        } else {
            ret = CreateExecuteUnit(exec_unit, "copy", "CopyBufferToImage");
        }
        CHECK_TNN_OK(ret)
        execute_units_.push_back(exec_unit);
    }

    return TNN_OK;
}

OpenCLSplitVLayerAcc::~OpenCLSplitVLayerAcc() {}

Status OpenCLSplitVLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("StrideSlice Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();

    auto input           = inputs[0];
    auto dims            = input->GetBlobDesc().dims;
    int exec_unit_idx    = 0;
    int input_dims_count = DimsVectorUtils::Count(input->GetBlobDesc().dims);
    if (use_buffer_) {
        // use buffer, convert to buffer.
        int type_size = sizeof(float);
        if (opencl_runtime->GetPrecision() != PRECISION_HIGH) {
            type_size = 2;
        }
        cl_int cl_ret;
        cl::Buffer *cl_buffer = new cl::Buffer(*opencl_runtime->Context(), (cl_mem_flags)CL_MEM_READ_WRITE,
                                               (cl::size_type)(input_dims_count * type_size), nullptr, &cl_ret);
        if (cl_ret != CL_SUCCESS) {
            CHECK_CL_SUCCESS(cl_ret)
            if (nullptr != cl_buffer)
                delete cl_buffer;
            return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
        }
        buffer_.reset(cl_buffer);

        auto &unit   = execute_units_[exec_unit_idx];
        uint32_t idx = SetExecuteUnit2DSizeInfoDefault(unit, dims);
        unit.ocl_kernel.setArg(idx++, *buffer_);
        // input_height
        unit.ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(dims, 2)));
        // input_width
        unit.ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(dims, 3)));
        // input_channel
        unit.ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(dims, 1)));
        unit.ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
        exec_unit_idx++;
    }

    int unit_idx = 0;
    for (auto output : outputs) {
        auto output_dims = output->GetBlobDesc().dims;
        int input_wh[]   = {DimsFunctionUtils::GetDim(dims, 3), DimsFunctionUtils::GetDim(dims, 2)};
        int output_wh[]  = {DimsFunctionUtils::GetDim(output_dims, 3), DimsFunctionUtils::GetDim(output_dims, 2)};

        auto &unit       = execute_units_[exec_unit_idx];
        auto splitv_unit = splitv_units_[unit_idx];

        if (splitv_unit.type == SPLITV_IMAGE) {
            // kernel: CopyImage
            int input_offset[] = {0, 0, 0, 0};
            if (splitv_unit.axis == 1) {
                // if the axis is 1, then offset need to div 4
                input_offset[1] = splitv_unit.begin / 4;
            } else {
                input_offset[splitv_unit.axis] = splitv_unit.begin;
            }
            int output_offset[] = {0, 0, 0, 0};
            // default set execution
            int idx = SetExecuteUnit2DSizeInfoDefault(unit, output_dims);
            unit.ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
            unit.ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
            unit.ocl_kernel.setArg(idx++, input_offset);
            unit.ocl_kernel.setArg(idx++, output_offset);
            unit.ocl_kernel.setArg(idx++, input_wh);
            unit.ocl_kernel.setArg(idx++, output_wh);
            unit.ocl_kernel.setArg(idx++, output_wh);
        } else {
            // kernel: CopyBufferToImage
            int input_offset[]             = {0, 0, 0, 0};
            input_offset[splitv_unit.axis] = splitv_unit.begin;
            int output_offset[]            = {0, 0, 0, 0};
            // stride: input_channel * input_height * input_width, input_height * input_width, input_width, 1
            int input_channel = DimsFunctionUtils::GetDim(dims, 1);
            int input_height = DimsFunctionUtils::GetDim(dims, 2);
            int input_width = DimsFunctionUtils::GetDim(dims, 3);
            int input_stride[] = {input_channel * input_height * input_width,
                                  input_height * input_width, input_width, 1};
            int idx = SetExecuteUnit2DSizeInfoDefault(unit, output_dims);
            unit.ocl_kernel.setArg(idx++, *buffer_);
            unit.ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
            unit.ocl_kernel.setArg(idx++, input_offset);
            unit.ocl_kernel.setArg(idx++, output_offset);
            unit.ocl_kernel.setArg(idx++, input_stride);
            unit.ocl_kernel.setArg(idx++, output_wh);
            unit.ocl_kernel.setArg(idx++, output_wh);
            unit.ocl_kernel.setArg(idx++, input_dims_count - 1);
        }
        exec_unit_idx++;
        unit_idx++;
    }
    return TNN_OK;
}

REGISTER_OPENCL_ACC(SplitV, LAYER_SPLITV)
REGISTER_OPENCL_LAYOUT(LAYER_SPLITV, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
