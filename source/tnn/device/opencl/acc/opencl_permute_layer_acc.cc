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

namespace TNN_NS {

class OpenCLPermuteLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLPermuteLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    std::shared_ptr<cl::Buffer> inter_buffer_ = nullptr;
    std::vector<int> dims_                    = {};
};

Status OpenCLPermuteLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                   const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Permute Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "Permute";

    PermuteLayerParam *permute_param = dynamic_cast<PermuteLayerParam *>(param);
    CHECK_PARAM_NULL(permute_param);

    if (permute_param->orders.size() != 4) {
        LOGE("permute order size need to be 4!\n");
        return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "permute order size need to be 4!");
    }
    dims_.resize(4);
    for (unsigned int i = 0; i < permute_param->orders.size(); ++i) {
        int dim    = permute_param->orders[i];
        dims_[dim] = i;
    }

    execute_units_.resize(2);
    // image->buffer
    {
        ret = CreateExecuteUnit(execute_units_[0], "copy", "CopyImageToBuffer");
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    // buffer->image
    {
        ret = CreateExecuteUnit(execute_units_[1], "copy", "CopyBufferToImage");
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    return TNN_OK;
}

OpenCLPermuteLayerAcc::~OpenCLPermuteLayerAcc() {}

Status OpenCLPermuteLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Permute Acc Reshape\n");
    auto input  = inputs[0];
    auto output = outputs[0];

    auto input_dims  = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;

    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();

    int size0          = UP_DIV(output_dims[1], 4) * 4 * output_dims[0] * output_dims[2] * output_dims[3];
    int size1          = UP_DIV(input_dims[1], 4) * 4 * input_dims[0] * input_dims[2] * input_dims[3];
    int blob_elem_size = opencl_runtime->GetPrecision() != PRECISION_HIGH ? 2 : 4;
    int blob_size      = std::max(size0, size1) * blob_elem_size;

    inter_buffer_        = std::make_shared<cl::Buffer>(*opencl_runtime->Context(), CL_MEM_READ_WRITE, blob_size);
    int offset[4]        = {0, 0, 0, 0};
    int output_stride[4] = {output_dims[1] * output_dims[2] * output_dims[3], output_dims[2] * output_dims[3],
                            output_dims[3], 1};
    int permute_input_stride[4];
    for (int i = 0; i < dims_.size(); ++i) {
        permute_input_stride[i] = output_stride[dims_[i]];
    }
    int input_wh[2]  = {input_dims[3], input_dims[2]};
    int output_wh[2] = {output_dims[3], output_dims[2]};

    // image->buffer
    {
        int idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], input_dims);
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, *inter_buffer_.get());
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(offset), offset);
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(offset), offset);
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(input_wh), input_wh);
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(permute_input_stride), permute_input_stride);
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(int) * 4, input_dims.data());
    }

    // buffer->image
    {
        int idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[1], output_dims);
        execute_units_[1].ocl_kernel.setArg(idx++, *inter_buffer_.get());
        execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
        execute_units_[1].ocl_kernel.setArg(idx++, sizeof(offset), offset);
        execute_units_[1].ocl_kernel.setArg(idx++, sizeof(offset), offset);
        execute_units_[1].ocl_kernel.setArg(idx++, sizeof(output_stride), output_stride);
        execute_units_[1].ocl_kernel.setArg(idx++, sizeof(output_wh), output_wh);
        execute_units_[1].ocl_kernel.setArg(idx++, sizeof(output_wh), output_wh);
        execute_units_[1].ocl_kernel.setArg(idx++, std::max(size0, size1) - 1);
    }

    return TNN_OK;
}

REGISTER_OPENCL_ACC(Permute, LAYER_PERMUTE)

}  // namespace TNN_NS
