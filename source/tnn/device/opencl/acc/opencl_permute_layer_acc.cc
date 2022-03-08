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
    virtual std::vector<DataFormat> SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) override;

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

    if (permute_param->orders.size() <= 4) {
        dims_.resize(4);
        for (unsigned int i = 0; i < permute_param->orders.size(); ++i) {
            int dim    = permute_param->orders[i];
            dims_[dim] = i;
        }

        // pad permute order to 4 dims
        for (unsigned int i = permute_param->orders.size(); i < 4; ++i) {
            dims_[i] = i;
        }
    } else {
        dims_.resize(permute_param->orders.size());
        for (unsigned int i = 0; i < permute_param->orders.size(); ++i) {
            int dim    = permute_param->orders[i];
            dims_[dim] = i;
        }
    }

    std::string src_format = "Image", dst_format = "Image";
    std::string copy_program_name = "copy";
    src_format = dims_.size() == 5 ? "Image5D" : dims_.size() == 6 ? "Image6D" : src_format;
    copy_program_name = dims_.size() == 5 ? "copy_image_5d" : dims_.size() == 6 ? "copy_image_6d" : copy_program_name;
    dst_format = src_format;

    execute_units_.resize(2);
    // image->buffer
    {
        std::string kernel_name = "Copy" + src_format + "ToBuffer";
        ret = CreateExecuteUnit(execute_units_[0], copy_program_name, kernel_name, build_options_);
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    // buffer->image
    {
        std::string kernel_name = "CopyBufferTo" + dst_format;
        ret = CreateExecuteUnit(execute_units_[1], copy_program_name, kernel_name, build_options_);
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
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto input  = inputs[0];
    auto output = outputs[0];

    auto input_dims  = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;

    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();

    int output_n    = DimsFunctionUtils::GetDim(output_dims, 0);
    int output_c    = DimsFunctionUtils::GetDim(output_dims, 1);
    int output_count = 1;
    for (int i = 2; i < dims_.size(); ++i) {
        output_count *= DimsFunctionUtils::GetDim(output_dims, i);
    }

    int input_n     = DimsFunctionUtils::GetDim(input_dims, 0);
    int input_c     = DimsFunctionUtils::GetDim(input_dims, 1);
    int input_count = 1;
    for (int i = 2; i < dims_.size(); ++i) {
        input_count *= DimsFunctionUtils::GetDim(input_dims, i);
    }

    int size0          = UP_DIV(output_c, 4) * 4 * output_n * output_count;
    int size1          = UP_DIV(input_c, 4) * 4 * input_n * input_count;
    int blob_elem_size = opencl_runtime->GetPrecision() != PRECISION_HIGH ? 2 : 4;
    int blob_size      = std::max(size0, size1) * blob_elem_size;

    inter_buffer_           = std::make_shared<cl::Buffer>(*opencl_runtime->Context(), CL_MEM_READ_WRITE, blob_size);
    std::vector<int> offset(dims_.size(), 0);
    std::vector<int> output_stride(dims_.size());
    int count = 1;
    for (int i = 0; i < dims_.size(); ++i) {
        output_stride[dims_.size() - 1 - i] = count;
        count *= DimsFunctionUtils::GetDim(output_dims, dims_.size() - 1 - i);
    }
    std::vector<int> permute_input_stride(dims_.size());
    for (int i = 0; i < dims_.size(); ++i) {
        permute_input_stride[i] = output_stride[dims_[i]];
    }
    std::vector<int> input_size(dims_.size() - 2);
    std::vector<int> output_size(dims_.size() - 2);
    for (int i = 2; i < dims_.size(); ++i) {
        input_size[i - 2] = DimsFunctionUtils::GetDim(input_dims, i);
        output_size[i - 2] = DimsFunctionUtils::GetDim(output_dims, i);
    }

    if (dims_.size() == 4) {
        // dim-4 store w,h
        std::reverse(input_size.begin(), input_size.end());
        std::reverse(output_size.begin(), output_size.end());
    }

    std::vector<int> buffer_output_size(dims_.size());
    for (int i = 0; i < dims_.size(); ++i) {
        buffer_output_size[i] = DimsFunctionUtils::GetDim(input_dims, i);
    }
    // image->buffer
    {
        int idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], input_dims);
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, *inter_buffer_.get());
        execute_units_[0].ocl_kernel.setArg(idx++, offset.size() * sizeof(int), offset.data());
        execute_units_[0].ocl_kernel.setArg(idx++, offset.size() * sizeof(int), offset.data());
        execute_units_[0].ocl_kernel.setArg(idx++, input_size.size() * sizeof(int), input_size.data());
        execute_units_[0].ocl_kernel.setArg(idx++, permute_input_stride.size() * sizeof(int), permute_input_stride.data());
        execute_units_[0].ocl_kernel.setArg(idx++, buffer_output_size.size() * sizeof(int), buffer_output_size.data());
    }

    // buffer->image
    {
        int idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[1], output_dims);
        execute_units_[1].ocl_kernel.setArg(idx++, *inter_buffer_.get());
        execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
        execute_units_[1].ocl_kernel.setArg(idx++, offset.size() * sizeof(int), offset.data());
        execute_units_[1].ocl_kernel.setArg(idx++, offset.size() * sizeof(int), offset.data());
        execute_units_[1].ocl_kernel.setArg(idx++, output_stride.size() * sizeof(int), output_stride.data());
        execute_units_[1].ocl_kernel.setArg(idx++, output_size.size() * sizeof(int), output_size.data());
        execute_units_[1].ocl_kernel.setArg(idx++, output_size.size() * sizeof(int), output_size.data());
        execute_units_[1].ocl_kernel.setArg(idx++, std::max(size0, size1) - 1);
    }

    return TNN_OK;
}

std::vector<DataFormat> OpenCLPermuteLayerAcc::SupportDataFormat(DataType data_type,
                                                                 int dims_size,
                                                                 BlobType blob_type) {
    std::vector<DataFormat> support_list;
    if (data_type == DATA_TYPE_INT32) {
        support_list.push_back(DATA_FORMAT_NHC4W4);
    } else if (dims_size >= 2 && dims_size <= 6) { // only support up to 6 dims
        support_list.push_back(DATA_FORMAT_NHC4W4);
    }
    return support_list;
}

REGISTER_OPENCL_ACC(Permute, LAYER_PERMUTE)
REGISTER_OPENCL_LAYOUT(LAYER_PERMUTE, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
