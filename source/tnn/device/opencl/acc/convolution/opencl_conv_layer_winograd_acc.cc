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

#include "tnn/device/opencl/acc/convolution/opencl_conv_layer_winograd_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"

#include "tnn/utils/winograd_generator.h"

namespace TNN_NS {

#define UNIT 2

bool OpenCLConvLayerWinogradAcc::IsPrefered(const ConvLayerParam *param, const std::vector<Blob *> &,
                                       const std::vector<Blob *> &) {
    if (!param) {
        return false;
    }
    return param->group == 1 && param->kernels[0] == 3 && param->kernels[1] == 3 && param->dialations[0] == 1 && 
            param->dialations[1] == 1 && param->strides[0] == 1 && param->strides[1] == 1;
}

Status OpenCLConvLayerWinogradAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                   const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Conv Winograd Acc \n");

    conv_type_ = CT_CONV_WINOGRAD;
    op_name_   = "Conv_Winograd";

    Status ret = OpenCLConvLayerAccImpl::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    ConvLayerResource *conv_resource = dynamic_cast<ConvLayerResource *>(resource_);

    auto input_dims          = inputs[0]->GetBlobDesc().dims;
    auto output_dims         = outputs[0]->GetBlobDesc().dims;
    const int input_channel  = input_dims[1];
    const int output_channel = output_dims[1];

    //convert filter
    ret = ConvertWinogradTransformWeigths(conv_resource->filter_handle, ocl_weights_, input_channel, output_channel);
    CHECK_TNN_OK(ret)

    //convert bias
    ret = ConvertChannelWeights(conv_resource->bias_handle, ocl_bias_, conv_params_.output_channel,
                                conv_params_.has_bias, false);
    CHECK_TNN_OK(ret)

    ret = AllocateWinogradMatrixVAndM(input_dims, output_dims);
    CHECK_TNN_OK(ret)

    //create kernels 
    execute_units_.resize(3);
    std::string program_name = "winograd";
    std::string kernel_name;
    //kernel WinogradTransformSource
    kernel_name = "WinogradTransformSource";
    ret         = CreateExecuteUnit(execute_units_[0], program_name, kernel_name);    
    CHECK_TNN_OK(ret)
    //kernel MatrixInnerProduct
    kernel_name = "MatrixInnerProduct";
    ret         = CreateExecuteUnit(execute_units_[1], program_name, kernel_name);    
    CHECK_TNN_OK(ret)
    //kernel WinogradTransformDest 
    kernel_name = "WinogradTransformDest";
    ret         = CreateExecuteUnit(execute_units_[2], program_name, kernel_name);
    CHECK_TNN_OK(ret)

    return TNN_OK;
}

OpenCLConvLayerWinogradAcc::~OpenCLConvLayerWinogradAcc() {}

Status OpenCLConvLayerWinogradAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status OpenCLConvLayerWinogradAcc::ConvertWinogradTransformWeigths(RawBuffer &raw_handle, shared_ptr<OpenCLMemory> &ocl_handle, int input_channel, int output_channel) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    const int kernel_size       = conv_params_.kernel_x;
    int unit_output = UNIT;
    int unit_input = UNIT + kernel_size - 1;
    WinogradGenerator generator(unit_output, kernel_size, 1.0f);
    auto transform_weight =  generator.allocTransformWeight(output_channel, input_channel, kernel_size, kernel_size, 4, 4);
    // if filter handle is half, need convert to float first.
    auto filter_data = GetFloatFromRawBuffer(raw_handle);
    if (filter_data == nullptr) {
        return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
    }
    generator.transformWeight(transform_weight, filter_data.get(), output_channel, input_channel, kernel_size, kernel_size);

    auto dims = std::get<1>(transform_weight);
    
    cl_int ret = CL_SUCCESS;
    cl::Buffer buffer(*opencl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                      DimsVectorUtils::Count(dims) * sizeof(float), nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL Conv malloc memory falied");
    }
    auto transform_weight_clbuffer_ptr = ocl_context_->CommandQueue()->enqueueMapBuffer(
        buffer, true, CL_MAP_WRITE, 0, DimsVectorUtils::Count(dims) * sizeof(float), nullptr, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMMAP_ERROR, "OpenCL Conv MemMap failed");
    }
    memcpy(transform_weight_clbuffer_ptr, std::get<0>(transform_weight).get(), DimsVectorUtils::Count(dims) * sizeof(float));
    ret = ocl_context_->CommandQueue()->enqueueUnmapMemObject(buffer, transform_weight_clbuffer_ptr);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL Conv MemUnMap falied");
    }

    cl_channel_type data_type = CL_FLOAT;
    if (opencl_runtime->GetPrecision() != PRECISION_HIGH)
        data_type = CL_HALF_FLOAT;
    int image_height = DimsVectorUtils::Count(dims, 0, 2);
    int image_width = DimsVectorUtils::Count(dims, 2);
    cl::Image2D *image =
        new cl::Image2D(*opencl_runtime->Context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, data_type),
                        image_width, image_height, 0, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        if (nullptr != image)
            delete image;
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL Conv malloc memory falied");
    }
    ocl_weights_.reset(new OpenCLMemory(TNN_CL_IMAGE));
    ocl_weights_->SetData(image, true);
    CopyBufferToImage(opencl_runtime, ocl_context_, buffer, *image, image_width, image_height, true);
    return TNN_OK;
}

Status OpenCLConvLayerWinogradAcc::AllocateWinogradMatrixVAndM(DimsVector input_dims, DimsVector output_dims) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    cl_channel_type data_type = CL_FLOAT;
    if (opencl_runtime->GetPrecision() != PRECISION_HIGH)
        data_type = CL_HALF_FLOAT;
    cl_int ret = CL_SUCCESS;
    return TNN_OK;
}

std::vector<uint32_t> OpenCLConvLayerWinogradAcc::Conv2dWinogradLocalWS2D(std::vector<uint32_t> &gws,
                                                                const uint32_t max_workgroup_size) {
    std::vector<uint32_t> lws(2, 1);
    return lws;
}

}  // namespace TNN_NS
