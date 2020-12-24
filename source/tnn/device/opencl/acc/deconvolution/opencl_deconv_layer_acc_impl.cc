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

#include "tnn/device/opencl/acc/deconvolution/opencl_deconv_layer_common_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

Status OpenCLDeconvLayerAccImpl::Init(Context *context, LayerParam *param, LayerResource *resource,
                                      const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;

    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param);
    if (nullptr == conv_param) {
        LOGE("invalid deconv param!\n");
        return Status(TNNERR_NULL_PARAM, "invalid deconv param");
    }

    //interpreter deconv 2d param with name info.
    deconv_params_.kernel_x        = conv_param->kernels[0];
    deconv_params_.kernel_y        = conv_param->kernels[1];
    deconv_params_.pad_x           = conv_param->pads[0];
    deconv_params_.pad_y           = conv_param->pads[2];
    deconv_params_.stride_x        = conv_param->strides[0];
    deconv_params_.stride_y        = conv_param->strides[1];
    deconv_params_.dilation_x      = conv_param->dialations[0];
    deconv_params_.dilation_y      = conv_param->dialations[1];
    deconv_params_.pad_type        = conv_param->pad_type;
    deconv_params_.group           = conv_param->group;
    deconv_params_.has_bias        = conv_param->bias;
    deconv_params_.activation_type = conv_param->activation_type;

    deconv_params_.input_channel  = inputs[0]->GetBlobDesc().dims[1];
    deconv_params_.output_channel = outputs[0]->GetBlobDesc().dims[1];

    if ((deconv_params_.group <= 0 || deconv_params_.input_channel % deconv_params_.group != 0)) {
        LOGE("invalid group size in DeConv layer!\n");
        return Status(TNNERR_LAYER_ERR, "invalid group size in DeConv layer");
    }

    ConvLayerResource *deconv_resource = dynamic_cast<ConvLayerResource *>(resource);
    if (nullptr == deconv_resource) {
        LOGE("invalid deconv resource!\n");
        return Status(TNNERR_NULL_PARAM, "invalid deconv resource");
    }
    // get weights
    if (deconv_resource->filter_handle.GetDataType() == DATA_TYPE_FLOAT) {
        //get float pointer from raw buffer.
        float *weights_data_ptr = deconv_resource->filter_handle.force_to<float *>();
        if (weights_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertWeights(weights_data_ptr);
        CHECK_TNN_OK(ret)
    } else {
        //if filter handle is half, need convert to float first.
        auto float_data_ptr = GetFloatFromRawBuffer(deconv_resource->filter_handle);  // handle the memory
        if (float_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertWeights(float_data_ptr.get());
        CHECK_TNN_OK(ret)
    }

    //convert bias
    ret = ConvertChannelWeights(deconv_resource->bias_handle, ocl_bias_, deconv_params_.output_channel,
                                deconv_params_.has_bias);
    return ret;
}

OpenCLDeconvLayerAccImpl::~OpenCLDeconvLayerAccImpl() {}

Status OpenCLDeconvLayerAccImpl::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Deconv Impl Acc Reshape\n");
    auto input       = inputs[0];
    auto output      = outputs[0];
    auto input_dims  = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;

    // output_channel/4
    const int output_channel_blocks = UP_DIV(output_dims[1], 4);
    const int input_channel_blocks = UP_DIV(input_dims[1], 4);

    int pad_x_trans = deconv_params_.kernel_x - 1 - deconv_params_.pad_x;
    int pad_y_trans = deconv_params_.kernel_y - 1 - deconv_params_.pad_y;

    const int align_width  = deconv_params_.stride_x - 1 - pad_x_trans;
    const int align_height = deconv_params_.stride_y - 1 - pad_y_trans;

    //input_width, input_height
    int input_imageshape[2]  = {input_dims[3], input_dims[2]};
    //output_width, output_height
    int output_imageshape[2] = {output_dims[3], output_dims[2]};
    int stride_shape[2]      = {deconv_params_.stride_x, deconv_params_.stride_y};
    int align_shape[2]       = {align_width, align_height};
    int padding_shape[2]     = {pad_x_trans, pad_y_trans};
    int kernel_shape[2]      = {deconv_params_.kernel_x, deconv_params_.kernel_y};

    bool is_deconv_4x4_s2_p1_wb4    =
        (CT_DECONV_DEPTHWISE != deconv_type_ &&
        deconv_params_.kernel_x == 4 && deconv_params_.kernel_y == 4 &&
        deconv_params_.stride_x == 2 && deconv_params_.stride_y == 2 &&
        deconv_params_.pad_x == 1 && deconv_params_.pad_y == 1 &&
        deconv_params_.dilation_x == 1 && deconv_params_.dilation_y == 1 && output_dims[3] % 4 == 0);

    // output_width * output_channel/4, batch * output_height
    execute_units_[0].global_work_size = {
            static_cast<uint32_t>(output_dims[3] * UP_DIV(output_dims[1], 4)),
            static_cast<uint32_t>(output_dims[0] * output_dims[2])};

    if (is_deconv_4x4_s2_p1_wb4) {
        // output_width/4 * output_channel/4, batch * output_height
        execute_units_[0].global_work_size[0] =
            static_cast<uint32_t>(UP_DIV(output_dims[3], 4) * UP_DIV(output_dims[1], 4));
    }

    execute_units_[0].local_work_size = LocalWS2DDefault(execute_units_[0]);

    uint32_t idx = 0;
    for (auto gws : execute_units_[0].global_work_size) {
        execute_units_[0].ocl_kernel.setArg(idx++, gws);
    }

    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_weights_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_bias_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(input_imageshape), input_imageshape);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(output_imageshape), output_imageshape);

    if (is_deconv_4x4_s2_p1_wb4) {
        execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int32_t>(UP_DIV(output_dims[3], 4)));
    } else {
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(stride_shape), stride_shape);
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(align_shape), align_shape);
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(padding_shape), padding_shape);
        execute_units_[0].ocl_kernel.setArg(idx++, sizeof(kernel_shape), kernel_shape);
        execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int32_t>(kernel_shape[0] * kernel_shape[1]));
    }

    SetExtraKernelParameters(idx, inputs, outputs);

    return TNN_OK;
}

Status OpenCLDeconvLayerAccImpl::ConvertWeights(float *weights_data_ptr) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();

    float *wdata_ptr = weights_data_ptr;
    std::shared_ptr<float> weights_data_trans;
    if (CT_DECONV_COMMON == deconv_type_) {
        if (deconv_params_.group == 1) {
            int element_size = deconv_params_.output_channel * deconv_params_.input_channel * deconv_params_.kernel_y *
                               deconv_params_.kernel_x;
            weights_data_trans.reset(new float[element_size], [](float *p) { delete[] p; });
            IOHW2OIHW<float, int>(weights_data_ptr, weights_data_trans.get(), deconv_params_.output_channel,
                                  deconv_params_.input_channel, deconv_params_.kernel_y, deconv_params_.kernel_x);
            wdata_ptr = weights_data_trans.get();
        } else {
            //special for group deconv
            int element_size = deconv_params_.output_channel * deconv_params_.input_channel * deconv_params_.kernel_y *
                               deconv_params_.kernel_x;
            weights_data_trans.reset(new float[element_size], [](float *p) { delete[] p; });
            GROUP_PADDING<float, int>(weights_data_ptr, weights_data_trans.get(), deconv_params_.group,
                                      deconv_params_.output_channel, deconv_params_.input_channel,
                                      deconv_params_.kernel_y, deconv_params_.kernel_x, GIOHW);
            wdata_ptr = weights_data_trans.get();
        }
    }

    // copy weights data into clBuffer
    DimsVector filter_shape;
    if (CT_DECONV_DEPTHWISE == deconv_type_) {
        filter_shape = {1, deconv_params_.output_channel, deconv_params_.kernel_y, deconv_params_.kernel_x};
    } else {
        filter_shape = {deconv_params_.output_channel, deconv_params_.input_channel, deconv_params_.kernel_y,
                        deconv_params_.kernel_x};
    }

    uint32_t bytes_size = DimsVectorUtils::Count(filter_shape) * sizeof(float);
    shared_ptr<OpenCLMemory> weight_buffer(new OpenCLMemory(TNN_CL_BUFFER));
    cl_int ret = CL_SUCCESS;
    cl::Buffer buffer(*opencl_runtime->Context(), (cl_mem_flags)(CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR),
                      (cl::size_type)bytes_size, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL Deconv malloc memory falied");
    }
    weight_buffer->SetData(&buffer);
    auto weight_clbuffer_ptr = ocl_context_->CommandQueue()->enqueueMapBuffer(buffer, true, CL_MAP_WRITE, 0, bytes_size,
                                                                              nullptr, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMMAP_ERROR, "OpenCL Deconv  MemMap failed");
    }
    memcpy(weight_clbuffer_ptr, wdata_ptr, bytes_size);
    ret = ocl_context_->CommandQueue()->enqueueUnmapMemObject(buffer, weight_clbuffer_ptr);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL Deconv MemUnMap falied");
    }

    // create ocl_weights_
    DimsVector filter_imageshape;
    if (CT_DECONV_DEPTHWISE == deconv_type_) {
        filter_imageshape = {deconv_params_.kernel_x * deconv_params_.kernel_y,
                            (int)(UP_DIV(deconv_params_.output_channel, 4))};
    } else {
        filter_imageshape = {deconv_params_.input_channel, (int)(UP_DIV(deconv_params_.output_channel, 4) *
                                                                 deconv_params_.kernel_x * deconv_params_.kernel_y)};
    }
    cl_channel_type data_type = CL_FLOAT;
    if (opencl_runtime->GetPrecision() != PRECISION_HIGH)
        data_type = CL_HALF_FLOAT;
    cl::Image2D *image =
        new cl::Image2D(*opencl_runtime->Context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, data_type),
                        filter_imageshape[0], filter_imageshape[1], 0, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        if (nullptr != image)
            delete image;
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory falied");
    }
    ocl_weights_.reset(new OpenCLMemory(TNN_CL_IMAGE));
    ocl_weights_->SetData(image, true);

    // transfer from clBuffer to clImage
    ImageBufferConvertor convertor(opencl_runtime, ocl_context_->CommandQueue());
    OpenCLBufferFormat buffer_format = CONV2D_FILTER;
    if (CT_DECONV_DEPTHWISE == deconv_type_) {
        buffer_format = DW_CONV2D_FILTER;
    }
    return convertor.ConvertBufferToImage(weight_buffer.get(), buffer_format, filter_shape, ocl_weights_.get(), true);
}

void OpenCLDeconvLayerAccImpl::SetExtraKernelParameters(uint32_t idx, const std::vector<Blob *> &inputs,
                                                        const std::vector<Blob *> &outputs) {}

#if TNN_PROFILE
double OpenCLDeconvLayerAccImpl::GetFlops() {
    return 2.0 * DimsVectorUtils::Count(input_dims_) * output_dims_[1] / deconv_params_.group * deconv_params_.kernel_x *
           deconv_params_.kernel_y / 1000.0 / 1000.0;
}
#endif

}  // namespace TNN_NS
