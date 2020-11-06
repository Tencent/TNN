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

#include "tnn/device/opencl/acc/convolution/opencl_conv_layer_acc_impl.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

// (inputs + weights + outputs) * array_size * sizeof(float)
static const uint32_t kernel_cache_size = (4 + 4 + 4) * 4 * 4;
// magic number
static const uint32_t lws_limit = 128;

OpenCLConvLayerAccImpl::OpenCLConvLayerAccImpl() {
    conv_type_ = CT_CONV_COMMON;
    op_name_   = "Conv";
}

Status OpenCLConvLayerAccImpl::Init(Context *context, LayerParam *param, LayerResource *resource,
                                    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param);
    if (nullptr == conv_param) {
        LOGE("invalid conv param!\n");
        return Status(TNNERR_NULL_PARAM, "invalid conv param");
    }

    // interpreter conv 2d param with name info.
    conv_params_.kernel_x        = conv_param->kernels[0];
    conv_params_.kernel_y        = conv_param->kernels[1];
    conv_params_.pad_x           = conv_param->pads[0];
    conv_params_.pad_y           = conv_param->pads[2];
    conv_params_.stride_x        = conv_param->strides[0];
    conv_params_.stride_y        = conv_param->strides[1];
    conv_params_.dilation_x      = conv_param->dialations[0];
    conv_params_.dilation_y      = conv_param->dialations[1];
    conv_params_.pad_type        = conv_param->pad_type;
    conv_params_.group           = conv_param->group;
    conv_params_.has_bias        = conv_param->bias;
    conv_params_.activation_type = conv_param->activation_type;

    conv_params_.input_channel  = inputs[0]->GetBlobDesc().dims[1];
    conv_params_.output_channel = outputs[0]->GetBlobDesc().dims[1];

    if ((conv_params_.group <= 0 || conv_params_.input_channel % conv_params_.group != 0)) {
        LOGE("invalid group size in Conv layer!\n");
        return Status(TNNERR_LAYER_ERR, "invalid group size in Conv layer");
    }

    // depthwise kernel use 2d ndragne.
    if (CT_CONV_DEPTHWISE == conv_type_) {
        run_3d_ndrange_ = false;
    }

    return TNN_OK;
}

OpenCLConvLayerAccImpl::~OpenCLConvLayerAccImpl() {}

Status OpenCLConvLayerAccImpl::AllocateWeightsBias(LayerResource *resource) {
    Status ret                       = TNN_OK;
    ConvLayerResource *conv_resource = dynamic_cast<ConvLayerResource *>(resource);
    if (nullptr == conv_resource) {
        LOGE("invalid conv resource!\n");
        return Status(TNNERR_NULL_PARAM, "invalid conv resource");
    }
    // get weights
    if (conv_resource->filter_handle.GetDataType() == DATA_TYPE_FLOAT) {
        // get float pointer from raw buffer.
        float *weights_data_ptr = conv_resource->filter_handle.force_to<float *>();
        if (weights_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertWeights(weights_data_ptr);
        CHECK_TNN_OK(ret)
    } else {
        // if filter handle is half, need convert to float first.
        auto float_data_ptr = GetFloatFromRawBuffer(conv_resource->filter_handle);
        if (float_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertWeights(float_data_ptr.get());
        CHECK_TNN_OK(ret)
    }

    // convert bias
    ret = ConvertChannelWeights(conv_resource->bias_handle, ocl_bias_, conv_params_.output_channel,
                                conv_params_.has_bias, false, use_buffer_);
    return ret;
}

// convert weights will copy data to buffer, then:
// if use clBuffer weigths for kernel, will convert buffer to buffer with target format.
// if use clImage weights for kernel, will convert buffer to image with target format.
Status OpenCLConvLayerAccImpl::ConvertWeights(float *weights_data_ptr) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    float *wdata_ptr              = weights_data_ptr;
    std::shared_ptr<float> weights_data_trans;
    // special for group conv
    if (CT_CONV_COMMON == conv_type_ && conv_params_.group > 1) {
        int element_size =
            conv_params_.output_channel * conv_params_.input_channel * conv_params_.kernel_y * conv_params_.kernel_x;
        weights_data_trans.reset(new float[element_size], [](float *p) { delete[] p; });
        GROUP_PADDING<float, int>(weights_data_ptr, weights_data_trans.get(), conv_params_.group,
                                  conv_params_.output_channel, conv_params_.input_channel, conv_params_.kernel_y,
                                  conv_params_.kernel_x, GOIHW);
        wdata_ptr = weights_data_trans.get();
    }

    // copy weights data into clBuffer
    DimsVector filter_shape;
    if (CT_CONV_DEPTHWISE == conv_type_) {
        filter_shape = {1, conv_params_.output_channel, conv_params_.kernel_y, conv_params_.kernel_x};
    } else {
        filter_shape = {conv_params_.output_channel, conv_params_.input_channel, conv_params_.kernel_y,
                        conv_params_.kernel_x};
    }

    shared_ptr<OpenCLMemory> weight_memory(new OpenCLMemory(TNN_CL_BUFFER));
    cl_int ret = CL_SUCCESS;
    cl::Buffer buffer(*opencl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                      DimsVectorUtils::Count(filter_shape) * sizeof(float), nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL Conv malloc memory falied");
    }
    weight_memory->SetData(&buffer);
    auto weight_clbuffer_ptr = ocl_context_->CommandQueue()->enqueueMapBuffer(
        buffer, true, CL_MAP_WRITE, 0, DimsVectorUtils::Count(filter_shape) * sizeof(float), nullptr, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMMAP_ERROR, "OpenCL Conv MemMap failed");
    }
    memcpy(weight_clbuffer_ptr, wdata_ptr, DimsVectorUtils::Count(filter_shape) * sizeof(float));
    ret = ocl_context_->CommandQueue()->enqueueUnmapMemObject(buffer, weight_clbuffer_ptr);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL Conv MemUnMap falied");
    }

    // create ocl_weights_
    if (use_buffer_) {
        // create weights use clBuffer
        DimsVector filter_buffershape;
        filter_buffershape = {ROUND_UP(conv_params_.output_channel, 4), ROUND_UP(conv_params_.input_channel, 4),
                              conv_params_.kernel_y, conv_params_.kernel_x};
        ocl_weights_.reset(new OpenCLMemory(TNN_CL_BUFFER));
        size_t type_size = sizeof(float);
        if (opencl_runtime->GetPrecision() != PRECISION_HIGH)
            type_size = 2;
        cl::Buffer *weights_clbuffer =
            new cl::Buffer(*opencl_runtime->Context(), CL_MEM_READ_WRITE,
                           DimsVectorUtils::Count(filter_buffershape) * type_size, nullptr, &ret);
        if (ret != CL_SUCCESS) {
            CHECK_CL_SUCCESS(ret)
            if (nullptr != weights_clbuffer)
                delete weights_clbuffer;
            return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL Conv malloc memory falied");
        }
        ocl_weights_->SetData(weights_clbuffer, true);

        // transfer from clBuffer to clBuffer
        ImageBufferConvertor convertor(opencl_runtime, ocl_context_->CommandQueue());
        return convertor.ConvertBufferToBuffer(weight_memory.get(), CONV2D_FILTER, filter_shape, ocl_weights_.get(),
                                               true);
    } else {
        // create weights use clImage
        DimsVector filter_imageshape;
        if (CT_CONV_DEPTHWISE == conv_type_) {
            filter_imageshape = {conv_params_.kernel_x * conv_params_.kernel_y,
                                 (int)(UP_DIV(conv_params_.output_channel, 4))};  // {w,h}
        } else {
            filter_imageshape = {conv_params_.input_channel, (int)(UP_DIV(conv_params_.output_channel, 4) *
                                                                   conv_params_.kernel_x * conv_params_.kernel_y)};
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
            return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL Conv malloc memory falied");
        }
        ocl_weights_.reset(new OpenCLMemory(TNN_CL_IMAGE));
        ocl_weights_->SetData(image, true);

        // transfer from clBuffer to clImage
        ImageBufferConvertor convertor(opencl_runtime, ocl_context_->CommandQueue());
        OpenCLBufferFormat buffer_format = CONV2D_FILTER;
        if (CT_CONV_DEPTHWISE == conv_type_) {
            buffer_format = DW_CONV2D_FILTER;
        }
        return convertor.ConvertBufferToImage(weight_memory.get(), buffer_format, filter_shape, ocl_weights_.get(),
                                              true);
    }
}

#if TNN_PROFILE
double OpenCLConvLayerAccImpl::GetFlops() {
    return 2.0 * DimsVectorUtils::Count(output_dims_) * input_dims_[1] / conv_params_.group * conv_params_.kernel_x *
           conv_params_.kernel_y / 1000.0 / 1000.0;
}
#endif

// local size 2d calculate, special for conv default.
std::vector<uint32_t> OpenCLConvLayerAccImpl::Conv2dCommonLocalWS2D(std::vector<uint32_t> &gws,
                                                                    const uint32_t max_workgroup_size,
                                                                    const uint32_t subgroup_size) {
    uint32_t compute_units = OpenCLRuntime::GetInstance()->DeviceComputeUnits();

    std::vector<uint32_t> lws;
    lws.clear();

    if (ADRENO == gpu_info_.type) {
        lws = AdrenoLocalSize2D(gws, gpu_info_, compute_units, max_workgroup_size, subgroup_size);
    }

    return lws;
}

// local size 3d calculate, special for conv default.
std::vector<uint32_t> OpenCLConvLayerAccImpl::Conv2dCommonLocalWS3DKernel3x3(std::vector<uint32_t> &gws,
                                                                    const uint32_t kernel_size,
                                                                    const uint32_t max_workgroup_size) {
    uint32_t compute_units = std::max<uint32_t>(OpenCLRuntime::GetInstance()->DeviceComputeUnits() / 2, 1);
    uint64_t cache_size    = OpenCLRuntime::GetInstance()->DeviceGlobalMemeryCacheSize();
    const uint32_t base    = std::max<uint32_t>(std::min<uint32_t>(cache_size / g_base_gpu_mem_cachesize, 4), 1);
    std::vector<uint32_t> lws(3, 1);
    if (max_workgroup_size > 0) {
        lws[1] = std::min<uint32_t>(gws[1], max_workgroup_size);
        lws[0] = std::min<uint32_t>(std::min<uint32_t>(gws[0], base), max_workgroup_size / lws[1]);
        const uint32_t lws_size = lws[0] * lws[1];

        lws[2] = std::min<uint32_t>(ROUND_UP(cache_size / kernel_cache_size / lws_size / compute_units, base), gws[2]);
        if (lws[2] == 0) {
            lws[2] = std::min<uint32_t>(gws[2], base);
        }
        lws[2] = std::max<uint32_t>(std::min<uint32_t>(lws[2], max_workgroup_size / lws_size), 1);
    }

    LOGD("compute_units : %d , max_workgroup_size : %d\n", compute_units, max_workgroup_size);
    LOGD("layer: %s conv_common [%d, %d, %d] -- [%d, %d, %d] \n", layer_name_.c_str(), gws[0], gws[1], gws[2], lws[0],
         lws[1], lws[2]);
    return lws;
}

// local size 3d calculate, special for conv default.
std::vector<uint32_t> OpenCLConvLayerAccImpl::Conv2dCommonLocalWS3DGeneral(std::vector<uint32_t> &gws,
                                                                    const uint32_t kernel_size,
                                                                    const uint32_t max_workgroup_size) {

    uint32_t compute_units = OpenCLRuntime::GetInstance()->DeviceComputeUnits();
    uint64_t cache_size    = OpenCLRuntime::GetInstance()->DeviceGlobalMemeryCacheSize();
    const uint32_t base    = std::max<uint32_t>(cache_size / g_base_gpu_mem_cachesize, 1);
    LOGD("cache_size: %d\n", (int)cache_size);
    std::vector<uint32_t> lws(3, 1);
    if (max_workgroup_size > 0) {
        lws[1] = std::min<uint32_t>(gws[1], max_workgroup_size);
        lws[0] = gws[0] / 4;
        if (lws[0] == 0) {
            lws[0] = gws[0];
        }
        lws[0]                  = std::min<uint32_t>(lws[0], max_workgroup_size / lws[1]);
        const uint32_t lws_size = lws[0] * lws[1];
        lws[2] =
            std::min<uint32_t>((cache_size / kernel_cache_size / kernel_size / lws_size / compute_units) * 8, gws[2]);
        if (lws[2] == 0) {
            if (gws[2] < lws_limit) {
                lws[2] = gws[2];
            } else {
                lws[2] = base;
            }
        }
        lws[2] = std::max<uint32_t>(std::min<uint32_t>(lws[2], max_workgroup_size / lws_size), 1);
    }

    LOGD("compute_units : %d , max_workgroup_size : %d\n", compute_units, max_workgroup_size);
    LOGD("layer: %s conv_common [%d, %d, %d] -- [%d, %d, %d] \n", layer_name_.c_str(), gws[0], gws[1], gws[2], lws[0],
         lws[1], lws[2]);
    return lws;
}

}  // namespace TNN_NS
