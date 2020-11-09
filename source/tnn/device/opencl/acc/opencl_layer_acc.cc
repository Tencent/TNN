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
#include "tnn/utils/string_utils_inner.h"

namespace TNN_NS {

//#define LOCAL_SIZE_FINE_TUNE

Status OpenCLLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    AbstractLayerAcc::Init(context, param, resource, inputs, outputs);

    param_       = param;
    resource_    = resource;
    layer_name_  = param->name;
    ocl_context_ = dynamic_cast<OpenCLContext *>(context);
    if (ocl_context_ == nullptr) {
        return Status(TNNERR_NULL_PARAM, "OpenCL Context Convert failed");
    }
    execute_units_.resize(1);

    if (context->GetPrecision() != PRECISION_HIGH) {
        LOGD("OpenCL Blob Pricision is Half!\n");
        for (auto blob : inputs) {
            blob->GetBlobDesc().data_type = DATA_TYPE_HALF;
        }
        for (auto blob : outputs) {
            blob->GetBlobDesc().data_type = DATA_TYPE_HALF;
        }
    } else {
        LOGD("OpenCL Blob Pricision is Float!\n");
        for (auto blob : inputs) {
            blob->GetBlobDesc().data_type = DATA_TYPE_FLOAT;
        }
        for (auto blob : outputs) {
            blob->GetBlobDesc().data_type = DATA_TYPE_FLOAT;
        }
    }

    input_dims_  = inputs[0]->GetBlobDesc().dims;
    output_dims_ = outputs[0]->GetBlobDesc().dims;

    ConfigKernelStrategy();

    return TNN_OK;
}

OpenCLLayerAcc::~OpenCLLayerAcc() {}

Status OpenCLLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
#if defined(LOCAL_SIZE_FINE_TUNE) && TNN_PROFILE
    auto execute_unit_org                                 = execute_units_[0];
    auto max_wgs                                          = execute_unit_org.workgroupsize_max;
    std::vector<std::vector<uint32_t>> local_size_list_3d = {
        {16, 4, 1}, {8, 8, 1},   {4, 16, 1}, {2, 32, 1}, {1, 64, 1}, {2, 64, 1}, {4, 64, 1},
        {8, 64, 1}, {16, 64, 1}, {8, 64, 2}, {4, 64, 4}, {2, 64, 8}, {2, 64, 4}, {},
    };
    std::vector<std::vector<uint32_t>> local_size_list_2d = {
        {2, max_wgs / 2},   {4, max_wgs / 4},   {8, max_wgs / 8},
        {16, max_wgs / 16}, {max_wgs / 2, 2},   {max_wgs / 4, 4},
        {max_wgs / 8, 8},   {max_wgs / 16, 16}, {},
    };
    std::vector<uint32_t> local_size_default;
    if (execute_unit_org.global_work_size.size() == 2) {
        local_size_default = LocalWS2DDefault(execute_unit_org);
    } else if (execute_unit_org.global_work_size.size() == 3) {
        local_size_default = LocalWS3DDefault(execute_unit_org);
    }

    OpenCLExecuteUnit exec_unit_default = execute_unit_org;
    exec_unit_default.local_work_size   = local_size_default;
    execute_units_.push_back(exec_unit_default);

    if (execute_unit_org.global_work_size.size() == 2) {
        for (auto local_size : local_size_list_2d) {
            OpenCLExecuteUnit exec_unit_temp = execute_unit_org;
            exec_unit_temp.local_work_size   = local_size;
            execute_units_.push_back(exec_unit_temp);
        }
    } else if (execute_unit_org.global_work_size.size() == 3) {
        for (auto local_size : local_size_list_3d) {
            OpenCLExecuteUnit exec_unit_temp = execute_unit_org;
            exec_unit_temp.local_work_size   = local_size;
            execute_units_.push_back(exec_unit_temp);
        }
    }

#endif

    Status ret   = TNN_OK;
    int unit_idx = 0;
    for (auto execute_unit : execute_units_) {
#if TNN_PROFILE
        std::shared_ptr<OpenCLProfilingData> pdata(new OpenCLProfilingData());
        UpdateProfilingData(pdata.get(), execute_unit.global_work_size, execute_unit.local_work_size, unit_idx);
        ret = RunKernel(execute_unit.ocl_kernel, execute_unit.global_work_size, execute_unit.local_work_size,
                        ocl_context_->CommandQueue(), op_name_, pdata.get());
        CHECK_TNN_OK(ret)
        ocl_context_->AddProfilingData(pdata);
#else

        ret = RunKernel(execute_unit.ocl_kernel, execute_unit.global_work_size, execute_unit.local_work_size,
                        ocl_context_->CommandQueue(), op_name_);
        CHECK_TNN_OK(ret)

#endif
        unit_idx++;
    }

    if (NeedFlush()) {
        ocl_context_->CommandQueue()->flush();
    }

    return TNN_OK;
}

#if TNN_PROFILE
void OpenCLLayerAcc::UpdateProfilingData(OpenCLProfilingData *pdata, std::vector<uint32_t> gws,
                                         std::vector<uint32_t> lws, int idx) {
    AbstractLayerAcc::UpdateProfilingData(pdata, param_, input_dims_, output_dims_);
    if (idx != 0)
        pdata->layer_name += "_" + ToString(idx);
    pdata->op_name         = op_name_;
    pdata->global_worksize = gws;
    pdata->local_worksize  = lws;
}
#endif

void OpenCLLayerAcc::ConfigKernelStrategy() {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    gpu_info_                     = opencl_runtime->GetGpuInfo();
    if (gpu_info_.type == ADRENO && gpu_info_.opencl_version >= 2.0f) {
        if (gpu_info_.model_num > 509)
            run_3d_ndrange_ = true;
    }
}

std::vector<DataFormat> OpenCLLayerAcc::SupportDataFormat(DataType data_type, int dims_size) {
    std::vector<DataFormat> support_list;
    if (dims_size == 4) {
        support_list.push_back(DATA_FORMAT_NHC4W4);
    }
    return support_list;
}

bool OpenCLLayerAcc::NeedFlush() {
    // flush by magic number
    if (0 == ocl_context_->AddAndGetFlushCount() % 10) {
        return true;
    }
    return false;
}

Status OpenCLLayerAcc::ConvertChannelWeights(RawBuffer &raw_handle, shared_ptr<OpenCLMemory> &ocl_handle,
                                             int output_channel, bool has_handle, bool share_channel, bool use_buffer) {
    // convert first check handle is null and handle data type is float or half,
    // then process with float pointer.
    Status ret = TNN_OK;
    if (!has_handle) {
        ret = ConvertChannelWeights(nullptr, ocl_handle, output_channel, has_handle, share_channel, use_buffer);
        CHECK_TNN_OK(ret)
    } else if (raw_handle.GetDataType() == DATA_TYPE_FLOAT) {
        // get float pointer from raw buffer
        float *handle_data_ptr = raw_handle.force_to<float *>();
        if (handle_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertChannelWeights(handle_data_ptr, ocl_handle, output_channel, has_handle, share_channel, use_buffer);
        CHECK_TNN_OK(ret)
    } else {
        // if handle is half, need convert to float first.
        auto float_data_ptr = GetFloatFromRawBuffer(raw_handle);
        if (float_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertChannelWeights(float_data_ptr.get(), ocl_handle, output_channel, has_handle, share_channel,
                                    use_buffer);
        CHECK_TNN_OK(ret)
    }

    return ret;
}

// ConvertChannelWeights only convert weights dims equal to 1 or output_channel.
// Convert Weights to clBuffer or ClImage, pack c4.
Status OpenCLLayerAcc::ConvertChannelWeights(float *handle_data_ptr, shared_ptr<OpenCLMemory> &ocl_handle,
                                             int output_channel, bool has_handle, bool share_channel, bool use_buffer) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();

    // copy weights data into clBuffer
    int handle_size = UP_DIV(output_channel, 4) * 4;
    cl_int ret      = CL_SUCCESS;
    cl::Buffer handle_clbuffer(*opencl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               handle_size * sizeof(float), nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory falied");
    }
    float *handle_clbuffer_ptr = (float *)ocl_context_->CommandQueue()->enqueueMapBuffer(
        handle_clbuffer, true, CL_MAP_WRITE, 0, handle_size * sizeof(float), nullptr, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMMAP_ERROR, "OpenCL MemMap failed");
    }
    memset(handle_clbuffer_ptr, 0, handle_size * sizeof(float));
    if (has_handle) {
        for (int i = 0; i < output_channel; ++i) {
            handle_clbuffer_ptr[i] = share_channel ? handle_data_ptr[0] : handle_data_ptr[i];
        }
    }
    ret = ocl_context_->CommandQueue()->enqueueUnmapMemObject(handle_clbuffer, handle_clbuffer_ptr);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL MemUnMap falied");
    }

    // create ocl_handle_
    if (use_buffer_) {
        // use clBuffer
        ocl_handle.reset(new OpenCLMemory(TNN_CL_BUFFER));
        size_t type_size = sizeof(float);
        if (opencl_runtime->GetPrecision() != PRECISION_HIGH)
            type_size = 2;
        cl::Buffer *buffer =
            new cl::Buffer(*opencl_runtime->Context(), CL_MEM_READ_WRITE, handle_size * type_size, nullptr, &ret);
        if (ret != CL_SUCCESS) {
            CHECK_CL_SUCCESS(ret)
            if (nullptr != buffer)
                delete buffer;
            return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory falied");
        }
        ocl_handle->SetData(buffer, true);

        // convert buffer to buffer
        shared_ptr<OpenCLMemory> input(new OpenCLMemory(TNN_CL_BUFFER));
        input->SetData(&handle_clbuffer);
        ImageBufferConvertor convertor(opencl_runtime, ocl_context_->CommandQueue());
        return convertor.ConvertBufferToBuffer(input.get(), ARGUMENT, {output_channel}, ocl_handle.get(), true);

    } else {
        // use clImage
        int ocl_handle_w          = UP_DIV(output_channel, 4);
        int ocl_handle_h          = 1;
        cl_channel_type data_type = CL_FLOAT;
        if (opencl_runtime->GetPrecision() != PRECISION_HIGH)
            data_type = CL_HALF_FLOAT;
        cl::Image2D *image =
            new cl::Image2D(*opencl_runtime->Context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, data_type),
                            ocl_handle_w, ocl_handle_h, 0, nullptr, &ret);
        if (ret != CL_SUCCESS) {
            CHECK_CL_SUCCESS(ret)
            if (nullptr != image)
                delete image;
            return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory falied");
        }
        ocl_handle.reset(new OpenCLMemory(TNN_CL_IMAGE));
        ocl_handle->SetData(image, true);

        // convert buffer to image
        shared_ptr<OpenCLMemory> input_blob(new OpenCLMemory(TNN_CL_BUFFER));
        input_blob->SetData(&handle_clbuffer);
        ImageBufferConvertor convertor(opencl_runtime, ocl_context_->CommandQueue());
        return convertor.ConvertBufferToImage(input_blob.get(), ARGUMENT, {output_channel}, ocl_handle.get(), true);
    }
}

}  // namespace TNN_NS
