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

#include "tnn/device/opencl/opencl_device.h"

#include "tnn/device/opencl/acc/opencl_cpu_adapter_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/device/opencl/opencl_context.h"
#include "tnn/device/opencl/opencl_memory.h"
#include "tnn/device/opencl/opencl_runtime.h"
#include "tnn/utils/blob_memory_size_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

OpenCLDevice::OpenCLDevice(DeviceType device_type) : AbstractDevice(device_type) {}

OpenCLDevice::~OpenCLDevice() {}

BlobMemorySizeInfo OpenCLDevice::Calculate(BlobDesc& desc) {
    OpenCLRuntime* opencl_runtime = OpenCLRuntime::GetInstance();
    std::vector<size_t> image_2d_max_size = opencl_runtime->GetImage2dMaxSize();
    ASSERT(image_2d_max_size.size() == 2);
    BlobMemorySizeInfo info = Calculate2DCLImageMemorySize(desc);
    ASSERT(info.dims.size() == 2);
    if (info.dims[0] > image_2d_max_size[0] || info.dims[1] > image_2d_max_size[1]) {
        LOGD("Exceed clImage limit, dims: [%d, %d]\n", info.dims[0], info.dims[1]);
        desc.data_format = DATA_FORMAT_NCHW;
        info = Calculate1DMemorySize(desc);
    }
    return info;
}

Status OpenCLDevice::Allocate(void** handle, MatType mat_type, DimsVector dims) {
    if (dims.size() != 4) {
        LOGE("invalid dim size: %d\n", (int)dims.size());
        return Status(TNNERR_PARAM_ERR, "invalid dim size");
    }

    BlobDesc desc;
    desc.dims        = dims;
    desc.device_type = GetDeviceType();
    desc.data_type   = DATA_TYPE_HALF; // try to use half precision
    if (mat_type == N8UC4) {
        auto size_info = Calculate(desc);
        return Allocate(handle, size_info);
    } else if (mat_type == NGRAY) {
        desc.data_type = DATA_TYPE_INT8;
        desc.data_format = DATA_FORMAT_NCHW;
        auto size_info = Calculate1DMemorySize(desc);
        return Allocate(handle, size_info);
    } else {
        LOGE("opencl allocator not support this mat type: %d\n", mat_type);
        return Status(TNNERR_PARAM_ERR, "opencl not support this mat type");
    }
}

//allocate clImage/clBuffer
Status OpenCLDevice::Allocate(void** handle, BlobMemorySizeInfo& desc) {
    OpenCLRuntime* opencl_runtime = OpenCLRuntime::GetInstance();

    if (DATA_TYPE_HALF != desc.data_type && DATA_TYPE_FLOAT != desc.data_type && DATA_TYPE_INT32 != desc.data_type &&
        DATA_TYPE_INT8 != desc.data_type) {
        LOGE("opencl allocator not support this data type: %d\n", desc.data_type);
        return Status(TNNERR_PARAM_ERR, "opencl not support this data type");
    }

    if (desc.dims.size() == 2) {
        // allocate clImage
        cl_mem_flags mem_flag     = CL_MEM_READ_WRITE;
        cl_channel_type data_type = CL_FLOAT;

        if (DATA_TYPE_HALF == desc.data_type && opencl_runtime->GetPrecision() != PRECISION_HIGH) {
            data_type = CL_HALF_FLOAT;
        }
        if (DATA_TYPE_INT32 == desc.data_type) {
            data_type = CL_SIGNED_INT32;
        }
        int w = desc.dims[0];
        int h = desc.dims[1];
        cl_int error;
        *handle = new cl::Image2D(*opencl_runtime->Context(), mem_flag, cl::ImageFormat(CL_RGBA, data_type), w, h, 0,
                                nullptr, &error);
        if (error != CL_SUCCESS) {
            CHECK_CL_SUCCESS(error);
            char error_str[128];
            sprintf(error_str, "OpenCL Allocate Image Failed (w=%d, h=%d)", w, h);
            return Status(TNNERR_OPENCL_API_ERROR, error_str);
        }
    } else if (desc.dims.size() == 1) {
        // allocate clBuffer
        cl_mem_flags mem_flag     = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
        int type_size = sizeof(float);

        if (DATA_TYPE_HALF == desc.data_type && opencl_runtime->GetPrecision() != PRECISION_HIGH) {
            type_size = 2;
        }
        if (DATA_TYPE_INT32 == desc.data_type) {
            type_size = sizeof(int);
        }
        if (DATA_TYPE_INT8 == desc.data_type) {
            type_size = 1;
        }
        cl_int error;
        *handle = new cl::Buffer(*opencl_runtime->Context(), mem_flag, (cl::size_type)(type_size * desc.dims[0]),
                                 nullptr, &error);
        if (error != CL_SUCCESS) {
            CHECK_CL_SUCCESS(error);
            char error_str[128];
            sprintf(error_str, "OpenCL Allocate Buffer Failed (count=%d)", desc.dims[0]);
            return Status(TNNERR_OPENCL_API_ERROR, error_str);
        }
    } else {
        char error_str[128];
        sprintf(error_str, "OpenCL not support Allocate (dims=%d)", (int)desc.dims.size());
        return Status(TNNERR_PARAM_ERR, error_str);
    }
    return TNN_OK;
}

//release clImage
Status OpenCLDevice::Free(void* handle) {
    cl::Image2D* buffer = static_cast<cl::Image2D*>(handle);
    if (buffer != NULL)
        delete buffer;
    return TNN_OK;
}

//Copy data from Cpu To Device, format is same.
Status OpenCLDevice::CopyToDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc, void* command_queue) {
    OpenCLRuntime* opencl_runtime          = OpenCLRuntime::GetInstance();
    cl::CommandQueue* opencl_command_queue = static_cast<cl::CommandQueue*>(command_queue);
    if (opencl_command_queue == nullptr)
        return Status(TNNERR_DEVICE_INVALID_COMMAND_QUEUE, "command_queue is nullptr");

    // Todo: convert src data type
    cl_int cl_ret;
    std::shared_ptr<OpenCLMemory> clbuffer(new OpenCLMemory(TNN_CL_BUFFER));
    cl::Buffer buffer(*opencl_runtime->Context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                      DimsVectorUtils::Count(desc.dims) * sizeof(float));
    clbuffer->SetData(&buffer);
    auto clbuffer_ptr = opencl_command_queue->enqueueMapBuffer(
        buffer, true, CL_MAP_WRITE, 0, DimsVectorUtils::Count(desc.dims) * sizeof(float), nullptr, nullptr, &cl_ret);
    if (cl_ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(cl_ret)
        return Status(TNNERR_OPENCL_MEMMAP_ERROR, "OpenCL MemMap failed");
    }
    memcpy(clbuffer_ptr, reinterpret_cast<char*>(src->base) + src->bytes_offset,
           DimsVectorUtils::Count(desc.dims) * sizeof(float));
    cl_ret = opencl_command_queue->enqueueUnmapMemObject(buffer, clbuffer_ptr);
    if (cl_ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(cl_ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL MemUnMap failed");
    }

    std::shared_ptr<OpenCLMemory> climage(new OpenCLMemory(TNN_CL_IMAGE));
    climage->SetData(reinterpret_cast<char*>(dst->base) + dst->bytes_offset);

    ImageBufferConvertor convertor(opencl_runtime, opencl_command_queue);
    return convertor.ConvertBufferToImage(clbuffer.get(), NCHW_BUFFER, desc.dims, climage.get(), true);
}

//Copy data from Device To Cpu, format is same.
Status OpenCLDevice::CopyFromDevice(BlobHandle* dst, const BlobHandle* src, BlobDesc& desc, void* command_queue) {
    OpenCLRuntime* opencl_runtime          = OpenCLRuntime::GetInstance();
    cl::CommandQueue* opencl_command_queue = static_cast<cl::CommandQueue*>(command_queue);
    if (opencl_command_queue == nullptr)
        return Status(TNNERR_DEVICE_INVALID_COMMAND_QUEUE, "command_queue is nullptr");

    std::shared_ptr<OpenCLMemory> clbuffer(new OpenCLMemory(TNN_CL_BUFFER));
    cl::Buffer buffer(*opencl_runtime->Context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                      DimsVectorUtils::Count(desc.dims) * sizeof(float));
    clbuffer->SetData(&buffer);

    std::shared_ptr<OpenCLMemory> climage(new OpenCLMemory(TNN_CL_IMAGE));
    climage->SetData(reinterpret_cast<char*>(src->base) + src->bytes_offset);

    ImageBufferConvertor convertor(opencl_runtime, opencl_command_queue);
    Status ret = convertor.ConvertImageToBuffer(climage.get(), NCHW_BUFFER, desc.dims, clbuffer.get(), true);
    if (ret != TNN_OK)
        return ret;

    cl_int cl_ret;
    auto clbuffer_ptr = opencl_command_queue->enqueueMapBuffer(
        buffer, true, CL_MAP_READ, 0, DimsVectorUtils::Count(desc.dims) * sizeof(float), nullptr, nullptr, &cl_ret);
    if (cl_ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(cl_ret)
        return Status(TNNERR_OPENCL_MEMMAP_ERROR, "OpenCL MemMap failed");
    }
    memcpy(reinterpret_cast<char*>(dst->base) + dst->bytes_offset, clbuffer_ptr,
           DimsVectorUtils::Count(desc.dims) * sizeof(float));
    cl_ret = opencl_command_queue->enqueueUnmapMemObject(buffer, clbuffer_ptr);
    if (cl_ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(cl_ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL MemUnMap failed");
    }

    return TNN_OK;
}

//create layer acc with layer type
AbstractLayerAcc* OpenCLDevice::CreateLayerAcc(LayerType type) {
    auto& layer_creator_map = GetLayerCreatorMap();
    if (layer_creator_map.count(type) > 0) {
        return layer_creator_map[type]->CreateLayerAcc(type);
    } else {
        return new OpenCLCpuAdapterAcc(type);
    }
}

std::shared_ptr<const ImplementedLayout> OpenCLDevice::GetImplementedLayout(LayerType type) {
    auto &layer_layout_map = GetLayerLayoutMap();
    if (layer_layout_map.count(type) > 0) {
        return layer_layout_map[type];
    }
    return std::make_shared<ImplementedLayout>();
}

Context* OpenCLDevice::CreateContext(int device_id) {
    return new OpenCLContext();
}

NetworkType OpenCLDevice::ConvertAutoNetworkType() {
    return NETWORK_TYPE_DEFAULT;
}

std::map<LayerType, std::shared_ptr<LayerAccCreator>>& OpenCLDevice::GetLayerCreatorMap() {
    static std::map<LayerType, std::shared_ptr<LayerAccCreator>> layer_creator_map;
    return layer_creator_map;
}

Status OpenCLDevice::RegisterLayerAccCreator(LayerType type, LayerAccCreator* creator) {
    GetLayerCreatorMap()[type] = std::shared_ptr<LayerAccCreator>(creator);
    return TNN_OK;
}

Status OpenCLDevice::RegisterLayerLayout(LayerType type, std::shared_ptr<ImplementedLayout> layout) {
    GetLayerLayoutMap()[type] = layout;
    return TNN_OK;
}

std::map<LayerType, std::shared_ptr<ImplementedLayout>> &OpenCLDevice::GetLayerLayoutMap() {
    static std::map<LayerType, std::shared_ptr<ImplementedLayout>> layer_layout_map;
    return layer_layout_map;
}

TypeDeviceRegister<OpenCLDevice> g_opencl_device_register(DEVICE_OPENCL);

}  // namespace TNN_NS
