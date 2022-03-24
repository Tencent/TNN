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

#include "tnn/device/directx/acc/directx_layer_acc.h"

#include <vector>
#include <set>

#include "tnn/device/directx/directx_device.h"
#include "tnn/device/directx/directx_memory.h"
#include "tnn/device/directx/directx_util.h"
#include "tnn/utils/string_utils_inner.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/blob_transfer_utils.h"

namespace TNN_NS {

namespace directx {

Status DirectXLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status status = AbstractLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(status, TNN_OK);

    param_       = param;
    resource_    = resource;
    layer_name_  = param->name;
    dx_context_ = dynamic_cast<DirectXContext *>(context);
    if (dx_context_ == nullptr) {
        return Status(TNNERR_NULL_PARAM, "DirectX Context Convert failed");
    }

    auto set_data_type = [](const std::vector<Blob *> blobs, DataType data_type) {
        std::set<DataType> tunable_types({DATA_TYPE_FLOAT, DATA_TYPE_HALF});
        for(auto blob : blobs) {
            if ( tunable_types.find(blob->GetBlobDesc().data_type) != tunable_types.end()) {
                blob->GetBlobDesc().data_type = data_type;
            }
        }
    };

    if (context->GetPrecision() == PRECISION_LOW) {
        LOGD("OpenCL Blob Pricision is Half!\n");
        set_data_type(inputs, DATA_TYPE_HALF);
        set_data_type(outputs, DATA_TYPE_HALF);
    } else {
        LOGD("OpenCL Blob Pricision is Float!\n");
        set_data_type(inputs, DATA_TYPE_FLOAT);
        set_data_type(outputs, DATA_TYPE_FLOAT);
    }

    input_dims_  = inputs[0]->GetBlobDesc().dims;
    output_dims_ = outputs[0]->GetBlobDesc().dims;

    status = ReloadConstantBlobs(inputs, false);
    RETURN_ON_NEQ(status, TNN_OK);

    return TNN_OK;
}

DirectXLayerAcc::~DirectXLayerAcc() {}

Status DirectXLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CheckBlob(inputs, outputs);
}

Status DirectXLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    auto d3d_context = GetID3DContext();

    auto in_memory = DirectXMemory::CreateRefMemoryFromBlob(inputs[0]); 
    auto out_memory = DirectXMemory::CreateRefMemoryFromBlob(outputs[0]); 

    auto in_srv = in_memory->GetSRV();
    auto out_uav = out_memory->GetUAV();

    auto in_buffer = (ID3D11Buffer *) in_memory->GetData();
    auto out_buffer = (ID3D11Buffer *) out_memory->GetData();
    std::shared_ptr<ID3D11ComputeShader> cs;
    Status ret = GetShaderByName("buffer_add", cs);
    RETURN_ON_NEQ(ret, TNN_OK);

    typedef struct launch_param {
        UINT n;
        UINT c;
        UINT h;
        UINT w;
    } launch_param_t;

    launch_param_t args;
    args.n = inputs[0]->GetBlobDesc().dims[0];
    args.c = inputs[0]->GetBlobDesc().dims[1];
    args.h = inputs[0]->GetBlobDesc().dims[2];
    args.w = inputs[0]->GetBlobDesc().dims[3];

    std::shared_ptr<ID3D11Buffer> const_buffer;
    ret = CreateConstBuffer<launch_param_t>(args, GetID3DDevice(), const_buffer);
    RETURN_ON_NEQ(ret, TNN_OK);

    const int THREADS_PER_BLOCK = 128;
    const int ELE_PER_THREAD    = 4;

    const int ele_count = DimsVectorUtils::Count(inputs[0]->GetBlobDesc().dims);

    ret = DispatchShader(cs, {in_srv}, {out_uav}, {const_buffer.get()}, {UP_DIV(ele_count, THREADS_PER_BLOCK * ELE_PER_THREAD)});
    RETURN_ON_NEQ(ret, TNN_OK);

    return TNN_OK;
}
std::vector<DataFormat> DirectXLayerAcc::SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) {
    std::vector<DataFormat> support_list;
    if (data_type == DATA_TYPE_INT32) {
        // support_list.push_back(DATA_FORMAT_NHC4W4);
        support_list.push_back(DATA_FORMAT_NCHW);
    } else {
        support_list.push_back(DATA_FORMAT_NCHW);
    }
    return support_list;
}

std::vector<DataType> DirectXLayerAcc::SupportDataType(int dims_size, BlobType blob_type) {
    std::vector<DataType> support_list;
    return {DATA_TYPE_FLOAT, DATA_TYPE_HALF};
}

/*
Status DirectXLayerAcc::ConvertChannelWeights(RawBuffer &raw_handle, shared_ptr<DirectXMemory> &ocl_handle,
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
Status DirectXLayerAcc::ConvertChannelWeights(float *handle_data_ptr, shared_ptr<DirectXMemory> &ocl_handle,
                                             int output_channel, bool has_handle, bool share_channel, bool use_buffer) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();

    // copy weights data into clBuffer
    int handle_size = UP_DIV(output_channel, 4) * 4;
    cl_int ret      = CL_SUCCESS;
    cl::Buffer handle_clbuffer(*opencl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               handle_size * sizeof(float), nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
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
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL MemUnMap failed");
    }

    // create ocl_handle_
    if (use_buffer_) {
        // use clBuffer
        ocl_handle.reset(new DirectXMemory(TNN_CL_BUFFER));
        size_t type_size = sizeof(float);
        if (opencl_runtime->GetPrecision() != PRECISION_HIGH)
            type_size = 2;
        cl::Buffer *buffer =
            new cl::Buffer(*opencl_runtime->Context(), CL_MEM_READ_WRITE, handle_size * type_size, nullptr, &ret);
        if (ret != CL_SUCCESS) {
            CHECK_CL_SUCCESS(ret)
            if (nullptr != buffer)
                delete buffer;
            return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
        }
        ocl_handle->SetData(buffer, true);

        // convert buffer to buffer
        shared_ptr<DirectXMemory> input(new DirectXMemory(TNN_CL_BUFFER));
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
            return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
        }
        ocl_handle.reset(new DirectXMemory(TNN_CL_IMAGE));
        ocl_handle->SetData(image, true);

        // convert buffer to image
        shared_ptr<DirectXMemory> input_blob(new DirectXMemory(TNN_CL_BUFFER));
        input_blob->SetData(&handle_clbuffer);
        ImageBufferConvertor convertor(opencl_runtime, ocl_context_->CommandQueue());
        return convertor.ConvertBufferToImage(input_blob.get(), ARGUMENT, {output_channel}, ocl_handle.get(), true);
    }
}
*/

Status DirectXLayerAcc::ReloadConstantBlobs(const std::vector<Blob *> &inputs, bool only_reload_shape_differ_blob) {
    auto const_resource = const_resource_;
    auto const_resource_flag = const_resource_flag_;
    auto const_blob_map = const_blob_map_;
    for (auto iter : inputs) {
        auto name = iter->GetBlobDesc().name;
        if (const_resource == nullptr || const_resource->find(name) == const_resource->end()) {
            continue;
        }
        if (only_reload_shape_differ_blob && const_resource_flag &&
            const_resource_flag->find(name) == const_resource_flag->end()) {
            continue;
        }

        auto buffer = (*const_resource)[name];
        // int32 blob is not supported on opencl, only used on cpu
        if (buffer->GetDataType() == DATA_TYPE_INT32) {
            continue;
        }
        std::shared_ptr<Blob> blob = nullptr;
        if (const_blob_map.find(name) != const_blob_map.end()) {
            blob = const_blob_map[name];
        }

        auto status = RawBuffer2DirectXBlob(buffer.get(), blob);
        RETURN_ON_NEQ(status, TNN_OK);

        blob->SetFlag(DATA_FLAG_CHANGE_NEVER);
        auto dims = iter->GetBlobDesc().dims;
        auto data_type_size = DataTypeUtils::GetBytesSize(iter->GetBlobDesc().data_type);
        const_blob_map[name] = blob;
        iter->SetHandle(blob->GetHandle());
        iter->GetBlobDesc() = blob->GetBlobDesc();
        LOGD("Reload constant blob: %s\n", name.c_str());
    }
    const_blob_map_ = const_blob_map;
    return TNN_OK;
}

Status DirectXLayerAcc::RawBuffer2DirectXBlob(RawBuffer *buffer, std::shared_ptr<Blob> &blob, DataFormat format) {
        
    if (nullptr == buffer){
        LOGE("Got null RawBuffer");  
        return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "Got null Rawbuffer");
    }
    
    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return Status(TNNERR_DX_ACC_INIT_ERR, "Got null device");
    }

    BlobDesc desc;
    desc.device_type = DEVICE_DIRECTX;
    desc.data_type = buffer->GetDataType();
    desc.dims = buffer->GetBufferDims();
    desc.data_format = format;
    if (DimsVectorUtils::Count(desc.dims) > 0) {
        blob = std::make_shared<Blob>(desc, true);
    } else {
        return Status(TNNERR_PARAM_ERR, "raw buffer for opencl blob is empty");
    }

    if (format != DATA_FORMAT_NCHW) {
        LOGE("only supported NCHW now");
        return Status(TNNERR_PARAM_ERR, "only supported NCHW now");
    }

    auto d3d_context = tnn_device->GetID3DContext();

    // Only work on DATA_FORMAT_NCHW and TNN_DX_BUFFER now
    auto mem_type = GetMemoryType(desc);
    if (TNN_DX_BUFFER == mem_type) {
        ID3D11Buffer * dx_buf = (ID3D11Buffer*) blob->GetHandle().base;
        d3d_context->UpdateSubresource(dx_buf, 0, nullptr, buffer->force_to<void *>(), 0, 0);
    } else {
        LOGE("Directx Texture2D not supported now");
        return Status(TNNERR_PARAM_ERR, "directx not supported now");
    }

    return TNN_OK;
}

Status DirectXLayerAcc::CheckBlob(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    /*
     * Check whether the format is supported by DirectXLayerAcc or not.
     * The supported format of each layer is given by LayerAcc.
     * DirectX Blob may change format after allocate.
     */
    for (auto blob : outputs) {
        Status ret = ResolveBlobDataFormat(blob, BLOB_OUTPUT);
        if (ret != TNN_OK) {
            LOGE("Resolve Layer(%s)-Output Blob(%s) Data Format(%d) failed\n",
                 layer_name_.c_str(), blob->GetBlobDesc().name.c_str(), blob->GetBlobDesc().data_format);
            return ret;
        }

        ret = ResolveBlobDataType(blob, BLOB_OUTPUT);
        if (ret != TNN_OK) {
            LOGE("Resolve Layer(%s)-Output Blob(%s) Data Type(%d) failed\n",
                 layer_name_.c_str(), blob->GetBlobDesc().name.c_str(), blob->GetBlobDesc().data_type);
            return ret;
        }
    }

    for (auto blob : inputs) {
        Status ret = ResolveBlobDataFormat(blob, BLOB_INPUT);
        if (ret != TNN_OK) {
            LOGE("Resolve Layer(%s)-Input Blob(%s) Data Format(%d) failed\n",
                 layer_name_.c_str(), blob->GetBlobDesc().name.c_str(), blob->GetBlobDesc().data_format);
            return ret;
        }

        ret = ResolveBlobDataType(blob, BLOB_INPUT);
        if (ret != TNN_OK) {
            LOGE("Resolve Layer(%s)-Input Blob(%s) Data Type(%d) failed\n",
                 layer_name_.c_str(), blob->GetBlobDesc().name.c_str(), blob->GetBlobDesc().data_type);
            return ret;
        }
    }

    return TNN_OK;
}

Status DirectXLayerAcc::ResolveBlobDataType(Blob *blob, BlobType blob_type) {
    auto desc = blob->GetBlobDesc();
    auto support_list = SupportDataType(static_cast<int>(desc.dims.size()), blob_type);
    if (support_list.size() <= 0) {
        return Status(TNNERR_DEVICE_ACC_DATA_FORMAT_NOT_SUPPORT,
                      "unsupported data type for device acc");
    }

    /*
     * DATA_TYPE_AUTO : first type supported by the LayerAcc
     * Others:  return error if LayerAcc not support.
     */
    if (desc.data_type == DATA_TYPE_AUTO) {
        desc.data_type = support_list[0];
        blob->SetBlobDesc(desc);
        return TNN_OK;
    } else {
        auto iter = std::find(support_list.begin(), support_list.end(), desc.data_type);
        if (iter != support_list.end()) {
            return TNN_OK;
        } else {
            return Status(TNNERR_DEVICE_ACC_DATA_FORMAT_NOT_SUPPORT, "unsupported data type for device acc");
        }
    }
}

std::shared_ptr<ID3D11DeviceContext> DirectXLayerAcc::GetID3DContext() {

    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return std::shared_ptr<ID3D11DeviceContext>(nullptr);
    }
    auto context = tnn_device->GetID3DContext();
    if (!context) {
        LOGE("Got null ID3DDeviceContext");
        return std::shared_ptr<ID3D11DeviceContext>(nullptr);
    }
    return context;
}

std::shared_ptr<ID3D11Device> DirectXLayerAcc::GetID3DDevice() {

    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return std::shared_ptr<ID3D11Device>(nullptr);
    }
    auto device = tnn_device->GetID3DDevice();
    if (!device) {
        LOGE("Got null ID3Ddevice");
        return std::shared_ptr<ID3D11Device>(nullptr);
    }
    return device ;
}

} // namespace directx

}  // namespace TNN_NS
