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
    precision_ = dx_context_->GetPrecision();

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
        LOGD("DirectX Blob Pricision is Half!\n");
        set_data_type(inputs, DATA_TYPE_HALF);
        set_data_type(outputs, DATA_TYPE_HALF);
    } else {
        LOGD("DirectX Blob Pricision is Float!\n");
        set_data_type(inputs, DATA_TYPE_FLOAT);
        set_data_type(outputs, DATA_TYPE_FLOAT);
    }

    input_dims_  = inputs[0]->GetBlobDesc().dims;
    output_dims_ = outputs[0]->GetBlobDesc().dims;

    status = ReloadConstantBlobs(inputs, false);
    RETURN_ON_NEQ(status, TNN_OK);

#if TNN_PROFILE
    profiling_data = std::shared_ptr<DirectXProfilingData>(new DirectXProfilingData());
    RETURN_ON_NEQ(profiling_data->Init(), TNN_OK);
#endif

    return TNN_OK;
}

DirectXLayerAcc::~DirectXLayerAcc() {}

Status DirectXLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
//    return CheckBlob(inputs, outputs);
    return TNN_OK;
}

Status DirectXLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

#if TNN_PROFILE
    UpdateProfilingData(profiling_data.get(), param_, input_dims_, output_dims_);
    // context only accepts profilingdata after the StartProfile() call
    // DirectXProfilingResult will remove duplicated datas, on worry 
    dx_context_->AddProfilingData(profiling_data);
    profiling_data->Begin();
#endif

    Status ret = this->DoForward(inputs, outputs);

#if TNN_PROFILE
    profiling_data->End();
#endif

    return ret;
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

Status DirectXLayerAcc::ConvertChannelWeights(RawBuffer &raw_handle, shared_ptr<DirectXMemory> &handle,
                                             int output_channel, bool has_handle, bool share_channel, bool use_buffer) {
    // convert first check handle is null and handle data type is float or half,
    // then process with float pointer.
    Status ret = TNN_OK;
    if (!has_handle) {
        ret = ConvertChannelWeights(nullptr, handle, output_channel, has_handle, share_channel, use_buffer);
        RETURN_ON_NEQ(ret, TNN_OK);
    } else if (raw_handle.GetDataType() == DATA_TYPE_FLOAT) {
        // get float pointer from raw buffer
        float *handle_data_ptr = raw_handle.force_to<float *>();
        if (handle_data_ptr == nullptr) {
            return Status(TNNERR_DX_ACC_INIT_ERR, "pointer is null");
        }
        ret = ConvertChannelWeights(handle_data_ptr, handle, output_channel, has_handle, share_channel, use_buffer);
        RETURN_ON_NEQ(ret, TNN_OK);
    } else {
        // if handle is half, need convert to float first.
        auto float_data_ptr = GetFloatFromRawBuffer(raw_handle);
        if (float_data_ptr == nullptr) {
            return Status(TNNERR_DX_ACC_INIT_ERR, "pointer is null");
        }
        ret = ConvertChannelWeights(float_data_ptr.get(), handle, output_channel, has_handle, share_channel,
                                    use_buffer);
        RETURN_ON_NEQ(ret, TNN_OK);
    }

    return ret;
}

// ConvertChannelWeights only convert weights dims equal to 1 or output_channel.
// Convert Weights to DirectXBuffer of Texture2D (pack c4)
Status DirectXLayerAcc::ConvertChannelWeights(float *handle_data_ptr, shared_ptr<DirectXMemory> &handle,
                                             int output_channel, bool has_handle, bool share_channel, bool use_buffer) {
    std::shared_ptr<float> host_ptr = std::shared_ptr<float>(new float[output_channel], [](float * p){delete[] p;});
    memset(host_ptr.get(), 0, output_channel * sizeof(float));
    if (has_handle) {
        for(int i=0;i<output_channel;i++) {
            host_ptr.get()[i] = share_channel ? handle_data_ptr[0] : handle_data_ptr[i];
        }
    }

    auto dx_mem = DirectXMemory::CreateBufferMemoryFromHost(host_ptr.get(), {output_channel}, DATA_TYPE_FLOAT, DATA_FORMAT_NCHW);
    if (!dx_mem) {
        LOGE("CreateBufferMemoryFromHost failed\n");
        return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "craete directx memory failed.");
    }

    if (precision_ == PRECISION_LOW) {
        LOGE("FP16 Weigths not supported now.");
        return Status(TNNERR_DX_ACC_INIT_ERR, "FP16 weights not supported now.");
    }

    if (use_buffer) {
        handle = std::move(dx_mem);
    } else {
        LOGE("Convert ChannelWeights to Texture not implemented\n");
        return Status(TNNERR_DX_ACC_INIT_ERR, "Convert ChannelWeights to Texture not implemented");
    } 

    return TNN_OK;
}

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
