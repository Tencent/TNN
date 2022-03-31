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

#include "tnn/device/directx/directx_blob_converter.h"

#include "tnn/core/macro.h"
#include "tnn/core/blob_int8.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/string_utils_inner.h"
#include "tnn/device/directx/directx_device.h"
#include "tnn/device/directx/directx_util.h"

namespace TNN_NS {
namespace directx {

std::string DirectXBlobConverterAcc::GetUniqueBlobConvertKey(MatType mat_type, DataType data_type,
                                                         BlobConvertDirection cvt_dir) {
    return ToString(mat_type) + "_" + ToString(data_type) + "_" + ToString(cvt_dir);
}

std::map<std::string, DirectXBlobConvertFunc>& DirectXBlobConverterAcc::GetBlobConvertFuncMap() {
    static std::map<std::string, DirectXBlobConvertFunc> cvt_map;
    return cvt_map;
}

Status DirectXBlobConverterAcc::RegisterBlobConvertFunc(MatType mat_type, DataType data_type,
                                                    BlobConvertDirection cvt_dir, DirectXBlobConvertFunc cvt_func) {
    auto& cvt_map       = GetBlobConvertFuncMap();
    const auto& cvt_key = GetUniqueBlobConvertKey(mat_type, data_type, cvt_dir);
    cvt_map[cvt_key] = cvt_func;
    return TNN_OK;
}

Status DirectXBlobConverterAcc::GetBlobConvertFunc(MatType mat_type, DataType data_type,
                                               BlobConvertDirection cvt_dir, DirectXBlobConvertFunc& cvt_func) {
    const auto& cvt_map = GetBlobConvertFuncMap();
    const auto& cvt_key = GetUniqueBlobConvertKey(mat_type, data_type, cvt_dir);
    if (cvt_map.find(cvt_key) == cvt_map.end() || cvt_map.at(cvt_key) == nullptr) {
        LOGE("DirectXBlobConverterAcc::GetBlobConvertFunc, convert type not support yet. mat_type:%d data_type:%d cvt_dir:%d\n", mat_type, data_type, cvt_dir);
        return Status(TNNERR_PARAM_ERR, "DirectXBlobConverterAcc::GetBlobConvertFunc, convert type not support yet");
    }
    cvt_func = cvt_map.at(cvt_key);
    return TNN_OK;
}

Status DirectXBlobConverterAcc::ConvertToMatAsync(Mat &image, MatConvertParam param, void *command_queue) {
    Status ret = TNN_OK;
    if (blob_ == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input/output blob is null");
    }

    ret = GetBlobConvertFunc(image.GetMatType(), DATA_TYPE_FLOAT, CVT_DIR_BLOB2MAT, cvt_func_);
    RETURN_ON_NEQ(ret, TNN_OK);

    return cvt_func_(image, blob_, param, command_queue);
}

Status DirectXBlobConverterAcc::ConvertFromMatAsync(Mat &image, MatConvertParam param, void *command_queue) {
    Status ret = TNN_OK;
    if (blob_ == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input/output blob is null");
    }

    ret = GetBlobConvertFunc(image.GetMatType(), DATA_TYPE_FLOAT, CVT_DIR_MAT2BLOB, cvt_func_);
    RETURN_ON_NEQ(ret, TNN_OK);

    return cvt_func_(image, blob_, param, command_queue);
}

Status DirectXBlobConverterAcc::ConvertToMat(Mat &image, MatConvertParam param, void *command_queue) {
    auto ret = ConvertToMatAsync(image, param, command_queue);
    RETURN_ON_NEQ(ret, TNN_OK);
    // TODO: add synchronization
    return TNN_OK;
}

Status DirectXBlobConverterAcc::ConvertFromMat(Mat &image, MatConvertParam param, void *command_queue) {
    auto ret = ConvertFromMatAsync(image, param, command_queue);
    RETURN_ON_NEQ(ret, TNN_OK);
    // TODO: add synchronization
    return TNN_OK;
}

DECLARE_BLOB_CONVERTER_CREATER(DirectX);
REGISTER_BLOB_CONVERTER(DirectX, DEVICE_DIRECTX);

static Status NCHWToBlob(Mat& image,
                       Blob * blob,
                       const MatConvertParam& param,
                       void * command_queue) {
    
    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return Status(TNNERR_CONTEXT_ERR, "got null directx device");
    }

    if (image.GetDeviceType() == DEVICE_X86 || 
       image.GetDeviceType() == DEVICE_NAIVE || 
       image.GetDeviceType() == DEVICE_ARM) {
        BlobHandle cpu_handle;
        cpu_handle.base = image.GetData();
        tnn_device->CopyToDevice(&blob->GetHandle(), &cpu_handle, blob->GetBlobDesc(), command_queue);
    } else {
        return Status(TNNERR_PARAM_ERR, "DirectX Conterter support dx image now"); 
    }
    return TNN_OK;
}

static Status N8UC3ToBlob(Mat& image,
                         Blob * blob,
                         const MatConvertParam& param,
                         void * command_queue) {

    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return Status(TNNERR_CONTEXT_ERR, "got null directx device");
    }

    auto device = tnn_device->GetID3DDevice();
    if (!device) {
        LOGE("Got null ID3Ddevice");
        return  Status(TNNERR_CONTEXT_ERR, "got null directx device");
    }
    auto pDevice = device.get();

    auto blob_memory = DirectXMemory::CreateRefMemoryFromBlob(blob);
    shared_ptr<DirectXMemory> mat_buffer(new DirectXMemory(TNN_DX_BUFFER));

    DimsVector dims = image.GetDims();
    BlobMemorySizeInfo desc;
    desc.data_type = DATA_TYPE_INT8;
    desc.dims = {DimsVectorUtils::Count(dims)};
    Status status = AllocateBuffer(mat_buffer, desc,  image.GetData());
    RETURN_ON_NEQ(status, TNN_OK);

    auto mat_srv = mat_buffer->GetSRV();
    auto blob_uav = blob_memory->GetUAV();

    ParamCB param_cb_host = {param.scale[0], param.scale[1], param.scale[2], param.scale[3],
                            param.bias[0], param.bias[1],param.bias[2],param.bias[3],
                            dims[0], dims[1], dims[2], dims[3]};

    std::shared_ptr<ID3D11Buffer> param_cb;
    status = CreateConstBuffer<ParamCB>(param_cb_host, device, param_cb);

    RETURN_ON_NEQ(status, TNN_OK);

    std::shared_ptr<ID3D11ComputeShader> cs;
    status = GetShaderByName("N8UC3ToNCHW", cs);
    RETURN_ON_NEQ(status, TNN_OK);

    int hw = dims[2]*dims[3];
    Status  ret = DispatchShader(cs, {mat_srv}, {blob_uav}, {param_cb.get()}, {hw/4,1,1});

    return ret;
}

static Status N8UC4ToBlob(Mat& image,
                          Blob * blob,
                          const MatConvertParam& param,
                          void * command_queue) {

    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return Status(TNNERR_CONTEXT_ERR, "got null directx device");
    }

    auto device = tnn_device->GetID3DDevice();
    if (!device) {
        LOGE("Got null ID3Ddevice");
        return  Status(TNNERR_CONTEXT_ERR, "got null directx device");
    }
    auto pDevice = device.get();

    auto blob_memory = DirectXMemory::CreateRefMemoryFromBlob(blob);
    shared_ptr<DirectXMemory> mat_buffer(new DirectXMemory(TNN_DX_BUFFER));

    DimsVector dims = image.GetDims();
    BlobMemorySizeInfo desc;
    desc.data_type = DATA_TYPE_INT8;
    desc.dims = {DimsVectorUtils::Count(dims)};
    Status status = AllocateBuffer(mat_buffer, desc,  image.GetData());
    RETURN_ON_NEQ(status, TNN_OK);

    auto mat_srv = mat_buffer->GetSRV();
    auto blob_uav = blob_memory->GetUAV();

    ParamCB param_cb_host = {param.scale[0], param.scale[1], param.scale[2], param.scale[3],
                            param.bias[0], param.bias[1],param.bias[2],param.bias[3],
                            dims[0], dims[1], dims[2], dims[3]};

    std::shared_ptr<ID3D11Buffer> param_cb;
    status = CreateConstBuffer<ParamCB>(param_cb_host, device, param_cb);
    RETURN_ON_NEQ(status, TNN_OK);

    std::shared_ptr<ID3D11ComputeShader> cs;
    status = GetShaderByName("N8UC4ToNCHW", cs);
    RETURN_ON_NEQ(status, TNN_OK);

    int hw = dims[2]*dims[3];
    Status  ret = DispatchShader(cs,{mat_srv},{blob_uav}, {param_cb.get()},{hw,1,1});

    return ret;
}

static Status BlobToNCHW(Mat& image,
                       Blob * blob,
                       const MatConvertParam& param,
                       void * command_queue) {
    
    auto tnn_device = dynamic_cast<DirectXDevice*>(GetDevice(DEVICE_DIRECTX));
    if (!tnn_device) {
        LOGE("Got null directx device");
        return Status(TNNERR_CONTEXT_ERR, "got null directx device");
    }

    if (image.GetDeviceType() == DEVICE_X86 || 
       image.GetDeviceType() == DEVICE_NAIVE || 
       image.GetDeviceType() == DEVICE_ARM) {
        BlobHandle cpu_handle;
        cpu_handle.base = image.GetData();
        tnn_device->CopyFromDevice(&cpu_handle, &blob->GetHandle(), blob->GetBlobDesc(), command_queue);
    } else {
        return Status(TNNERR_PARAM_ERR, "DirectX Conterter support dx image now"); 
    }
    return TNN_OK;
}

REGISTER_DIRECTX_BLOB_CONVERT_FUNC(NCHW_FLOAT, DATA_TYPE_FLOAT,  CVT_DIR_MAT2BLOB, NCHWToBlob)
REGISTER_DIRECTX_BLOB_CONVERT_FUNC(N8UC3, DATA_TYPE_FLOAT,  CVT_DIR_MAT2BLOB, N8UC3ToBlob)
REGISTER_DIRECTX_BLOB_CONVERT_FUNC(N8UC4, DATA_TYPE_FLOAT,  CVT_DIR_MAT2BLOB, N8UC4ToBlob)

REGISTER_DIRECTX_BLOB_CONVERT_FUNC(NCHW_FLOAT, DATA_TYPE_FLOAT,  CVT_DIR_BLOB2MAT, BlobToNCHW)

}
}  // namespace TNN_NS
