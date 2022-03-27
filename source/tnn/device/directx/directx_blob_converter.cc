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

struct ParamCB
{
    float scale0;
    float scale1;
    float scale2;
    float scale3;

    float bias0;
    float bias1;
    float bias2;
    float bias3;

    int n;
    int c;
    int h;
    int w;
};


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

    auto context = tnn_device->GetID3DContext();
    if (!context) {
        LOGE("Got null ID3DDeviceContext");
        return  Status(TNNERR_CONTEXT_ERR, "got null directx device context");
    }

    auto blob_memory = DirectXMemory::CreateRefMemoryFromBlob(blob);
    shared_ptr<DirectXMemory> mat_buffer(new DirectXMemory(TNN_DX_BUFFER));

    auto dims = image.GetDims();
    auto mat_size = DimsVectorUtils::Count(dims);
    size_t type_size = sizeof(uint8_t);
    // allocate Buffer
    ID3D11Buffer * buffer;

    D3D11_BUFFER_DESC buffer_desc = {};
    buffer_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    buffer_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    buffer_desc.Usage = D3D11_USAGE_DEFAULT;
    buffer_desc.ByteWidth = type_size * mat_size;

    D3D11_SUBRESOURCE_DATA srd = {};
    srd.pSysMem = image.GetData();
    srd.SysMemPitch = 0;
    srd.SysMemSlicePitch = 0;

    LOGI("DirectX create buffer of len %u \n", type_size * mat_size);
    HRESULT hr = pDevice->CreateBuffer( &buffer_desc, &srd, &buffer);
    if (FAILED(hr)) {
        mat_buffer->SetData(nullptr, false);
        LOGE("DirectX createbuffer failed. erro code %d", (long) hr);
        return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "DirectX buffer allocation failed.");
    }
    mat_buffer->SetData(buffer, false);

    auto mat_srv = mat_buffer->GetSRV();
    auto blob_uav = blob_memory->GetUAV();

    // Constant Buffer
    std::shared_ptr<ID3D11Buffer> inputCBBuffer;
    ID3D11Buffer* pInputCBBuffer = inputCBBuffer.get();
    ParamCB paramCB_data = {param.scale[0], param.scale[1], param.scale[2], param.scale[3],
                            param.bias[0], param.bias[1],param.bias[2],param.bias[3],
                            dims[0], dims[1], dims[2], dims[3]};

    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth = sizeof(ParamCB);
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.CPUAccessFlags = 0u;
    bd.StructureByteStride = 0u;
    bd.MiscFlags = 0u;

    D3D11_SUBRESOURCE_DATA srd1 = {};
    srd1.pSysMem = &paramCB_data;
    srd1.SysMemPitch = 0;
    srd1.SysMemSlicePitch = 0;

    hr = pDevice->CreateBuffer(&bd, &srd1, &pInputCBBuffer);
    if (FAILED(hr)) {
        LOGE("DirectX create constant buffer failed. erro code %d", (long) hr);
        return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "DirectX constant buffer allocation failed.");
    }

    std::shared_ptr<ID3D11ComputeShader> ComputerShader;
    ID3D11ComputeShader* pComputerShader = ComputerShader.get();

    hr = pDevice->CreateComputeShader(g_N8UC3ToNCHW, sizeof(g_N8UC3ToNCHW), NULL, &pComputerShader);
    if (FAILED(hr))
    {
        LOGE("DirectX CreateComputeShader failed. erro code");
        return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "DirectX CreateComputeShader failed.");
    }
    unsigned int hw = dims[2]*dims[3];
    Status  ret = DispatchShader(pComputerShader,{mat_srv},{blob_uav},pInputCBBuffer,{hw/4,1,1});

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

REGISTER_DIRECTX_BLOB_CONVERT_FUNC(NCHW_FLOAT, DATA_TYPE_FLOAT,  CVT_DIR_BLOB2MAT, BlobToNCHW)

}
}  // namespace TNN_NS
