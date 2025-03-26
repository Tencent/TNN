// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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


#include "tnn/core/blob.h"
#include "tnn/device/snpe/snpe_dsp_context.h"
#include "tnn/device/snpe/snpe_dsp_device.h"
#include "tnn/utils/blob_memory_size_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

BlobMemorySizeInfo SnpeDspDevice::Calculate1DMemorySize(BlobDesc &desc) {
    BlobMemorySizeInfo info;
    info.data_type = desc.data_type;
    int count      = 0;
    if (desc.data_format == DATA_FORMAT_NC4HW4) {
        count = desc.dims[0] * ROUND_UP(DimsFunctionUtils::GetDim(desc.dims, 1), 4) * DimsVectorUtils::Count(desc.dims, 2);
    } else {
        count = DimsVectorUtils::Count(desc.dims);
    }
    info.dims.push_back(count);
    return info;
}

SnpeDspDevice::SnpeDspDevice(DeviceType device_type) : AbstractDevice(device_type) {}

SnpeDspDevice::~SnpeDspDevice() {}

BlobMemorySizeInfo SnpeDspDevice::Calculate(BlobDesc &desc) {
    return SnpeDspDevice::Calculate1DMemorySize(desc);
}

Status SnpeDspDevice::Allocate(void **handle, MatType mat_type, DimsVector dims) {
    // Use CPU Allocation now, update to SNPE dsp later.
    BlobDesc desc;
    desc.dims        = dims;
    desc.device_type = DEVICE_NAIVE;
    if (mat_type == NCHW_FLOAT) {
        desc.data_type   = DATA_TYPE_FLOAT;
        desc.data_format = DATA_FORMAT_NCHW;
        auto size_info   = Calculate(desc);
        return Allocate(handle, size_info);
    } else {
        LOGE("SnpeDspDevice does not support mat_type:%d\n", mat_type);
        return Status(TNNERR_PARAM_ERR, "SNPE DSP does not support mat_type");
    }
}

Status SnpeDspDevice::Allocate(void **handle, BlobMemorySizeInfo &size_info) {
    // Use CPU Allocation now, update to SNPE dsp later.
    if (handle) {
        auto size = GetBlobMemoryBytesSize(size_info);
        if (size > 0) {
            *handle = malloc(size);
            if (*handle && size > 0) {
                memset(*handle, 0, size);
            }
        } else if (size == 0) {
            *handle = nullptr;
        } else {
            return Status(TNNERR_PARAM_ERR, "CpuDevice::Allocate malloc bytes size < 0");
        }
    }
    return TNN_OK;
}

Status SnpeDspDevice::Free(void *handle) {
    if (handle) {
        free(handle);
    }
    return TNN_OK;
}

Status SnpeDspDevice::CopyToDevice(BlobHandle *dst, const BlobHandle *src, BlobDesc &desc, void *command_queue) {
    // TODO: Use CPU CopyToDevice now, update to SNPE DSP Version later.
    auto size_info       = Calculate(desc);
    size_t size_in_bytes = GetBlobMemoryBytesSize(size_info);
    memcpy(reinterpret_cast<char*>(dst->base) + dst->bytes_offset,
           reinterpret_cast<char*>(src->base) + src->bytes_offset, size_in_bytes);
    return TNN_OK;
}

Status SnpeDspDevice::CopyFromDevice(BlobHandle *dst, const BlobHandle *src, BlobDesc &desc, void *command_queue) {
    // TODO: Use CPU CopyFromDevice now, update to SNPE DSP Version later.
    auto size_info       = Calculate(desc);
    size_t size_in_bytes = GetBlobMemoryBytesSize(size_info);
    memcpy(reinterpret_cast<char*>(dst->base) + dst->bytes_offset,
           reinterpret_cast<char*>(src->base) + src->bytes_offset, size_in_bytes);
    return TNN_OK;
}


AbstractLayerAcc *SnpeDspDevice::CreateLayerAcc(LayerType type) {
    return nullptr;
}


NetworkType SnpeDspDevice::ConvertAutoNetworkType() {
    return NETWORK_TYPE_SNPE;
}

Context* SnpeDspDevice::CreateContext(int device_id) {
    return new SnpeDspContext();
}


TypeDeviceRegister<SnpeDspDevice> g_snpe_dsp_device_register(DEVICE_DSP);

} // namespace TNN_NS
