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

#ifndef TNN_SOURCE_TNN_NETWORK_TORCH_UTILS_H_
#define TNN_SOURCE_TNN_NETWORK_TORCH_UTILS_H_

#include <memory>
#include <vector>

#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/core/macro.h"
#include "tnn/core/abstract_device.h"
#include "tnn/device/cuda/cuda_device.h"
#include "tnn/network/torch/torch_types.h"
#include "tnn/network/torch/torch_tensor.h"
#include "tnn/extern_wrapper/foreign_blob.h"

#include <torch/script.h>
#include <ATen/Functions.h>

namespace TNN_NS {

inline Status ConvertToTorchDevice(c10::Device& device, const DeviceType device_type, int device_id = -1) {
    Status ret = TNN_OK;
    switch (device_type) {
        case DEVICE_X86:
            device = c10::Device(c10::kCPU, c10::DeviceIndex(device_id));
            break;
        case DEVICE_CUDA:
            if (device_id == -1) {
                RETURN_ON_FAIL(dynamic_cast<CudaDevice*>(GetDevice(DEVICE_CUDA))->GetCurrentDeviceId(device_id));
            }
            device = c10::Device(c10::kCUDA, c10::DeviceIndex(device_id));
            break;
        default:
            ret = Status(TNNERR_DEVICE_NOT_SUPPORT, "device not supported by TorchNetwork");
            break;
    }

    return ret;
}

inline Status ConvertToDeviceType(DeviceType &device_type, const c10::Device& device) {
    Status ret = TNN_OK;
    switch (device.type()) {
        case c10::kCPU:
            device_type = DEVICE_X86;
            break;
        case c10::kCUDA:
            device_type = DEVICE_CUDA;
            break;
        default:
            ret = Status(TNNERR_DEVICE_NOT_SUPPORT, "device_type converting not implemented");
            break;
    }

    return ret;
}

inline Status ConvertToTorchDataType(at::ScalarType& scalar_type, DataType data_type) {
    Status ret = TNN_OK;
    switch (data_type) {
        case DATA_TYPE_FLOAT:
            scalar_type = at::ScalarType::Float;
            break;
        case DATA_TYPE_INT8:
            scalar_type = at::ScalarType::QInt8;
            break;
        case DATA_TYPE_HALF:
            scalar_type = at::ScalarType::Half;
            break;
        case DATA_TYPE_INT64:
            scalar_type = at::ScalarType::Long;
            break;
        case DATA_TYPE_INT32:
            scalar_type = at::ScalarType::Int;
            break;
        default:
            ret = Status(TNNERR_PARAM_ERR, "data_type not supported by TorchNetwork");
            break;
    }

    return ret;
}

inline Status ConvertToDataType(DataType &data_type, at::ScalarType& scalar_type) {
    Status ret = TNN_OK;
    switch (scalar_type) {
        case at::ScalarType::Float:
            data_type = DATA_TYPE_FLOAT;
            break;
        case at::ScalarType::QInt8:
            data_type = DATA_TYPE_INT8; 
            break;
        case at::ScalarType::Half:
            data_type = DATA_TYPE_HALF;
            break;
        case at::ScalarType::Long:
            data_type = DATA_TYPE_INT64;
            break;
        case at::ScalarType::Int:
            data_type = DATA_TYPE_INT32;
            break;
        default:
            ret = Status(TNNERR_PARAM_ERR, "data_type converting not implemented");
            break;
    }

    return ret;
}

inline std::vector<int64_t> ConvertDimsToIntArrayRef(DimsVector dims) {
    return std::vector<int64_t>(dims.begin(), dims.end());
}

inline bool mapKeysEqualTo(BlobMap &map, std::vector<std::string> &names) {
    if (map.size() != names.size()) {
        return false;
    }
    for(auto name : names) {
        if (map.find(name) == map.end()) {
            return false;
        }
    }
    return true;
}

Status CreateTensorByBlobDesc(std::shared_ptr<torch::Tensor> &tensor, BlobDesc desc);

Status CreateTensorByBlob(std::shared_ptr<torch::Tensor> &tensor, Blob *blob);

Status ConvertIValueToTensors(std::vector<torch::Tensor> &tensor, const torch::jit::IValue &ivalue);

Status GetBlobDescFromTensor(BlobDesc &desc, const torch::Tensor &tensor);

Status CreateIValueFromTypePtr(c10::IValue &ivalue, c10::TypePtr type);

Status IValueTensorTo(c10::IValue &ivalue, at::ScalarType scalar_type);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_NETWORK_TORCH_UTILS_H_
