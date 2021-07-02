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
        default:
            ret = Status(TNNERR_PARAM_ERR, "data_type converting not implemented");
            break;
    }

    return ret;
}

std::vector<int64_t> ConvertDimsToIntArrayRef(DimsVector dims) {
    return std::vector<int64_t>(dims.begin(), dims.end());
}

Status CreateTensorByBlobDesc(std::shared_ptr<torch::Tensor> &tensor, BlobDesc desc) {


    c10::Device device(c10::kCPU);
    RETURN_ON_FAIL(ConvertToTorchDevice(device, desc.device_type));

    at::ScalarType scalar_type;
    RETURN_ON_FAIL(ConvertToTorchDataType(scalar_type, desc.data_type));

    tensor = std::make_shared<torch::Tensor>(at::zeros(
                    ConvertDimsToIntArrayRef(desc.dims), scalar_type, c10::Layout::Strided, device, false));

    return TNN_OK;
}

Status CreateTensorByBlob(std::shared_ptr<torch::Tensor> &tensor, Blob *blob) {
    auto desc = blob->GetBlobDesc();

    c10::Device device(c10::kCPU);
    RETURN_ON_FAIL(ConvertToTorchDevice(device, desc.device_type));

    at::ScalarType scalar_type;
    RETURN_ON_FAIL(ConvertToTorchDataType(scalar_type, desc.data_type));

    tensor = std::make_shared<torch::Tensor>(torch::from_blob(blob->GetHandle().base,
                    ConvertDimsToIntArrayRef(desc.dims), c10::TensorOptions(scalar_type).device(device)));

    return TNN_OK;
}

Status ConvertIValueToTensors(std::vector<torch::Tensor> &tensor, const torch::jit::IValue &ivalue) {
    tensor.resize(0);
    if (ivalue.isTensor()) {
        tensor.push_back(ivalue.toTensor());
    } else {
        return Status(TNNERR_PARAM_ERR, "Converting from Tuple, List or other types are not implemented.");
    }
    return TNN_OK;
}

Status attachTensor(ForeignBlob * blob) {
    if (blob == nullptr)  {
        return TNNERR_NULL_PARAM;
    }

    std::shared_ptr<torch::Tensor> tensor;
    RETURN_ON_FAIL(CreateTensorByBlob(tensor, blob));
    blob->SetForeignTensor(std::make_shared<TorchTensor>(tensor));

    return TNN_OK; 
}

Status ForeignBlobToIValue(torch::IValue &ivalue, ForeignBlob * blob) {
    if (blob == nullptr)  {
        return TNNERR_NULL_PARAM;
    }

    at::Tensor tensor = *std::dynamic_pointer_cast<TorchTensor>(blob->GetForeignTensor())->GetTensor().get();

    // IValue constructor will take ownership of tensor_impl, so we need to clone one.
    // zerocopy for Tensors which created by torch::from_blob 
    ivalue = tensor;

    return TNN_OK; 
}

Status GetTensor(at::Tensor &tensor, Blob * blob) {
    if (blob == nullptr)  {
        return TNNERR_NULL_PARAM;
    }
    auto foreign_blob = dynamic_cast<ForeignBlob*>(blob);
    if (foreign_blob == nullptr) {
        return Status(TNNERR_PARAM_ERR, "not a instance of ForeignBlob.");
    }

    tensor = *std::dynamic_pointer_cast<TorchTensor>(foreign_blob->GetForeignTensor())->GetTensor().get();

    return TNN_OK; 
}

Status GetBlobDescFromTensor(BlobDesc &desc, const torch::Tensor &tensor) {
    auto device = tensor.device();

    RETURN_ON_FAIL(ConvertToDeviceType(desc.device_type, device));
    desc.dims = std::vector<int>(tensor.sizes().begin(), tensor.sizes().end());

    auto scalar_type = tensor.dtype().toScalarType();
    RETURN_ON_FAIL(ConvertToDataType(desc.data_type, scalar_type));

    return TNN_OK; 
}


}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_NETWORK_TORCH_UTILS_H_