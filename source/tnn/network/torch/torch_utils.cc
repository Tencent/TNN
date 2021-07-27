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

#include "tnn/network/torch/torch_utils.h"

#include <memory>
#include <map>
#include <set>
#include <vector>
#include <regex>
#include <mutex>
#include <stdlib.h>

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

Status GetBlobDescFromTensor(BlobDesc &desc, const torch::Tensor &tensor) {
    auto device = tensor.device();

    RETURN_ON_FAIL(ConvertToDeviceType(desc.device_type, device));
    desc.dims = std::vector<int>(tensor.sizes().begin(), tensor.sizes().end());

    auto scalar_type = tensor.dtype().toScalarType();
    RETURN_ON_FAIL(ConvertToDataType(desc.data_type, scalar_type));

    return TNN_OK; 
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


Status CreateIValueFromTypePtr(c10::IValue &ivalue, c10::TypePtr type) {
    switch(type->kind()) {
        case c10::TypeKind::ListType:
            {
                c10::TypePtr elementType = type->expect<c10::ListType>()->getElementType();
                ivalue = c10::impl::GenericList(elementType);
            }
            break;
        case c10::TypeKind::TupleType:
            {   
                // since the elements in tuple is known and will not change, 
                // here we create the underlying types recursively
                auto contained_types = type->expect<c10::TupleType>()->containedTypes();
                std::vector<c10::IValue> elements(contained_types.size());
                for(int i=0;i< contained_types.size();i++)  {
                    RETURN_ON_FAIL(CreateIValueFromTypePtr(elements[i], contained_types[i]));
                }
                // the ownership of elements will transfer to the created IValue.
                ivalue = c10::IValue(torch::ivalue::Tuple::createNamed(elements, type->expect<c10::TupleType>()));
            }
            break;
        case c10::TypeKind::DictType:
            {   
                c10::TypePtr key_type   = type->expect<c10::DictType>()->getKeyType();
                c10::TypePtr value_type = type->expect<c10::DictType>()->getValueType();
                ivalue = c10::impl::GenericDict(key_type, value_type);
            }
            break;
        case c10::TypeKind::TensorType:
            {
                auto scalar_type = type->expect<c10::TensorType>()->scalarType();
                auto device = type->expect<c10::TensorType>()->device();
                ivalue = at::zeros(std::vector<int64_t>({1}), scalar_type, c10::Layout::Strided, device, false);
            }
            break;
        default:
            return Status(TNNERR_PARAM_ERR, "CreateIValueFromTypePtr unsupported type.");
            break;
    }
    return TNN_OK;
}



}