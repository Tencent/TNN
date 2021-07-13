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

#ifndef TNN_SOURCE_TNN_NETWORK_TNNTORCH_TORCH_TENSOR_H_
#define TNN_SOURCE_TNN_NETWORK_TNNTORCH_TORCH_TENSOR_H_

#include <memory>

#include "tnn/core/status.h"
#include "tnn/core/macro.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/network/torch/torch_types.h"

#include <torch/script.h>

namespace TNN_NS {


// @brief Base Type of a Torch Tensor
class TorchTensor : public ForeignTensor {
public:
    explicit TorchTensor() {};

    explicit TorchTensor(at::TensorPtr tensor):tensor_(tensor) {};

    explicit TorchTensor(IValueRouterPtr router):router_(router) {};

    explicit TorchTensor(at::TensorPtr tensor, IValueRouterPtr router):tensor_(tensor), router_(router) {};

    // @brief virtual destructor
    virtual ~TorchTensor() {};

    // @brief get the ITensor
    at::TensorPtr GetTensor() {
        return tensor_;
    }

    Status SetTensor(at::TensorPtr tensor) {
        tensor_ = tensor;
        return TNN_OK;
    }

    IValueRouterPtr GetRouter() {
        return router_;
    }


    Status SetRouter(IValueRouterPtr router) {
        router_ = router;
        return TNN_OK;
    }

private:

    at::TensorPtr tensor_;
    IValueRouterPtr router_;

};

Status GetIValueRouterFromBlob(Blob * blob, IValueRouterPtr &router);

Status SetTensorToBlob(Blob * blob, at::TensorPtr tensor);


}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_NETWORK_TNNTORCH_TORCH_TENSOR_H_W
