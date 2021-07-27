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

#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/core/macro.h"
#include "tnn/network/torch/torch_tensor.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/network/torch/torch_types.h"

#include <torch/script.h>

namespace TNN_NS {

Status GetIValueRouterFromBlob(Blob * blob, IValueRouterPtr &router) {
    ForeignBlob * foreign_blob = dynamic_cast<ForeignBlob*>(blob);
    TNN_CHECK(foreign_blob, "Cast to foreign_blob failed");

    auto foreign_tensor = foreign_blob->GetForeignTensor();
    TNN_CHECK(foreign_tensor, "Got null foreign tensor");

    auto torchtensor = std::dynamic_pointer_cast<TorchTensor>(foreign_tensor);
    TNN_CHECK(torchtensor, "Cast to TorchTensor failed.");

    auto torchtensor_router = torchtensor->GetRouter();
    TNN_CHECK(torchtensor_router, "Got null IValueRouterPtr");
    router = torchtensor_router;

    return TNN_OK;
}

Status SetTensorToBlob(Blob * blob, at::TensorPtr tensor) {
    ForeignBlob * foreign_blob = dynamic_cast<ForeignBlob*>(blob);
    TNN_CHECK(foreign_blob, "Cast to foreign_blob failed");

    auto foreign_tensor = foreign_blob->GetForeignTensor();
    TNN_CHECK(foreign_tensor, "Got null foreign tensor");

    auto torchtensor = std::dynamic_pointer_cast<TorchTensor>(foreign_tensor);
    TNN_CHECK(torchtensor, "Cast to TorchTensor failed.");

    torchtensor->SetTensor(tensor);
    return TNN_OK;
}

}