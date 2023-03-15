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

#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/utils/half_utils.h"
#include "tools/converter/source/resource/reource_base_convert.h"

namespace TNN_CONVERTER {

DECLARE_RESOURCE_CONVERT(Binary);

TNN_NS::Status ResourceBinaryConvert::ConvertToHalfResource(std::shared_ptr<TNN_NS::LayerParam> param,
                                                            std::shared_ptr<TNN_NS::LayerResource> layer_resource) {
    auto resource            = std::dynamic_pointer_cast<TNN_NS::EltwiseLayerResource>(layer_resource);
    resource->element_handle = TNN_NS::ConvertFloatToFP16(resource->element_handle);

    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_RESOURCE_CONVERT(Binary, Add);
REGISTER_RESOURCE_CONVERT(Binary, Sub);
REGISTER_RESOURCE_CONVERT(Binary, Mul);
REGISTER_RESOURCE_CONVERT(Binary, Div);
REGISTER_RESOURCE_CONVERT(Binary, Minimum);
REGISTER_RESOURCE_CONVERT(Binary, Maximum);
REGISTER_RESOURCE_CONVERT(Binary, Less);
REGISTER_RESOURCE_CONVERT(Binary, Greater);
REGISTER_RESOURCE_CONVERT(Binary, And);
REGISTER_RESOURCE_CONVERT(Binary, Not);
REGISTER_RESOURCE_CONVERT(Binary, Square);
REGISTER_RESOURCE_CONVERT(Binary, SquaredDifference);
}  // namespace TNN_CONVERTER
