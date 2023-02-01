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

#ifndef TNN_TOOLS_CONVERTER_SOURCE_RESOURCE_RESOURCE_CONVERT_H_
#define TNN_TOOLS_CONVERTER_SOURCE_RESOURCE_RESOURCE_CONVERT_H_

#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"

namespace TNN_CONVERTER {

typedef enum {
    RESOURCE_KEEP_ORIGINAL              = 0,
    RESOURCE_CONVERT_HALF               = 1,
    RESOURCE_CONVERT_FLOAT              = 2,
    RESOURCE_DYNAMIC_RANGE_QUANTIZATION = 3,
} ResourceConvertType;

class ResourceConvert {
public:
    ResourceConvert() = default;
    TNN_NS::Status SetResourceConvertType(ResourceConvertType resource_convert_type);
    TNN_NS::Status converter(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource);

    ~ResourceConvert() = default;

private:
    ResourceConvertType resource_convert_type_ = RESOURCE_KEEP_ORIGINAL;
};
}  // namespace TNN_CONVERTER

#endif  // TNN_TOOLS_CONVERTER_SOURCE_RESOURCE_RESOURCE_CONVERT_H_
