// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef TNN_SOURCE_DEVICE_ATLAS_ATLAS_COMMON_TYPES_H_
#define TNN_SOURCE_DEVICE_ATLAS_ATLAS_COMMON_TYPES_H_

#include <climits>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>

#include "acl/acl.h"
#include "tnn/core/blob.h"
#include "tnn/core/macro.h"

namespace TNN_NS {

enum class AtlasOmModelDynamicMode {
    Static         = 0,
    DynamicBatch   = 1,
    DynamicHW      = 2,
    GenericDynamic = 3,  // New Dynamic Mode, convert by input_shape_range or input_shape without dynamic dim/hw specified.
};

struct AtlasOMModelInfo {
    aclmdlDesc* model_desc       = nullptr;
    uint32_t model_id            = INT_MAX;
    aclmdlDataset* input_dataset = nullptr;
    aclrtContext aclrt_context   = nullptr;

    size_t memory_size = 0;
    size_t weight_size = 0;

    // Dynamic Input
    AtlasOmModelDynamicMode dynamic_mode;
    std::unordered_set<std::string> generic_dynamic_input_names;

    // AIPP Input
    std::map<std::string, aclAippInputFormat> aipp_input_format_map;
};

extern std::map<Blob*, std::shared_ptr<AtlasOMModelInfo>> global_blob_om_model_info_map;
extern std::map<aclrtStream, aclrtContext> global_stream_context_map;

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_ATLAS_COMMON_TYPES_H_
