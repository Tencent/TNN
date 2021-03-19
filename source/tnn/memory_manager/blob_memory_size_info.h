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

#ifndef TNN_SOURCE_TNN_MEMORY_MANAGER_BLOB_MEMORY_SIZE_INFO_H_
#define TNN_SOURCE_TNN_MEMORY_MANAGER_BLOB_MEMORY_SIZE_INFO_H_

#include <vector>

#include "tnn/core/common.h"

namespace TNN_NS {

// @brief blob memory info data type and data memory dims
struct BlobMemorySizeInfo {
    DataType data_type = DATA_TYPE_FLOAT;
    std::vector<int> dims = {};
};

int64_t GetBlobMemoryBytesSize(BlobMemorySizeInfo& size_info);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_MEMORY_MANAGER_BLOB_MEMORY_SIZE_INFO_H_
