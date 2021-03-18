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

#include "tnn/memory_manager/blob_memory_size_info.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

int64_t GetBlobMemoryBytesSize(BlobMemorySizeInfo& size_info) {
    if (size_info.dims.size() == 1) {
        int64_t dims_count = DimsVectorUtils::Count(size_info.dims);
        return dims_count * DataTypeUtils::GetBytesSize(size_info.data_type);

    } else if (size_info.dims.size() == 2) {
        // 2d blob memory with 4 channel
        int64_t dims_count = 1;
        for (auto dim : size_info.dims) {
            dims_count *= dim;
        }
        return dims_count * 4 * DataTypeUtils::GetBytesSize(size_info.data_type);
    } else {
        return 0;
    }
}

}  // namespace TNN_NS
