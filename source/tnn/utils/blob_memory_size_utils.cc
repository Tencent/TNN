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

#include "tnn/utils/blob_memory_size_utils.h"

#include "tnn/core/common.h"
#include "tnn/core/macro.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

BlobMemorySizeInfo Calculate1DMemorySize(BlobDesc& desc) {
    BlobMemorySizeInfo info;
    info.data_type = desc.data_type;
    int count      = 0;
    if (desc.data_format == DATA_FORMAT_NC4HW4) {
        count = desc.dims[0] * ROUND_UP(desc.dims[1], 4) * desc.dims[2] * desc.dims[3];
    } else if (desc.data_format == DATA_FORMAT_NHWC4) {
        count = desc.dims[0] * ROUND_UP(desc.dims[1], 4) * ROUND_UP(desc.dims[2] * desc.dims[3], 4);
    } else {
        count = DimsVectorUtils::Count(desc.dims);
    }
    info.dims.push_back(count);
    return info;
}

BlobMemorySizeInfo Calculate2DCLImageMemorySize(BlobDesc& desc) {
    BlobMemorySizeInfo info;
    info.data_type = desc.data_type;
    int batch, channel, height, width;
    batch            = desc.dims[0];
    channel          = desc.dims[1];
    height           = desc.dims[2];
    width            = desc.dims[3];
    int image_width  = UP_DIV(channel, 4) * width;
    int image_height = batch * height;
    info.dims.push_back(image_width);
    info.dims.push_back(image_height);
    return info;
}

}  // namespace TNN_NS
